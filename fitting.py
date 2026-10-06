"""Least-squares fitting of an expression's free parameters to target data.

The fit expression may be a full formula (e.g. "a*Curve1 + (1-a)*Curve2") or
simply the designation of an Expression panel; in the latter case its free
parameters are discovered transitively (an Expression panel may itself
reference other Expression panels).

Fitting strategy (the "smart" part): most population-fitting expressions are
*linear* in their free parameters (e.g. "a*Curve1 + (1-a)*Curve2" -- a linear
combination of curves, exactly what the original tool solved via Eigen's
SVD/QR). For that common case we detect linearity numerically (probe the
model at zero and each unit parameter direction, then verify against random
points) and solve it exactly with a single bounded linear least-squares call
-- no initial guess, no iteration count, no local-minima risk, effectively
instant. When the expression is genuinely nonlinear in its parameters (or
log-fitting is on, which makes even a linear model's residual nonlinear), we
fall back to a global search (differential evolution, when bounds are finite)
polished by local refinement, plus several random-start local fits -- far
more robust against bad initial guesses / local minima than a single
scipy.optimize.least_squares run from one starting point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import differential_evolution, least_squares, lsq_linear

from evaluation import interpolate
from expr_eval import ExpressionError, SelfReferenceError, evaluate, parse_expression, referenced_names
from models import Curve, Panel, Variable


@dataclass
class FitResult:
    values: Dict[str, float]
    success: bool
    message: str
    rms: float
    chi2: float
    method: str = ""


def _collect_transitive(tree, panels_by_designation: Dict[str, Panel],
                         variables: Dict[str, Variable],
                         visiting: Set[str] = None) -> Tuple[Set[str], Set[str]]:
    """Walk the expression tree, expanding any referenced Expression panel
    into its own dependencies, and return (leaf file-curve designations,
    free-parameter names) reachable from it."""
    if visiting is None:
        visiting = set()
    curve_leaves: Set[str] = set()
    params: Set[str] = set()
    for nm in referenced_names(tree):
        if nm in variables:
            params.add(nm)
            continue
        panel = panels_by_designation.get(nm)
        if panel is None:
            raise ExpressionError(f"Unknown reference '{nm}'.")
        if not panel.is_expression:
            curve_leaves.add(nm)
            continue
        if nm in visiting:
            raise SelfReferenceError(f"Circular/self reference detected involving '{nm}'.")
        visiting.add(nm)
        sub_tree = parse_expression(panel.expression)
        sub_leaves, sub_params = _collect_transitive(sub_tree, panels_by_designation, variables, visiting)
        curve_leaves |= sub_leaves
        params |= sub_params
        visiting.discard(nm)
    return curve_leaves, params


def discover_expression_parameters(expression: str, panels: List[Panel],
                                    variables: Dict[str, Variable]) -> Set[str]:
    """Return the free-parameter names reachable from `expression`, expanding
    through any referenced Expression panels. Used by the UI to show/let the
    user pick which variables a given fit expression can vary."""
    tree = parse_expression(expression)
    by_designation = {p.designation: p for p in panels}
    _, params = _collect_transitive(tree, by_designation, variables)
    return params


def discover_curve_leaves(expression: str, panels: List[Panel],
                           variables: Dict[str, Variable]) -> Set[str]:
    """Return the File panel (leaf curve) designations transitively
    referenced by `expression`, expanding through any nested Expression
    panels -- i.e. which underlying data files were actually combined to
    produce this expression's curve. Used to report that provenance when
    exporting an expression."""
    tree = parse_expression(expression)
    by_designation = {p.designation: p for p in panels}
    curve_leaves, _ = _collect_transitive(tree, by_designation, variables)
    return curve_leaves


def _eval_ref(name: str, panels_by_designation: Dict[str, Panel],
              curve_env: Dict[str, np.ndarray], param_env: Dict[str, float],
              cache: Dict[str, object]):
    if name in cache:
        return cache[name]
    if name in curve_env:
        return curve_env[name]
    if name in param_env:
        return param_env[name]
    panel = panels_by_designation[name]
    tree = parse_expression(panel.expression)
    env = {
        nm: _eval_ref(nm, panels_by_designation, curve_env, param_env, cache)
        for nm in referenced_names(tree)
    }
    result = evaluate(tree, env)
    cache[name] = result
    return result


def _eval_top(tree, panels_by_designation: Dict[str, Panel],
              curve_env: Dict[str, np.ndarray], param_env: Dict[str, float]):
    cache: Dict[str, object] = {}
    env = {
        nm: _eval_ref(nm, panels_by_designation, curve_env, param_env, cache)
        for nm in referenced_names(tree)
    }
    return evaluate(tree, env)


def _probe_linear(build_model, n_params: int, lo: np.ndarray, hi: np.ndarray,
                   rng: np.random.Generator) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Try to detect that build_model(params) is affine in params: probe at
    zero and each unit parameter direction to construct a candidate (A, b)
    with model(p) == A @ p + b, then verify it against a few random points
    within bounds. Returns (A, b) if verified, else None (genuinely
    nonlinear, or verification failed for any other reason)."""
    zero = np.zeros(n_params)
    b = np.asarray(build_model(zero), dtype=float)
    n_out = b.shape[0]
    A = np.zeros((n_out, n_params))
    for i in range(n_params):
        step = 1.0
        if np.isfinite(lo[i]) and np.isfinite(hi[i]) and hi[i] > lo[i]:
            step = max((hi[i] - lo[i]) / 2.0, 1e-9)
        probe = np.zeros(n_params)
        probe[i] = step
        A[:, i] = (np.asarray(build_model(probe), dtype=float) - b) / step

    finite_bounds = np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo)
    for _ in range(3):
        test_p = rng.uniform(lo, hi) if finite_bounds else rng.uniform(-5.0, 5.0, size=n_params)
        try:
            actual = np.asarray(build_model(test_p), dtype=float)
        except Exception:
            return None
        predicted = A @ test_p + b
        scale = max(1.0, float(np.max(np.abs(actual))))
        if not np.allclose(predicted, actual, rtol=1e-6, atol=1e-8 * scale):
            return None
    return A, b


def _solve_linear(A: np.ndarray, b: np.ndarray, y: np.ndarray,
                   lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exact bounded solve of min ||A@p + b - y|| s.t. lo <= p <= hi."""
    result = lsq_linear(A, y - b, bounds=(lo, hi))
    return result.x


def _robust_nonlinear_fit(residuals_fn, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                           max_iterations: int, rng: np.random.Generator) -> Tuple[object, bool]:
    """Fit via a global search (when bounds are finite and not absurdly
    wide) followed by local polishing, plus several random-start local fits
    as a fallback/complement -- much less prone to a bad local minimum than
    a single least_squares run from one starting point. Returns
    (best least_squares result, whether a global search was used)."""
    candidates = [np.clip(x0, lo, hi)]
    finite_bounds = bool(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)))
    span = (hi - lo) if finite_bounds else None
    tight_bounds = finite_bounds and bool(np.all(span > 0) and np.all(span <= 1.0e6))

    used_global = False
    if tight_bounds:
        try:
            de_result = differential_evolution(
                lambda p: float(np.sum(np.asarray(residuals_fn(p)) ** 2)),
                bounds=list(zip(lo, hi)),
                maxiter=int(np.clip(max_iterations // 10, 50, 200)),
                polish=False,
                seed=0,
                tol=1e-10,
            )
            candidates.append(de_result.x)
            used_global = True
        except Exception:
            pass

    n_random_starts = 6
    for _ in range(n_random_starts):
        if tight_bounds:
            candidates.append(rng.uniform(lo, hi))
        else:
            spread = np.maximum(np.abs(x0), 1.0) * 5.0
            low = x0 - spread
            high = x0 + spread
            if finite_bounds:
                low = np.maximum(low, lo)
                high = np.minimum(high, hi)
            candidates.append(rng.uniform(low, high))

    best_result = None
    best_cost = np.inf
    for cand in candidates:
        cand = np.clip(cand, lo, hi)
        try:
            res = least_squares(residuals_fn, cand, bounds=(lo, hi), max_nfev=max(1, max_iterations))
        except Exception:
            continue
        cost = float(np.sum(res.fun ** 2))
        if cost < best_cost:
            best_cost = cost
            best_result = res
    if best_result is None:
        raise ExpressionError("Fit failed to converge from any starting point.")
    return best_result, used_global


def fit_expression_to_curve(
    expression: str,
    panels: List[Panel],
    variables: Dict[str, Variable],
    target: Curve,
    use_log: bool = False,
    max_iterations: int = 200,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
) -> Tuple[FitResult, Curve]:
    """Fit the free (vary=True) parameters reachable from `expression`
    (directly, or transitively through referenced Expression panels) so that
    the expression, evaluated on target's Q grid, best matches target.i.

    If q_min/q_max are given, only data points with q_min <= Q <= q_max are
    used to compute the fit (RMS/Chi2 are reported over that same restricted
    range) -- but the returned model curve still spans target's *full* Q
    grid, so the fit can be visually compared against the whole experimental
    curve, including outside the fitted range.

    Updates `variables` in place with the fitted values. Returns
    (FitResult, model Curve on the target's full Q grid).
    """
    tree = parse_expression(expression)
    by_designation = {p.designation: p for p in panels}
    curve_leaves, param_names_set = _collect_transitive(tree, by_designation, variables)
    if not curve_leaves:
        raise ExpressionError("Fit expression does not reference any curve.")
    param_names_all = sorted(param_names_set)

    curve_env_full: Dict[str, np.ndarray] = {}
    for nm in curve_leaves:
        dep = by_designation[nm]
        if dep.curve is None:
            raise ExpressionError(f"Panel '{nm}' has no data loaded.")
        curve_env_full[nm] = interpolate(dep.curve, target.q)

    mask = np.ones(target.q.shape, dtype=bool)
    if q_min is not None:
        mask &= target.q >= q_min
    if q_max is not None:
        mask &= target.q <= q_max
    if not np.any(mask):
        raise ExpressionError("No data points fall within the selected Q range.")
    q_fit = target.q[mask]
    y = target.i[mask]
    curve_env_fit = {nm: arr[mask] for nm, arr in curve_env_full.items()}

    fit_names = [n for n in param_names_all if variables[n].vary]
    fixed_params = {n: variables[n].value for n in param_names_all if n not in fit_names}

    def build_model(curve_env: Dict[str, np.ndarray], q_arr: np.ndarray, params_vec) -> np.ndarray:
        param_env = dict(fixed_params)
        param_env.update(zip(fit_names, params_vec))
        result = _eval_top(tree, by_designation, curve_env, param_env)
        return np.broadcast_to(np.asarray(result, dtype=float), q_arr.shape).astype(float)

    def build_model_fit(params_vec) -> np.ndarray:
        return build_model(curve_env_fit, q_fit, params_vec)

    method = "fixed (no free parameters to fit)"
    if not fit_names:
        x = np.array([])
    else:
        x0 = np.array([variables[n].value for n in fit_names], dtype=float)
        lo = np.array([variables[n].min for n in fit_names], dtype=float)
        hi = np.array([variables[n].max for n in fit_names], dtype=float)
        x0 = np.clip(x0, lo, hi)

        if use_log and np.any(y <= 0):
            raise ExpressionError(
                "Log fitting requires strictly positive target intensities "
                "(within the selected Q range, if one is set)."
            )

        rng = np.random.default_rng(0)

        # Fast path: if the model is affine in the free parameters (the
        # common "linear combination of curves" case) and we're not fitting
        # in log-space (which makes even a linear model's residual
        # nonlinear), solve it exactly -- no iteration, no local minima.
        linear = None if use_log else _probe_linear(build_model_fit, len(fit_names), lo, hi, rng)

        if linear is not None:
            A, b = linear
            x = _solve_linear(A, b, y, lo, hi)
            method = "linear (exact closed-form solve)"
        else:
            if use_log:
                y_target = np.log(y)

                def residuals(params_vec: np.ndarray) -> np.ndarray:
                    m = build_model_fit(params_vec)
                    safe_m = np.where(m > 0, m, np.nan)
                    res = np.log(safe_m) - y_target
                    return np.nan_to_num(res, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
            else:
                def residuals(params_vec: np.ndarray) -> np.ndarray:
                    return build_model_fit(params_vec) - y

            result, used_global = _robust_nonlinear_fit(residuals, x0, lo, hi, max_iterations, rng)
            x = result.x
            method = "global search + local refine" if used_global else "multi-start local optimization"

        for n, v in zip(fit_names, x):
            variables[n].value = float(v)

    model_full = build_model(curve_env_full, target.q, x)
    resid = build_model_fit(x) - y
    rms = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else float("nan")
    chi2 = float(np.sum(resid ** 2)) if len(resid) else float("nan")
    range_note = ""
    if q_min is not None or q_max is not None:
        qmin_text = "-inf" if q_min is None else f"{q_min:g}"
        qmax_text = "inf" if q_max is None else f"{q_max:g}"
        range_note = f" (Q range [{qmin_text}, {qmax_text}])"
    fit_res = FitResult(
        values={n: variables[n].value for n in param_names_all},
        success=True,
        message=f"Fit method: {method}{range_note}.",
        rms=rms,
        chi2=chi2,
        method=method,
    )
    return fit_res, Curve(target.q, model_full)


def compute_population_weights(
    expression: str,
    panels: List[Panel],
    variables: Dict[str, Variable],
    q: np.ndarray,
) -> Dict[str, Optional[float]]:
    """For each curve panel directly (or transitively) referenced by
    `expression`, recover its effective linear weight in the model -- i.e.
    verify the model can be written as sum_i(weight_i * curve_i(q)) and
    solve for each weight_i -- by perturbing each curve's values in turn and
    observing the model's response, with the CURRENT (already-fitted)
    variable values held fixed.

    In a well-formed population-fitting expression such as
    "a*Curve1 + (1-a)*Curve2", these weights ARE the fitted population
    (molar) fractions -- recovered generically, without needing to parse
    which literal parameter multiplies which curve. Returns
    {panel_designation: weight}, using None for a curve whose contribution
    isn't a clean constant multiple of itself (the expression doesn't have
    simple weighted-sum-of-curves structure for that curve, so a population
    fraction isn't meaningful there).
    """
    tree = parse_expression(expression)
    by_designation = {p.designation: p for p in panels}
    curve_leaves, param_names = _collect_transitive(tree, by_designation, variables)
    if not curve_leaves:
        return {}

    curve_env: Dict[str, np.ndarray] = {}
    for nm in curve_leaves:
        dep = by_designation[nm]
        if dep.curve is None:
            raise ExpressionError(f"Panel '{nm}' has no data loaded.")
        curve_env[nm] = interpolate(dep.curve, q)

    param_env = {n: variables[n].value for n in param_names if n in variables}

    baseline = np.broadcast_to(
        np.asarray(_eval_top(tree, by_designation, curve_env, param_env), dtype=float), q.shape
    ).astype(float)

    weights: Dict[str, Optional[float]] = {}
    eps = 1.0e-4
    for nm in curve_leaves:
        original = curve_env[nm]
        perturbed_env = dict(curve_env)
        perturbed_env[nm] = original * (1.0 + eps)
        perturbed = np.broadcast_to(
            np.asarray(_eval_top(tree, by_designation, perturbed_env, param_env), dtype=float), q.shape
        ).astype(float)

        diff = perturbed - baseline
        denom = eps * original
        scale = max(1.0, float(np.max(np.abs(original))))
        mask = np.abs(denom) > 1e-12 * scale
        if not np.any(mask):
            weights[nm] = None
            continue
        ratios = diff[mask] / denom[mask]
        model_scale = max(1.0, float(np.max(np.abs(baseline))))
        if np.allclose(ratios, ratios[0], rtol=1e-4, atol=1e-6 * model_scale):
            weights[nm] = float(np.mean(ratios))
        else:
            weights[nm] = None  # not a simple weighted sum for this curve
    return weights


def compute_mass_fractions(
    population_weights: Dict[str, Optional[float]],
    panels: List[Panel],
) -> Dict[str, Optional[float]]:
    """Convert population (molar) weights into mass fractions using each
    panel's subunit count (e.g. 1 for monomer, 2 for dimer, ...):

        mass_fraction_i = (weight_i * subunits_i) / sum_j(weight_j * subunits_j)

    Components with an undefined weight or a non-positive weight*subunits
    product are excluded from the normalization and reported as None.
    """
    by_designation = {p.designation: p for p in panels}
    products: Dict[str, float] = {}
    for name, weight in population_weights.items():
        if weight is None:
            continue
        panel = by_designation.get(name)
        subunits = panel.subunits if panel is not None else 1.0
        products[name] = weight * subunits

    total = sum(v for v in products.values() if v > 0)
    fractions: Dict[str, Optional[float]] = {}
    for name in population_weights:
        product = products.get(name)
        if product is None or product <= 0 or total <= 0:
            fractions[name] = None
        else:
            fractions[name] = product / total
    return fractions


def build_sum_to_one_expression(main_designation: str, other_designations: List[str],
                                 variable_names: List[str]) -> str:
    """Build a linear-combination expression where every component's
    coefficient sums to 1 by construction: the "main" component gets the
    derived coefficient (1 - sum of the others), and every other component
    gets its own free variable --

        (1 - a - b)*Main + a*Other1 + b*Other2

    This is what the "Smart Fit" dialog generates from a component picker,
    so the user never has to hand-write that expansion for 3+ components.
    """
    if len(other_designations) != len(variable_names):
        raise ValueError("other_designations and variable_names must have the same length")
    main_coeff = "(1 - " + " - ".join(variable_names) + ")" if variable_names else "1"
    terms = [f"{main_coeff}*{main_designation}"]
    for var, desig in zip(variable_names, other_designations):
        terms.append(f"{var}*{desig}")
    return " + ".join(terms)
