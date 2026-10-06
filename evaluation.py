"""Resolves the dependency graph between file/expression panels and produces
concrete Curve objects for every panel, including expressions that reference
other expressions. Detects cycles/self-references explicitly instead of
recursing until a stack overflow (a crash source in the original)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from expr_eval import ExpressionError, SelfReferenceError, evaluate, parse_expression, referenced_names
from models import Curve, Panel, Variable


def _common_grid(curves: List[Curve]) -> np.ndarray:
    lo = max(c.q.min() for c in curves)
    hi = min(c.q.max() for c in curves)
    if lo >= hi:
        raise ExpressionError("Referenced curves do not overlap in Q range.")
    npts = max(len(c) for c in curves)
    return np.linspace(lo, hi, npts)


def interpolate(curve: Curve, q: np.ndarray) -> np.ndarray:
    return np.interp(q, curve.q, curve.i)


def evaluate_all(panels: List[Panel], variables: Dict[str, Variable]):
    """Evaluate every panel's curve, resolving expression dependencies in
    topological order. Returns (resolved, errors):
      resolved: {panel.id: Curve} for every panel that resolved successfully
      errors:   {panel.id: message} for every panel that failed (missing
                data, unknown reference, non-overlapping Q range, self
                reference, ...)

    One panel's failure never hides another, independent panel: each
    panel's dependency subtree is evaluated and errors are attributed on a
    per-panel basis, so e.g. a single broken/incomplete Expression panel
    doesn't blank out every other expression's curve on the plot."""
    by_designation = {p.designation: p for p in panels}
    resolved: Dict[int, Curve] = {}
    errors: Dict[int, str] = {}
    visiting: set = set()

    def resolve(panel: Panel) -> Curve:
        if panel.id in resolved:
            return resolved[panel.id]
        if panel.id in errors:
            raise ExpressionError(errors[panel.id])
        if panel.id in visiting:
            raise SelfReferenceError(
                f"Circular/self reference detected involving '{panel.designation}'."
            )
        visiting.add(panel.id)
        try:
            if not panel.is_expression:
                if panel.curve is None:
                    raise ExpressionError(f"Panel '{panel.designation}' has no data loaded.")
                curve = panel.curve
            else:
                tree = parse_expression(panel.expression)
                names = referenced_names(tree)
                param_names = names & set(variables)
                curve_names = names - param_names
                dep_curves = []
                for nm in curve_names:
                    dep_panel = by_designation.get(nm)
                    if dep_panel is None:
                        raise ExpressionError(f"Unknown reference '{nm}' in '{panel.designation}'.")
                    if dep_panel.id == panel.id:
                        raise SelfReferenceError(
                            f"Expression '{panel.designation}' references itself."
                        )
                    dep_curves.append(resolve(dep_panel))
                if dep_curves:
                    grid = _common_grid(dep_curves)
                    env = {nm: interpolate(resolve(by_designation[nm]), grid) for nm in curve_names}
                else:
                    raise ExpressionError(
                        f"Expression '{panel.designation}' does not reference any curve."
                    )
                env.update({nm: variables[nm].value for nm in param_names})
                result = evaluate(tree, env)
                result = np.broadcast_to(np.asarray(result, dtype=float), grid.shape)
                curve = Curve(grid, np.array(result, dtype=float))
        finally:
            # Always clean up, even on failure -- otherwise a panel that
            # raised mid-resolution stays stuck in `visiting`, which would
            # make an unrelated LATER panel that happens to reference the
            # same dependency incorrectly fail with a bogus self-reference
            # error of its own.
            visiting.discard(panel.id)

        resolved[panel.id] = curve
        return curve

    for p in panels:
        if p.id in resolved or p.id in errors:
            continue
        try:
            resolve(p)
        except (ExpressionError, SelfReferenceError) as exc:
            errors[p.id] = str(exc)

    return resolved, errors


def collect_parameter_names(panels: List[Panel], curve_designations: set) -> set:
    """All identifiers used in expressions that are not other panels'
    designations -- i.e. candidate free parameters for the variable table."""
    names: set = set()
    for p in panels:
        if p.is_expression and p.expression.strip():
            try:
                tree = parse_expression(p.expression)
            except ExpressionError:
                continue
            names |= referenced_names(tree) - curve_designations
    return names
