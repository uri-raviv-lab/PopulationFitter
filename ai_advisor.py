"""Optional AI-powered fit advisor.

Sends a compact, human-readable summary of one completed fit (expression,
parameter values/bounds, RMS/Chi2, fit method, and a downsampled
data/model/residual table) to Anthropic's Claude API and returns its plain-
language commentary: fit quality, whether a parameter looks pinned at a
bound, whether the residuals show a systematic pattern suggesting a missing
component, and suggested next steps.

This module is entirely optional and inert until explicitly invoked: the
rest of the application has no dependency on it, requires no network access,
and works identically whether or not the 'anthropic' package is installed
or an API key is configured. Nothing is sent anywhere automatically -- only
when the user clicks "AI Fit Advisor...".
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from models import Variable

DEFAULT_MODEL = "claude-opus-5"
AVAILABLE_MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]


class AdvisorError(Exception):
    """A user-facing problem obtaining AI advice. Always safe to catch and
    display -- callers should never let this (or anything else from this
    module) propagate into a crash."""


def is_available() -> bool:
    """Whether the optional 'anthropic' package is installed."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _downsample_table(q: np.ndarray, data_i: np.ndarray, model_i: np.ndarray,
                       max_points: int = 40) -> List[str]:
    n = len(q)
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    lines = ["Q\tData_I\tModel_I\tResidual"]
    for i in idx:
        residual = data_i[i] - model_i[i]
        lines.append(f"{q[i]:.6g}\t{data_i[i]:.6g}\t{model_i[i]:.6g}\t{residual:.6g}")
    return lines


def build_summary(
    expression: str,
    variables: Dict[str, Variable],
    param_names: List[str],
    signal_designation: str,
    q: np.ndarray,
    data_i: np.ndarray,
    model_i: np.ndarray,
    rms: float,
    chi2: float,
    method: str,
    use_log: bool,
) -> str:
    """Build the plain-text prompt body describing one fit result."""
    lines = [
        f"Fit expression: {expression}",
        f"Target signal: {signal_designation} ({len(q)} data points, "
        f"Q range [{float(np.min(q)):.4g}, {float(np.max(q)):.4g}])",
        f"Log-space fitting: {'yes' if use_log else 'no'}",
        f"Fit method used: {method or 'unknown'}",
        f"RMS residual: {rms:.6g}" if rms is not None else "RMS residual: n/a",
        f"Sum-of-squares (Chi2): {chi2:.6g}" if chi2 is not None else "Chi2: n/a",
        "",
        "Fit parameters (name, fitted value, bounds, whether it was varied):",
    ]
    for name in param_names:
        var = variables.get(name)
        if var is None:
            continue
        lines.append(
            f"  {name} = {var.value:.6g}  (bounds [{var.min:.6g}, {var.max:.6g}], "
            f"{'varied' if var.vary else 'held fixed'})"
        )
    lines.append("")
    lines.append("Downsampled data / model / residual table (tab-separated):")
    lines.extend(_downsample_table(q, data_i, model_i))
    return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are an expert in small-angle X-ray/neutron scattering (SAXS/SANS) data \
analysis, advising a scientist using a curve-fitting tool. They fit an \
expression (a combination of reference scattering curves with free \
coefficients, often representing population fractions of different \
structural states) to an experimental signal via least squares.

Given a compact summary of one fit (expression, parameter values/bounds, \
RMS/Chi2, fit method, and a downsampled data/model/residual table), give \
brief, concrete feedback covering:
- Overall fit quality assessment.
- Whether any parameter is at or very near its bound (suggests the bound \
should be widened, or the model may be misspecified).
- Whether the residuals show a systematic pattern rather than noise (a \
trend, oscillation, or localized region of large misfit) -- this usually \
means the expression is missing a component or combines the wrong curves.
- One or two concrete, actionable next steps.

Be concise: plain text only, no markdown headers or bullet symbols beyond \
simple dashes, under 250 words. If the fit looks good, say so briefly \
instead of inventing problems.\
"""


def request_advice(summary: str, api_key: Optional[str] = None,
                    model: str = DEFAULT_MODEL) -> str:
    """Synchronous call to the Claude API. Intended to run off the GUI
    thread (it blocks on network I/O). Raises AdvisorError with a clear,
    user-facing message on any failure -- never lets an SDK exception type
    leak out."""
    try:
        import anthropic
    except ImportError as exc:
        raise AdvisorError(
            "The 'anthropic' package is not installed. Install it with "
            "'pip install anthropic' to enable the AI Fit Advisor."
        ) from exc

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001
        raise AdvisorError(f"Could not create the Anthropic client: {exc}") from exc

    try:
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=_SYSTEM_PROMPT,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": summary}],
        )
    except anthropic.BadRequestError as exc:
        raise AdvisorError(f"Request rejected by the API: {exc.message}") from exc
    except anthropic.AuthenticationError as exc:
        raise AdvisorError(
            "Authentication failed -- check the API key in Options > AI Settings... "
            "or the ANTHROPIC_API_KEY environment variable."
        ) from exc
    except anthropic.PermissionDeniedError as exc:
        raise AdvisorError("This API key does not have permission for this request.") from exc
    except anthropic.NotFoundError as exc:
        raise AdvisorError(f"Model '{model}' was not found.") from exc
    except anthropic.RateLimitError as exc:
        raise AdvisorError("Rate limited by the Anthropic API. Try again shortly.") from exc
    except anthropic.APIConnectionError as exc:
        raise AdvisorError(f"Network error reaching the Anthropic API: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise AdvisorError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
    except Exception as exc:  # noqa: BLE001
        raise AdvisorError(f"Unexpected error calling the AI advisor: {exc}") from exc

    if response.stop_reason == "refusal":
        raise AdvisorError("The AI advisor declined to respond to this request.")

    text = "\n".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise AdvisorError("The AI advisor returned an empty response.")
    return text
