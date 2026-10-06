"""Save/load a complete application session -- every panel (curves and
expressions, including their actual loaded data), every variable, every
loaded signal (series) with its fit results, and a few UI settings -- to a
single JSON file.

Curve data is embedded directly rather than just referencing the original
file path, so a saved session stays fully reproducible even if the source
data files are later moved, renamed, or no longer available (the path is
still recorded, for provenance, but is never required to reload).
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from models import Curve, Panel, SeriesEntry, Variable

SESSION_FORMAT = "population-fitter-session"
SESSION_VERSION = 1


class SessionError(Exception):
    """Raised for any problem saving/loading a session file."""


def _curve_to_json(curve: Curve) -> dict:
    if curve is None:
        return None
    return {"q": curve.q.tolist(), "i": curve.i.tolist()}


def _curve_from_json(data) -> Curve:
    if data is None:
        return None
    return Curve(np.array(data["q"], dtype=float), np.array(data["i"], dtype=float))


def _panel_to_json(panel: Panel) -> dict:
    return {
        "designation": panel.designation,
        "color": panel.color,
        "visible": panel.visible,
        "is_expression": panel.is_expression,
        "filepath": panel.filepath,
        "show_filename": panel.show_filename,
        "expression": panel.expression,
        "subunits": panel.subunits,
        "curve": _curve_to_json(panel.curve),
    }


def _panel_from_json(data: dict) -> Panel:
    panel = Panel(
        designation=data.get("designation") or "Panel",
        color=data.get("color") or "#1f77b4",
        visible=bool(data.get("visible", True)),
        is_expression=bool(data.get("is_expression", False)),
        filepath=data.get("filepath"),
        show_filename=bool(data.get("show_filename", False)),
        expression=data.get("expression") or "",
        subunits=float(data.get("subunits", 1.0)),
    )
    panel.curve = _curve_from_json(data.get("curve"))
    return panel


def _series_to_json(entry: SeriesEntry) -> dict:
    return {
        "filepath": entry.filepath,
        "designation": entry.designation,
        "curve": _curve_to_json(entry.curve),
        "model": _curve_to_json(entry.model),
        "rms": entry.rms,
        "chi2": entry.chi2,
        "fit_values": entry.fit_values,
        "fit_expression": entry.fit_expression,
        "fit_method": entry.fit_method,
        "population_weights": entry.population_weights,
        "mass_fractions": entry.mass_fractions,
    }


def _series_from_json(data: dict) -> SeriesEntry:
    entry = SeriesEntry(
        filepath=data.get("filepath") or "",
        designation=data.get("designation") or "signal",
    )
    entry.curve = _curve_from_json(data.get("curve"))
    entry.model = _curve_from_json(data.get("model"))
    entry.rms = data.get("rms")
    entry.chi2 = data.get("chi2")
    entry.fit_values = data.get("fit_values") or {}
    entry.fit_expression = data.get("fit_expression")
    entry.fit_method = data.get("fit_method")
    entry.population_weights = data.get("population_weights") or {}
    entry.mass_fractions = data.get("mass_fractions") or {}
    return entry


def _variable_to_json(var: Variable) -> dict:
    return {"value": var.value, "min": var.min, "max": var.max, "vary": var.vary}


def _variable_from_json(name: str, data: dict) -> Variable:
    return Variable(
        name=name,
        value=float(data.get("value", 1.0)),
        min=float(data.get("min", -1.0e12)),
        max=float(data.get("max", 1.0e12)),
        vary=bool(data.get("vary", True)),
    )


def save_session(path: str, panels: List[Panel], variables: Dict[str, Variable],
                  series: List[SeriesEntry], settings: dict) -> None:
    payload = {
        "format": SESSION_FORMAT,
        "version": SESSION_VERSION,
        "panels": [_panel_to_json(p) for p in panels],
        "variables": {name: _variable_to_json(v) for name, v in variables.items()},
        "series": [_series_to_json(e) for e in series],
        "settings": settings,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
    except OSError as exc:
        raise SessionError(
            f"Error writing session file '{path}'.\n"
            f"Please make sure the file is not open in another program.\n({exc})"
        ) from exc


def load_session(path: str) -> Tuple[List[Panel], Dict[str, Variable], List[SeriesEntry], dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except OSError as exc:
        raise SessionError(f"Could not read session file '{path}':\n{exc}") from exc
    except json.JSONDecodeError as exc:
        raise SessionError(f"'{path}' is not a valid session file (invalid JSON): {exc}") from exc

    if not isinstance(payload, dict) or payload.get("format") != SESSION_FORMAT:
        raise SessionError(f"'{path}' does not look like a Population Fitter session file.")

    try:
        panels = [_panel_from_json(p) for p in payload.get("panels", [])]
        variables = {
            name: _variable_from_json(name, data)
            for name, data in (payload.get("variables") or {}).items()
        }
        series = [_series_from_json(e) for e in payload.get("series", [])]
        settings = payload.get("settings") or {}
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionError(f"Session file '{path}' is malformed: {exc}") from exc

    return panels, variables, series, settings
