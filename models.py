"""Data models for Population Fitter: curves, panels, variables, series."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_color_cycle_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_color_cycle = itertools.cycle(_color_cycle_colors)


def next_color() -> str:
    return next(_color_cycle)


@dataclass
class Curve:
    """A scattering curve: Q (x) and I (y) arrays of equal length."""

    q: np.ndarray
    i: np.ndarray

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float)
        self.i = np.asarray(self.i, dtype=float)
        if self.q.shape != self.i.shape:
            raise ValueError("Q and I arrays must have the same length")

    def __len__(self):
        return len(self.q)

    def sorted(self) -> "Curve":
        order = np.argsort(self.q)
        return Curve(self.q[order], self.i[order])


@dataclass
class Variable:
    """A free parameter used inside expressions (a fit coefficient)."""

    name: str
    value: float = 1.0
    min: float = -1.0e12
    max: float = 1.0e12
    vary: bool = True


_panel_ids = itertools.count(1)


@dataclass
class Panel:
    """Shared state for a panel that is either a loaded file or an
    expression combining other panels. Mirrors the original
    FileExpressionPanel, which could be toggled between the two modes."""

    designation: str
    color: str = field(default_factory=next_color)
    visible: bool = True
    is_expression: bool = False
    filepath: Optional[str] = None
    show_filename: bool = False
    expression: str = ""
    curve: Optional[Curve] = None
    id: int = field(default_factory=lambda: next(_panel_ids))
    subunits: float = 1.0

    def label(self) -> str:
        if not self.is_expression and self.show_filename and self.filepath:
            return self.filepath
        return self.designation


@dataclass
class SeriesEntry:
    """One target data file loaded for batch fitting."""

    filepath: str
    designation: str
    curve: Optional[Curve] = None
    model: Optional[Curve] = None
    rms: Optional[float] = None
    chi2: Optional[float] = None
    error: Optional[str] = None
    fit_values: dict = field(default_factory=dict)
    fit_expression: Optional[str] = None
    fit_method: Optional[str] = None
    # Per-component (panel designation) population/molar fraction and, once
    # subunit counts are known, mass fraction -- see fitting.compute_population_weights.
    population_weights: dict = field(default_factory=dict)
    mass_fractions: dict = field(default_factory=dict)
