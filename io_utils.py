"""Robust file I/O for curve data and exports.

The original PopulationGUI treated any unparsable line as a fatal "Invalid
data file" error and could crash outright on Inf/NaN tokens written by some
instruments (e.g. "1.#INF"). This loader instead skips header/comment lines,
normalizes special float tokens, drops non-finite rows with a warning, and
only raises when there is truly nothing usable in the file.
"""
from __future__ import annotations

import csv
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from models import Curve

_SPECIAL_TOKENS = {
    "1.#inf": "inf", "+1.#inf": "inf", "-1.#inf": "-inf",
    "1.#infinity": "inf", "-1.#infinity": "-inf",
    "1.#qnan": "nan", "1.#ind": "nan", "-1.#ind": "nan", "nan(ind)": "nan",
    "infinity": "inf", "-infinity": "-inf",
}

MAX_DATA_COLUMNS = 3  # Q, I, [sigma]


class DataFileError(Exception):
    """Raised when a curve data file cannot be parsed at all."""


def _normalize_token(tok: str) -> str:
    return _SPECIAL_TOKENS.get(tok.lower(), tok)


def _parse_row(line: str) -> Optional[Tuple[float, ...]]:
    line = line.strip()
    if not line or line.startswith(("#", ";", "!", "//", "\"")):
        return None
    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        return None
    values = []
    for p in parts[:MAX_DATA_COLUMNS]:
        try:
            values.append(float(_normalize_token(p)))
        except ValueError:
            return None
    return tuple(values)


def load_curve_file(path: str) -> Tuple[Curve, List[str]]:
    """Load a 2+ column whitespace/tab/comma separated data file (Q, I[, sigma]).

    Returns (curve, warnings). Raises DataFileError only if no usable data
    could be extracted at all.
    """
    if not os.path.isfile(path):
        raise DataFileError(f"File not found: {path}")

    rows: List[Tuple[float, ...]] = []
    skipped_nonnumeric = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = _parse_row(line)
                if parsed is None:
                    skipped_nonnumeric += 1
                    continue
                rows.append(parsed)
    except OSError as exc:
        raise DataFileError(f"Could not read '{path}':\n{exc}") from exc

    if not rows:
        raise DataFileError(
            f"'{os.path.basename(path)}' contains no readable numeric data "
            f"(expected whitespace/tab/comma separated columns)."
        )

    ncols = min(len(r) for r in rows)
    data = np.array([r[:ncols] for r in rows], dtype=float)
    q, i = data[:, 0], data[:, 1]

    finite = np.isfinite(q) & np.isfinite(i)
    n_bad = int((~finite).sum())
    q, i = q[finite], i[finite]

    warnings: List[str] = []
    if n_bad:
        warnings.append(
            f"{n_bad} row(s) with non-finite (Inf/NaN) values were skipped "
            f"in '{os.path.basename(path)}'."
        )
    if len(q) < 2:
        raise DataFileError(
            f"'{os.path.basename(path)}' has fewer than 2 valid data points "
            f"after removing non-finite rows."
        )

    curve = Curve(q, i).sorted()
    return curve, warnings


def export_curve_tsv(path: str, q: Sequence[float], i: Sequence[float],
                      header: str = "Q\tI", comments: Sequence[str] = ()) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            for comment in comments:
                f.write(f"# {comment}\n")
            f.write(header + "\n")
            for qq, ii in zip(q, i):
                f.write(f"{qq:.6g}\t{ii:.6g}\n")
    except OSError as exc:
        raise DataFileError(
            f"Error writing file '{path}'.\n"
            f"Please make sure the file is not open in another program.\n({exc})"
        ) from exc


def export_table_tsv(path: str, headers: Sequence[str], rows: Sequence[Sequence],
                      comments: Sequence[str] = ()) -> None:
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            for comment in comments:
                f.write(f"# {comment}\n")
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(headers)
            writer.writerows(rows)
    except OSError as exc:
        raise DataFileError(
            f"Error writing file '{path}'.\n"
            f"Please make sure the file is not open in another program.\n({exc})"
        ) from exc
