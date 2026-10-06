"""Entry point for Population Fitter (Python/PyQt5 rewrite).

Installs a global exception hook so that any unexpected error is reported in
a dialog instead of silently crashing or freezing the application -- the
single biggest complaint about the original PopulationGUI.exe.
"""
from __future__ import annotations

import os
import sys
import traceback

# In a PyInstaller --windowed build there is no console, so sys.stdout and
# sys.stderr are None. Any warnings.warn() or print() call anywhere in the
# app or its dependencies would then raise AttributeError on a None stream
# and kill the process instantly with no visible error at all. Redirect them
# to a log file next to the executable before anything else runs.
if sys.stdout is None or sys.stderr is None:
    if getattr(sys, "frozen", False):
        _log_dir = os.path.dirname(sys.executable)
    else:
        _log_dir = os.path.dirname(os.path.abspath(__file__))
    _log_file = open(os.path.join(_log_dir, "PopulationFitter.log"), "w",
                      encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log_file
    sys.stderr = sys.stderr or _log_file

# NOTE: numpy/scipy must be imported before PyQt5/matplotlib. On some Windows
# + Anaconda installs, loading Qt5's bundled DLLs first shadows a symbol that
# scipy.sparse.linalg needs, producing:
#   ImportError: DLL load failed while importing _spropack: The specified
#   procedure could not be found.
# Importing scipy first makes it grab its DLLs before anything else can.
import numpy  # noqa: F401
import scipy.optimize  # noqa: F401

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMessageBox

from main_window import MainWindow


def install_excepthook(app: QApplication) -> None:
    def handle(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            print(text, file=sys.stderr)
            sys.stderr.flush()
        except Exception:
            pass
        try:
            QMessageBox.critical(
                None, "Unexpected error",
                "An unexpected error occurred and was caught before it could "
                f"crash the application:\n\n{exc_value}",
            )
        except Exception:
            pass

    sys.excepthook = handle


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Population Fitter")
    # Raise the app-wide base font a little (Qt's platform default is
    # often ~8-9pt, which reads as cramped on modern high-resolution
    # displays); everything -- labels, buttons, inputs, tables -- inherits
    # this unless a widget/stylesheet overrides it. Section headers and
    # captions get an extra bold/larger treatment on top of this in
    # MainWindow._apply_header_styles().
    font = app.font()
    font.setPointSize(max(10, font.pointSize()))
    app.setFont(font)
    install_excepthook(app)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
