"""Main window for Population Fitter: panel list, variable table, series
table/fitting controls, and the plot -- assembled from the pieces in
models.py, io_utils.py, expr_eval.py, evaluation.py, fitting.py and
panel_widgets.py."""
from __future__ import annotations

import copy
import functools
import os
import traceback
from typing import List, Optional, Set

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import QEvent, QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

import ai_advisor
import session_io
from evaluation import evaluate_all
from expr_eval import ExpressionError, is_valid_designation
from fitting import (
    build_sum_to_one_expression, compute_mass_fractions, compute_population_weights,
    discover_curve_leaves, discover_expression_parameters, fit_expression_to_curve,
)
from io_utils import DataFileError, export_curve_tsv, export_table_tsv, load_curve_file
from models import Curve, Panel, SeriesEntry, Variable
from panel_widgets import FileExpressionPanelWidget
from session_io import SessionError

SERIES_BASE_COLUMNS = ["File", "Designation", "RMS", "Chi2"]
DROPPABLE_EXTENSIONS = {".out", ".dat", ".txt", ".chi"}
DATA_FILE_FILTER = "Data Files (*.out *.dat *.chi *.txt);;All files (*.*)"


def _caption(text: str) -> QLabel:
    """A short field-label QLabel (e.g. 'Expression:', 'Signal to fit:')
    tagged so the app-wide stylesheet can make captions larger/bold without
    affecting word-wrapped description paragraphs or hint/preview labels,
    which keep their own look."""
    label = QLabel(text)
    label.setObjectName("FieldCaption")
    return label


def safe_slot(fn):
    """Wrap a Qt slot so an unexpected exception shows a dialog instead of
    propagating into the Qt event loop (where it could freeze or kill the
    app)."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except (ExpressionError, DataFileError, SessionError) as exc:
            QMessageBox.warning(self, "Population Fitter", str(exc))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            QMessageBox.critical(
                self, "Unexpected error",
                f"{exc}\n\nThe application caught this error and remains open."
            )
        return None

    return wrapper


class _AdvisorWorker(QThread):
    """Runs ai_advisor.request_advice() off the GUI thread so a slow/blocked
    network call never freezes the window -- the exact failure mode we
    fixed everywhere else in this app."""

    succeeded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, summary: str, api_key: str, model: str, parent=None):
        super().__init__(parent)
        self.summary = summary
        self.api_key = api_key
        self.model = model

    def run(self):
        try:
            text = ai_advisor.request_advice(
                self.summary, api_key=self.api_key or None, model=self.model
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced via signal, never crashes the thread
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(text)


class AISettingsDialog(QDialog):
    def __init__(self, current_key: str, current_model: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Settings")
        self.resize(420, 160)
        layout = QVBoxLayout(self)

        info = QLabel(
            "The AI Fit Advisor sends a compact summary of a fit (expression, "
            "parameter values/bounds, RMS/Chi2, and a downsampled data table) "
            "to Anthropic's Claude API and shows its commentary. This needs "
            "network access and an API key, and only runs when you click "
            "'AI Fit Advisor...' -- nothing is sent automatically."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        key_row = QHBoxLayout()
        key_row.addWidget(_caption("API key:"))
        self.key_edit = QLineEdit(current_key)
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("leave blank to use the ANTHROPIC_API_KEY env var")
        key_row.addWidget(self.key_edit)
        layout.addLayout(key_row)

        model_row = QHBoxLayout()
        model_row.addWidget(_caption("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(ai_advisor.AVAILABLE_MODELS)
        idx = self.model_combo.findText(current_model)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        return self.key_edit.text().strip(), self.model_combo.currentText()


def _generate_free_variable_names(count: int, taken: Set[str]) -> List[str]:
    """Pick `count` short identifiers (a, b, c, ... then a2, b2, ... if more
    than 26 are ever needed) that don't collide with anything in `taken`.
    Mutates `taken` in place as each name is claimed."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    names: List[str] = []
    i = 0
    while len(names) < count:
        letter = letters[i % 26]
        suffix = i // 26 + 1
        candidate = letter if suffix == 1 else f"{letter}{suffix}"
        i += 1
        if candidate in taken or not is_valid_designation(candidate):
            continue
        names.append(candidate)
        taken.add(candidate)
    return names


class SmartFitDialog(QDialog):
    """Lets the user pick which loaded reference curves are components of a
    mixture and which one is the 'main' component, then previews the
    auto-built sum-to-1 expression: (1-a-b)*Main + a*Other1 + b*Other2."""

    def __init__(self, candidate_panels: List[Panel], taken_names: Set[str],
                 default_designation: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Smart Fit")
        self.resize(440, 380)
        self._candidate_panels = candidate_panels
        self._taken_names = taken_names

        layout = QVBoxLayout(self)
        info = QLabel(
            "Pick which loaded curves are components of the mixture, and "
            "which one is the 'main' component. The main component's "
            "coefficient is derived as 1 minus the others, so the "
            "population fractions always sum to 1."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Created before any checkbox/radio below so that _update_preview()
        # -- triggered by their toggled signals, including the initial
        # setChecked(True) calls during construction -- always has a live
        # widget to write into.
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("font-family: monospace;")

        self.checkboxes = {}
        self.radios = {}
        self.main_group = QButtonGroup(self)
        for panel in self._candidate_panels:
            row = QHBoxLayout()
            cb = QCheckBox(panel.designation)
            cb.setChecked(True)
            radio = QRadioButton("main")
            radio.setEnabled(True)
            cb.toggled.connect(radio.setEnabled)
            cb.toggled.connect(self._update_preview)
            radio.toggled.connect(self._update_preview)
            self.main_group.addButton(radio)
            self.checkboxes[panel.designation] = cb
            self.radios[panel.designation] = radio
            row.addWidget(cb)
            row.addWidget(radio)
            layout.addLayout(row)

        if self._candidate_panels:
            self.radios[self._candidate_panels[0].designation].setChecked(True)

        name_row = QHBoxLayout()
        name_row.addWidget(_caption("New panel name:"))
        self.designation_edit = QLineEdit(default_designation)
        name_row.addWidget(self.designation_edit)
        layout.addLayout(name_row)

        layout.addWidget(self.preview_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_preview()

    def _selected_designations(self) -> List[str]:
        return [name for name, cb in self.checkboxes.items() if cb.isChecked()]

    def _main_designation(self) -> Optional[str]:
        for name, radio in self.radios.items():
            if radio.isChecked() and self.checkboxes[name].isChecked():
                return name
        return None

    def _update_preview(self, *_):
        selected = self._selected_designations()
        main = self._main_designation()
        if main is None or len(selected) < 2:
            self.preview_label.setText("(select at least 2 components and a main component)")
            return
        others = [n for n in selected if n != main]
        var_names = _generate_free_variable_names(len(others), set(self._taken_names))
        try:
            expr = build_sum_to_one_expression(main, others, var_names)
        except Exception:  # noqa: BLE001 -- preview only, never fatal
            expr = ""
        self.preview_label.setText(f"Expression: {expr}")

    def result_data(self):
        """Returns (main_designation, other_designations, new_panel_name),
        or (None, [], ...) if the selection is incomplete."""
        selected = self._selected_designations()
        main = self._main_designation()
        if main is None or len(selected) < 2:
            return None, [], self.designation_edit.text().strip()
        others = [n for n in selected if n != main]
        return main, others, self.designation_edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Population Fitter (Python)")
        self.resize(1500, 900)

        self.panels: list[Panel] = []
        self.panel_widgets: dict[int, FileExpressionPanelWidget] = {}
        self.variables: dict[str, Variable] = {}
        self.series: list[SeriesEntry] = []
        self.use_log_fitting = False
        self.fit_iterations = 200
        self._updating_table = False
        self._panel_counter = 0
        # Drop targets referenced by eventFilter(); declared before any
        # widget is built because Qt can dispatch internal events (and thus
        # call eventFilter) on a widget the moment installEventFilter() runs
        # on it, which happens partway through _build_central() -- before
        # both boxes necessarily exist yet.
        self.curves_box = None
        self.fit_box = None

        self._build_menu()
        self._build_central()
        self._apply_header_styles()
        self.statusBar().showMessage(
            "Ready. Drag & drop .out/.dat/.txt/.chi files onto Curves && "
            "Expressions to add reference curves, or onto Fit to load a signal."
        )

    def _apply_header_styles(self):
        """App-wide look: bold/larger section headers (Curves && Expressions,
        Variables, Fit), table headers and field captions (Expression:,
        Signal to fit:, Name:, ...), and a consistently larger, more legible
        size for every interactive control (buttons, text inputs, combo
        boxes, spin boxes, the menu bar) -- rather than the previous pass
        which only touched a few widget types and left everything else at
        Qt's cramped platform-default size.

        `QGroupBox#SectionGroupBox` gets explicit top margin/padding: with a
        13pt bold title and no extra headroom, the title can visually
        collide with whatever sits directly above the box in the layout
        (here, the menu bar) -- Qt does not automatically grow a group box's
        margin to fit a larger title.
        """
        self.setStyleSheet(
            self.styleSheet() + "\n"
            "QMenuBar {"
            "  font-size: 10pt;"
            "  padding: 2px;"
            "}\n"
            "QGroupBox#SectionGroupBox {"
            "  margin-top: 18px;"
            "  padding-top: 6px;"
            "  font-size: 10pt;"
            "}\n"
            "QGroupBox#SectionGroupBox::title {"
            "  font-size: 14pt;"
            "  font-weight: bold;"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  left: 6px;"
            "  padding: 2px 6px;"
            "}\n"
            "QHeaderView::section {"
            "  font-size: 10pt;"
            "  font-weight: bold;"
            "  padding: 4px;"
            "}\n"
            "QLabel#FieldCaption {"
            "  font-size: 11pt;"
            "  font-weight: bold;"
            "}\n"
            "QPushButton {"
            "  font-size: 10pt;"
            "  padding: 5px 12px;"
            "}\n"
            "QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {"
            "  font-size: 10pt;"
            "  padding: 3px 4px;"
            "}\n"
            "QTableWidget {"
            "  font-size: 10pt;"
            "}\n"
            "QCheckBox, QRadioButton {"
            "  font-size: 10pt;"
            "}\n"
        )

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        save_session_act = QAction("Save Session...", self)
        save_session_act.triggered.connect(self.action_save_session)
        file_menu.addAction(save_session_act)

        load_session_act = QAction("Load Session...", self)
        load_session_act.triggered.connect(self.action_load_session)
        file_menu.addAction(load_session_act)

        file_menu.addSeparator()

        add_panel_act = QAction("Add Panel", self)
        add_panel_act.triggered.connect(self.action_add_panel)
        file_menu.addAction(add_panel_act)

        add_many_act = QAction("Add Multiple Panels...", self)
        add_many_act.triggered.connect(self.action_add_multiple_panels)
        file_menu.addAction(add_many_act)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("Export")
        export_table_act = QAction("Export Series Table...", self)
        export_table_act.triggered.connect(self.action_export_series_table)
        export_menu.addAction(export_table_act)

        export_curves_act = QAction("Export Series Curves...", self)
        export_curves_act.triggered.connect(self.action_export_series_curves)
        export_menu.addAction(export_curves_act)

        export_graph_act = QAction("Export Graph as Image...", self)
        export_graph_act.triggered.connect(self.action_export_graph)
        export_menu.addAction(export_graph_act)

        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        options_menu = self.menuBar().addMenu("&Options")
        self.log_fitting_act = QAction("Use Log Fitting", self)
        self.log_fitting_act.setCheckable(True)
        self.log_fitting_act.toggled.connect(self._on_log_fitting_toggled)
        options_menu.addAction(self.log_fitting_act)

        fit_iter_act = QAction("Fit Iterations...", self)
        fit_iter_act.triggered.connect(self.action_set_fit_iterations)
        options_menu.addAction(fit_iter_act)

        ai_settings_act = QAction("AI Settings...", self)
        ai_settings_act.triggered.connect(self.action_ai_settings)
        options_menu.addAction(ai_settings_act)

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(self._build_panel_list_box())
        left_splitter.addWidget(self._build_variable_box())
        left_splitter.addWidget(self._build_fit_box())
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)
        left_splitter.setStretchFactor(2, 3)

        splitter.addWidget(left_splitter)
        splitter.addWidget(self._build_plot_box())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        left_splitter.setMinimumWidth(420)
        splitter.setSizes([560, 940])

    def _build_panel_list_box(self) -> QWidget:
        box = QGroupBox("Curves && Expressions")
        box.setObjectName("SectionGroupBox")
        box.setAcceptDrops(True)
        box.installEventFilter(self)
        self.curves_box = box
        layout = QVBoxLayout(box)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Panel")
        add_btn.clicked.connect(self.action_add_panel)
        add_many_btn = QPushButton("Add Multiple Panels...")
        add_many_btn.clicked.connect(self.action_add_multiple_panels)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(add_many_btn)
        layout.addLayout(btn_row)

        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_container = QWidget()
        self.panel_list_layout = QVBoxLayout(self.panel_container)
        self.panel_list_layout.addStretch()
        self.panel_scroll.setWidget(self.panel_container)
        layout.addWidget(self.panel_scroll)
        return box

    def _build_variable_box(self) -> QWidget:
        box = QGroupBox("Variables (fit coefficients)")
        box.setObjectName("SectionGroupBox")
        layout = QVBoxLayout(box)
        self.variable_table = QTableWidget(0, 5)
        self.variable_table.setHorizontalHeaderLabels(
            ["Variable", "Value", "Min", "Max", "Fit"]
        )
        self.variable_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.variable_table.cellChanged.connect(self._on_variable_cell_changed)
        layout.addWidget(self.variable_table)
        return box

    def _build_fit_box(self) -> QWidget:
        box = QGroupBox("Fit")
        box.setObjectName("SectionGroupBox")
        box.setAcceptDrops(True)
        box.installEventFilter(self)
        self.fit_box = box
        layout = QVBoxLayout(box)

        signal_row = QHBoxLayout()
        signal_row.addWidget(_caption("Signal to fit:"))
        self.signal_combo = QComboBox()
        self.signal_combo.setToolTip(
            "The loaded series (target/signal) curve this fit will be matched "
            "against. Drop a .out/.dat/.txt/.chi file here, or use "
            "'Load series...' below, to add one."
        )
        self.signal_combo.currentIndexChanged.connect(self._on_signal_combo_changed)
        signal_row.addWidget(self.signal_combo, 1)
        layout.addLayout(signal_row)

        expr_row = QHBoxLayout()
        expr_row.addWidget(_caption("Expression:"))
        self.fit_expression_combo = QComboBox()
        self.fit_expression_combo.setEditable(True)
        self.fit_expression_combo.setInsertPolicy(QComboBox.NoInsert)
        self.fit_expression_combo.setToolTip(
            "The expression to fit -- pick an Expression panel's name, or "
            "type any formula directly (e.g. a*Curve1 + (1-a)*Curve2)."
        )
        self.fit_expression_combo.currentTextChanged.connect(self._on_fit_expression_changed)
        expr_row.addWidget(self.fit_expression_combo, 1)
        smart_fit_btn = QPushButton("Smart Fit...")
        smart_fit_btn.setToolTip(
            "Pick which loaded curves are components of the mixture and "
            "which one is the 'main' component; auto-builds a sum-to-1 "
            "expression (e.g. (1-a-b)*Main + a*Other1 + b*Other2) so you "
            "don't have to hand-write it for 3+ components."
        )
        smart_fit_btn.clicked.connect(self.action_smart_fit)
        expr_row.addWidget(smart_fit_btn)
        layout.addLayout(expr_row)

        qrange_row = QHBoxLayout()
        qrange_row.addWidget(_caption("Q range:"))
        self.qmin_edit = QLineEdit()
        self.qmin_edit.setPlaceholderText("min (blank = no limit)")
        self.qmax_edit = QLineEdit()
        self.qmax_edit.setPlaceholderText("max (blank = no limit)")
        qrange_tip = (
            "Restrict fitting to data points with Q in this range (leave "
            "either side blank for no limit on that side). The fitted model "
            "curve is still shown/exported over the signal's full Q range."
        )
        self.qmin_edit.setToolTip(qrange_tip)
        self.qmax_edit.setToolTip(qrange_tip)
        qrange_row.addWidget(self.qmin_edit)
        qrange_row.addWidget(QLabel("to"))
        qrange_row.addWidget(self.qmax_edit)
        layout.addLayout(qrange_row)

        layout.addWidget(_caption("Variables to fit:"))
        self.fit_vars_container = QWidget()
        self.fit_vars_layout = QHBoxLayout(self.fit_vars_container)
        self.fit_vars_layout.setContentsMargins(0, 0, 0, 0)
        self.fit_vars_hint = QLabel("(choose an expression)")
        self.fit_vars_hint.setStyleSheet("color: gray;")
        self.fit_vars_layout.addWidget(self.fit_vars_hint)
        self.fit_vars_layout.addStretch()
        layout.addWidget(self.fit_vars_container)

        series_btn_row = QHBoxLayout()
        load_series_btn = QPushButton("Load series...")
        load_series_btn.clicked.connect(self.action_load_series)
        remove_series_btn = QPushButton("Remove Selected")
        remove_series_btn.clicked.connect(self.action_remove_selected_series)
        series_btn_row.addWidget(load_series_btn)
        series_btn_row.addWidget(remove_series_btn)
        layout.addLayout(series_btn_row)

        self.series_table = QTableWidget(0, len(SERIES_BASE_COLUMNS))
        self.series_table.setHorizontalHeaderLabels(SERIES_BASE_COLUMNS)
        self.series_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.series_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.series_table.itemSelectionChanged.connect(self._on_series_table_selection_changed)
        layout.addWidget(self.series_table)

        fit_btn_row = QHBoxLayout()
        fit_selected_btn = QPushButton("Fit Selected Signal")
        fit_selected_btn.setToolTip(
            "Fit the checked variables of the chosen expression to the chosen "
            "signal, and update the global Variables table."
        )
        fit_selected_btn.clicked.connect(self.action_fit_selected)
        fit_all_btn = QPushButton("Fit All (independently)")
        fit_all_btn.setToolTip(
            "Fit the expression to every loaded series row independently "
            "(each row gets its own coefficients, e.g. per-sample "
            "populations); the global Variables table is left untouched."
        )
        fit_all_btn.clicked.connect(self.action_fit_all)
        fit_btn_row.addWidget(fit_selected_btn)
        fit_btn_row.addWidget(fit_all_btn)
        layout.addLayout(fit_btn_row)

        self.ai_advisor_btn = QPushButton("AI Fit Advisor...")
        self.ai_advisor_btn.setToolTip(
            "Send this fit's expression, parameters, and residuals to Claude "
            "for a plain-language quality check and suggestions. Requires "
            "network access and an API key (Options > AI Settings...)."
        )
        self.ai_advisor_btn.clicked.connect(self.action_ai_advisor)
        layout.addWidget(self.ai_advisor_btn)

        return box

    def _build_plot_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)

        self.figure = Figure(figsize=(5, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        controls = QHBoxLayout()
        self.logq_check = QCheckBox("log(Q)")
        self.logi_check = QCheckBox("log(I)")
        self.logi_check.setChecked(True)
        self.logq_check.toggled.connect(self.refresh_plot)
        self.logi_check.toggled.connect(self.refresh_plot)
        controls.addWidget(self.logq_check)
        controls.addWidget(self.logi_check)
        controls.addStretch()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        controls.addWidget(reset_btn)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(controls)
        return box

    # ------------------------------------------------------------- panels
    def _new_designation(self) -> str:
        while True:
            self._panel_counter += 1
            name = f"Curve{self._panel_counter}"
            if is_valid_designation(name) and name not in {p.designation for p in self.panels}:
                return name

    def _add_panel_widget(self, panel: Panel):
        widget = FileExpressionPanelWidget(panel)
        widget.changed.connect(self._on_panels_changed)
        widget.removed.connect(self.action_remove_panel)
        widget.export_requested.connect(self.action_export_panel)
        self.panel_list_layout.insertWidget(self.panel_list_layout.count() - 1, widget)
        self.panel_widgets[panel.id] = widget

    @safe_slot
    def action_add_panel(self, *_):
        panel = Panel(designation=self._new_designation())
        self.panels.append(panel)
        self._add_panel_widget(panel)
        self._on_panels_changed()

    @safe_slot
    def action_add_multiple_panels(self, *_):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select files to be added", "", DATA_FILE_FILTER,
        )
        if not paths:
            return
        self._add_file_panels(paths)

    def _add_file_panels(self, paths: list) -> int:
        """Load each path as a reference-curve File panel. Returns how many
        were added successfully. Shared by 'Add Multiple Panels...' and
        drag & drop onto the Curves && Expressions box."""
        failures = []
        added = 0
        for path in paths:
            try:
                curve, warnings = load_curve_file(path)
            except DataFileError as exc:
                failures.append(str(exc))
                continue
            panel = Panel(designation=self._new_designation(), filepath=path, curve=curve)
            self.panels.append(panel)
            self._add_panel_widget(panel)
            self.panel_widgets[panel.id]._load_from_panel()
            added += 1
        if failures:
            QMessageBox.warning(
                self, "Invalid data file(s)",
                "One or more of the chosen files is invalid or empty and "
                "has been ignored:\n\n" + "\n".join(failures),
            )
        if added:
            self._on_panels_changed()
        return added

    @safe_slot
    def action_remove_panel(self, panel_id: int):
        widget = self.panel_widgets.pop(panel_id, None)
        if widget is not None:
            self.panel_list_layout.removeWidget(widget)
            widget.deleteLater()
        self.panels = [p for p in self.panels if p.id != panel_id]
        self._on_panels_changed()

    def _signal_path_comments(self) -> list:
        """Full paths of every currently loaded signal (series/target)
        file -- embedded as provenance in every export so an output file is
        self-describing about what session produced it."""
        if not self.series:
            return []
        lines = ["Loaded signal files:"]
        for entry in self.series:
            lines.append(f"  {entry.designation}: {entry.filepath}")
        return lines

    def _component_path_comments(self, expression: str) -> list:
        """Full paths of the reference-curve File panels that `expression`
        actually combines (transitively, through any nested Expression
        panels) -- i.e. provenance for which source data files went into
        this exported curve. Distinct from _signal_path_comments(), which
        lists the *target* signals loaded for fitting, not the components
        that make up an expression."""
        try:
            leaf_names = discover_curve_leaves(expression, self.panels, self.variables)
        except ExpressionError:
            return []
        if not leaf_names:
            return []
        by_designation = {p.designation: p for p in self.panels}
        lines = ["Component files:"]
        for name in sorted(leaf_names):
            panel = by_designation.get(name)
            path = panel.filepath if panel is not None and panel.filepath else "(no file loaded)"
            lines.append(f"  {name}: {path}")
        return lines

    def _parameter_value_comments(self, param_names) -> list:
        lines = []
        for name in sorted(param_names):
            var = self.variables.get(name)
            if var is None:
                continue
            lines.append(
                f"  {name} = {var.value:.6g}  (bounds [{var.min:.6g}, {var.max:.6g}], "
                f"{'varied' if var.vary else 'held fixed'})"
            )
        if lines:
            lines.insert(0, "Parameters:")
        return lines

    def _mass_fraction_lines(self, population_weights: dict, mass_fractions: dict) -> list:
        """Format a 'Mass fractions:' comment block, including a total-sum
        line over the components that got a defined mass fraction (should
        be ~1 if every contributing component is accounted for; less than 1
        if some component's weight couldn't be resolved -- see
        fitting.compute_population_weights)."""
        if not mass_fractions:
            return []
        lines = ["Mass fractions (population fraction x subunit count, normalized):"]
        total = 0.0
        any_valid = False
        for name in sorted(mass_fractions):
            frac = mass_fractions[name]
            pop = population_weights.get(name)
            pop_text = f"{pop:.6g}" if pop is not None else "n/a"
            frac_text = f"{frac:.6g}" if frac is not None else "n/a"
            lines.append(f"  {name}: population fraction = {pop_text}, mass fraction = {frac_text}")
            if frac is not None:
                total += frac
                any_valid = True
        if any_valid:
            lines.append(f"  Sum of mass fractions = {total:.6g}")
        return lines

    def _mass_fraction_comments(self, entry: SeriesEntry) -> list:
        return self._mass_fraction_lines(entry.population_weights, entry.mass_fractions)

    @safe_slot
    def action_export_panel(self, panel_id: int):
        panel = next((p for p in self.panels if p.id == panel_id), None)
        if panel is None:
            return
        resolved, errors = evaluate_all(self.panels, self.variables)
        curve = resolved.get(panel_id)
        if curve is None:
            message = errors.get(panel_id, "This panel has no data to export.")
            QMessageBox.warning(self, "Export", message)
            return
        comments = []
        if panel.is_expression and panel.expression.strip():
            comments.append(f"Expression: {panel.expression}")
            try:
                param_names = discover_expression_parameters(panel.expression, self.panels, self.variables)
            except ExpressionError:
                param_names = set()
            comments.extend(self._parameter_value_comments(param_names))
            try:
                weights = compute_population_weights(panel.expression, self.panels, self.variables, curve.q)
                mass_fractions = compute_mass_fractions(weights, self.panels)
                comments.extend(self._mass_fraction_lines(weights, mass_fractions))
            except ExpressionError:
                pass
            comments.extend(self._component_path_comments(panel.expression))
        else:
            if panel.filepath:
                comments.append(f"Source: {panel.filepath}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output file as", f"{panel.designation}.out",
            "Output Files (*.out);;Tab Separated Values (*.tsv);;All files (*.*)",
        )
        if not path:
            return
        export_curve_tsv(path, curve.q, curve.i, comments=comments)
        self.statusBar().showMessage(f"Exported '{panel.designation}' to {path}", 5000)

    def _on_panels_changed(self):
        self._refresh_fit_expression_combo()
        self._refresh_variables()
        self._refresh_series_table()
        self._refresh_fit_variables()
        self.refresh_plot()

    # ---------------------------------------------------------- variables
    def _refresh_variables(self):
        from evaluation import collect_parameter_names

        curve_designations = {p.designation for p in self.panels}
        needed = collect_parameter_names(self.panels, curve_designations)
        fit_expr = self.fit_expression_combo.currentText().strip()
        if fit_expr:
            try:
                needed |= discover_expression_parameters(fit_expr, self.panels, self.variables)
            except ExpressionError:
                pass
        for name in needed:
            if name not in self.variables:
                self.variables[name] = Variable(name=name)
        for stale in set(self.variables) - needed:
            del self.variables[stale]

        self._updating_table = True
        try:
            names = sorted(self.variables)
            self.variable_table.setRowCount(len(names))
            for row, name in enumerate(names):
                var = self.variables[name]
                name_item = QTableWidgetItem(name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.variable_table.setItem(row, 0, name_item)
                self.variable_table.setItem(row, 1, QTableWidgetItem(f"{var.value:g}"))
                self.variable_table.setItem(row, 2, QTableWidgetItem(f"{var.min:g}"))
                self.variable_table.setItem(row, 3, QTableWidgetItem(f"{var.max:g}"))
                check = QTableWidgetItem()
                check.setFlags((check.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
                check.setCheckState(Qt.Checked if var.vary else Qt.Unchecked)
                self.variable_table.setItem(row, 4, check)
        finally:
            self._updating_table = False

    @safe_slot
    def _on_variable_cell_changed(self, row: int, col: int):
        if self._updating_table:
            return
        name_item = self.variable_table.item(row, 0)
        if name_item is None:
            return
        var = self.variables.get(name_item.text())
        if var is None:
            return
        if col == 4:
            item = self.variable_table.item(row, 4)
            var.vary = item.checkState() == Qt.Checked
            return
        item = self.variable_table.item(row, col)
        try:
            value = float(item.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid value", f"'{item.text()}' is not a number.")
            self._refresh_variables()
            return
        if col == 1:
            var.value = value
        elif col == 2:
            var.min = value
        elif col == 3:
            var.max = value
        if var.min > var.max:
            QMessageBox.warning(self, "Invalid range", "Min must not exceed Max.")
            self._refresh_variables()
            return
        self.refresh_plot()

    # -------------------------------------------------------------- plot
    def refresh_plot(self, *_):
        self.ax.clear()
        resolved, errors = evaluate_all(self.panels, self.variables)

        any_plotted = False
        for panel in self.panels:
            if not panel.visible:
                continue
            curve = resolved.get(panel.id)
            if curve is None:
                continue
            self.ax.plot(curve.q, curve.i, color=panel.color, label=panel.label(), linewidth=1.5)
            any_plotted = True

        for entry in self.series:
            if entry.curve is not None:
                self.ax.plot(
                    entry.curve.q, entry.curve.i, linestyle="none", marker="o",
                    markersize=3, alpha=0.6, label=f"{entry.designation} (data)",
                )
                any_plotted = True
            if entry.model is not None:
                self.ax.plot(
                    entry.model.q, entry.model.i, linestyle="--", linewidth=1,
                    label=f"{entry.designation} (fit)",
                )

        self.ax.set_xscale("log" if self.logq_check.isChecked() else "linear")
        self.ax.set_yscale("log" if self.logi_check.isChecked() else "linear")
        self.ax.set_xlabel("Q", fontsize=11)
        self.ax.set_ylabel("Intensity [a.u.]", fontsize=11)
        self.ax.tick_params(axis="both", labelsize=10)
        if any_plotted:
            self.ax.legend(fontsize=10)
        self.canvas.draw_idle()

        if errors:
            failing = [p for p in self.panels if p.id in errors]
            first = failing[0]
            extra = f" (+{len(failing) - 1} more)" if len(failing) > 1 else ""
            self.statusBar().showMessage(f"'{first.designation}'{extra}: {errors[first.id]}", 8000)
        else:
            self.statusBar().clearMessage()

    def _reset_view(self):
        self.ax.relim()
        self.ax.autoscale()
        self.canvas.draw_idle()

    def _on_log_fitting_toggled(self, checked: bool):
        self.use_log_fitting = checked

    # ---------------------------------------------------- fit expression UI
    def _refresh_fit_expression_combo(self):
        """Repopulate the Expression combo from current Expression panels,
        preserving whatever text/selection the user already has (including
        an ad-hoc typed formula that isn't any panel's name)."""
        current = self.fit_expression_combo.currentText()
        self.fit_expression_combo.blockSignals(True)
        self.fit_expression_combo.clear()
        for p in self.panels:
            if p.is_expression and p.designation:
                self.fit_expression_combo.addItem(p.designation)
        if current:
            idx = self.fit_expression_combo.findText(current)
            if idx >= 0:
                self.fit_expression_combo.setCurrentIndex(idx)
            else:
                self.fit_expression_combo.setEditText(current)
        self.fit_expression_combo.blockSignals(False)

    def _on_fit_expression_changed(self, _text: str = ""):
        self._refresh_variables()
        self._refresh_fit_variables()

    def _refresh_fit_variables(self):
        """Rebuild the 'Variables to fit' checkboxes for whichever expression
        is currently chosen, so the user can explicitly pick which variables
        this fit should vary (vs. hold fixed)."""
        for i in reversed(range(self.fit_vars_layout.count())):
            item = self.fit_vars_layout.itemAt(i)
            widget = item.widget()
            if widget is not None and widget is not self.fit_vars_hint:
                self.fit_vars_layout.removeWidget(widget)
                widget.deleteLater()

        expr = self.fit_expression_combo.currentText().strip()
        if not expr:
            self.fit_vars_hint.setText("(choose an expression)")
            self.fit_vars_hint.show()
            return
        try:
            params = discover_expression_parameters(expr, self.panels, self.variables)
        except ExpressionError as exc:
            self.fit_vars_hint.setText(f"(invalid expression: {exc})")
            self.fit_vars_hint.show()
            return
        if not params:
            self.fit_vars_hint.setText("(no free variables)")
            self.fit_vars_hint.show()
            return
        self.fit_vars_hint.hide()

        insert_at = self.fit_vars_layout.count() - 1  # before the trailing stretch
        for name in sorted(params):
            var = self.variables.get(name)
            checkbox = QCheckBox(name)
            checkbox.setChecked(var.vary if var else True)
            checkbox.toggled.connect(functools.partial(self._on_fit_var_toggled, name))
            self.fit_vars_layout.insertWidget(max(insert_at, 0), checkbox)
            insert_at += 1

    def _on_fit_var_toggled(self, name: str, checked: bool):
        if name in self.variables:
            self.variables[name].vary = checked
        self._refresh_variables()  # keeps the main Variables table's Fit column in sync

    @safe_slot
    def action_set_fit_iterations(self, *_):
        value, ok = QInputDialog.getInt(
            self, "Fit Iterations", "Maximum fit iterations:",
            self.fit_iterations, 1, 1_000_000,
        )
        if ok:
            self.fit_iterations = value

    @safe_slot
    def action_export_graph(self, *_):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph As", "graph.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;"
            "GIF Image (*.gif);;Bitmap Image (*.bmp);;All files (*.*)",
        )
        if not path:
            return
        self.figure.savefig(path, dpi=150)
        self.statusBar().showMessage(f"Graph saved to {path}", 5000)

    # ------------------------------------------------------------- series
    def _refresh_series_table(self):
        var_cols = sorted({name for entry in self.series for name in entry.fit_values})
        mass_cols = sorted({name for entry in self.series for name in entry.mass_fractions})
        headers = SERIES_BASE_COLUMNS + var_cols + [f"MassFrac:{n}" for n in mass_cols]
        self.series_table.setColumnCount(len(headers))
        self.series_table.setHorizontalHeaderLabels(headers)
        self.series_table.setRowCount(len(self.series))
        for row, entry in enumerate(self.series):
            self.series_table.setItem(row, 0, QTableWidgetItem(os.path.basename(entry.filepath)))
            self.series_table.setItem(row, 1, QTableWidgetItem(entry.designation))
            rms_text = f"{entry.rms:.6g}" if entry.rms is not None else ""
            chi2_text = f"{entry.chi2:.6g}" if entry.chi2 is not None else ""
            self.series_table.setItem(row, 2, QTableWidgetItem(rms_text))
            self.series_table.setItem(row, 3, QTableWidgetItem(chi2_text))
            fit_values = entry.fit_values
            col = 4
            for name in var_cols:
                text = f"{fit_values[name]:.6g}" if name in fit_values else ""
                self.series_table.setItem(row, col, QTableWidgetItem(text))
                col += 1
            for name in mass_cols:
                frac = entry.mass_fractions.get(name)
                text = f"{frac:.4g}" if frac is not None else ""
                self.series_table.setItem(row, col, QTableWidgetItem(text))
                col += 1
        self._refresh_signal_combo()

    def _refresh_signal_combo(self):
        current = self.signal_combo.currentText() if self.signal_combo.count() else None
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        for entry in self.series:
            self.signal_combo.addItem(entry.designation)
        if current:
            idx = self.signal_combo.findText(current)
            if idx >= 0:
                self.signal_combo.setCurrentIndex(idx)
        self.signal_combo.blockSignals(False)

    def _on_signal_combo_changed(self, index: int):
        if 0 <= index < len(self.series):
            self.series_table.blockSignals(True)
            self.series_table.selectRow(index)
            self.series_table.blockSignals(False)

    def _on_series_table_selection_changed(self):
        idx = self._selected_series_index()
        if idx is not None and 0 <= idx < self.signal_combo.count():
            if self.signal_combo.currentIndex() != idx:
                self.signal_combo.blockSignals(True)
                self.signal_combo.setCurrentIndex(idx)
                self.signal_combo.blockSignals(False)

    @safe_slot
    def action_load_series(self, *_):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select files to be fit to", "", DATA_FILE_FILTER,
        )
        if not paths:
            return
        self._add_series_files(paths)

    def _add_series_files(self, paths: list) -> int:
        """Load each path as a series (target/signal) entry. Returns how many
        were added successfully. Shared by 'Load series...' and drag & drop."""
        failures = []
        added = 0
        for path in paths:
            try:
                curve, warnings = load_curve_file(path)
            except DataFileError as exc:
                failures.append(str(exc))
                continue
            designation = os.path.splitext(os.path.basename(path))[0]
            entry = SeriesEntry(filepath=path, designation=designation, curve=curve)
            self.series.append(entry)
            added += 1
        if failures:
            QMessageBox.warning(
                self, "Invalid data file(s)",
                "One or more of the chosen files is invalid or empty and "
                "has been ignored:\n\n" + "\n".join(failures),
            )
        if added:
            self._auto_fill_fit_expression()
        self._refresh_series_table()
        if added and self.series:
            idx = self.signal_combo.findText(self.series[-1].designation)
            if idx >= 0:
                self.signal_combo.setCurrentIndex(idx)
        self.refresh_plot()
        return added

    def _auto_fill_fit_expression(self):
        """If no fit expression is set yet and an Expression panel exists,
        point the fit box at it so a dropped/loaded signal can be fit right
        away with no extra setup."""
        if self.fit_expression_combo.currentText().strip():
            return
        expr_panels = [p for p in self.panels if p.is_expression and p.expression.strip()]
        if expr_panels:
            self.fit_expression_combo.setCurrentText(expr_panels[-1].designation)

    # --------------------------------------------------------- drag & drop
    # Drops are location-aware: files dropped on "Curves && Expressions"
    # become reference-curve File panels; files dropped on "Fit" become
    # series (signal) entries to fit against. Implemented via an event
    # filter (rather than overriding dragEnterEvent/dropEvent on each
    # target) so both group boxes can share the same handling code.
    def _extract_dropped_paths(self, event) -> list:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
        return [p for p in paths if os.path.splitext(p)[1].lower() in DROPPABLE_EXTENSIONS]

    def eventFilter(self, obj, event) -> bool:
        # NOTE: deliberately not wrapped with @safe_slot -- PyQt expects a
        # real bool back from an overridden eventFilter, and returning None
        # (what safe_slot yields after catching an exception) risks a type
        # error inside Qt's own event dispatch. Catch and report manually.
        if obj not in (self.curves_box, self.fit_box):
            return super().eventFilter(obj, event)

        et = event.type()
        try:
            if et == QEvent.DragEnter:
                paths = self._extract_dropped_paths(event)
                if paths:
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return True

            if et == QEvent.Drop:
                paths = self._extract_dropped_paths(event)
                if not paths:
                    return False
                event.acceptProposedAction()
                if obj is self.curves_box:
                    added = self._add_file_panels(paths)
                    if added:
                        self.statusBar().showMessage(f"Loaded {added} reference curve(s).", 6000)
                else:  # self.fit_box
                    added = self._add_series_files(paths)
                    if added:
                        self.statusBar().showMessage(f"Loaded {added} signal file(s) for fitting.", 6000)
                return True
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            QMessageBox.critical(
                self, "Unexpected error",
                f"{exc}\n\nThe application caught this error and remains open."
            )
            return True

        return super().eventFilter(obj, event)

    @safe_slot
    def action_remove_selected_series(self, *_):
        rows = sorted({idx.row() for idx in self.series_table.selectedIndexes()}, reverse=True)
        for row in rows:
            del self.series[row]
        self._refresh_series_table()
        self.refresh_plot()

    def _selected_series_index(self) -> Optional[int]:
        rows = sorted({idx.row() for idx in self.series_table.selectedIndexes()})
        if rows:
            return rows[0]
        if len(self.series) == 1:
            return 0
        return None

    @safe_slot
    def action_smart_fit(self, *_):
        candidate_panels = [p for p in self.panels if not p.is_expression and p.curve is not None]
        if len(candidate_panels) < 2:
            QMessageBox.information(
                self, "Smart Fit",
                "Load at least two reference curves (File panels) before using Smart Fit.",
            )
            return
        taken = set(self.variables) | {p.designation for p in self.panels}
        default_designation = self._new_designation()
        dlg = SmartFitDialog(candidate_panels, taken, default_designation, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        main, others, new_designation = dlg.result_data()
        if main is None or not others:
            QMessageBox.information(
                self, "Smart Fit", "Select at least two components and a main component.",
            )
            return
        if not new_designation:
            new_designation = default_designation
        elif not is_valid_designation(new_designation) or new_designation in {p.designation for p in self.panels}:
            QMessageBox.warning(
                self, "Smart Fit", f"'{new_designation}' is not a valid, unused panel name.",
            )
            return

        var_names = _generate_free_variable_names(len(others), taken)
        expr = build_sum_to_one_expression(main, others, var_names)

        n_components = len(others) + 1
        for name in var_names:
            self.variables[name] = Variable(name=name, value=1.0 / n_components, min=0.0, max=1.0, vary=True)

        panel = Panel(designation=new_designation, is_expression=True, expression=expr)
        self.panels.append(panel)
        self._add_panel_widget(panel)
        self._on_panels_changed()
        self.fit_expression_combo.setCurrentText(new_designation)
        self.statusBar().showMessage(f"Smart Fit created '{new_designation}': {expr}", 8000)

    def _get_q_range(self):
        """Parse the Q-range fields. Returns (q_min, q_max), each either a
        float or None (no limit on that side). Raises ExpressionError (shown
        as a warning dialog by @safe_slot) for non-numeric text or an
        inverted range."""

        def parse(edit: QLineEdit, label: str):
            text = edit.text().strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                raise ExpressionError(f"Q range {label} must be a number (or left blank).")

        q_min = parse(self.qmin_edit, "min")
        q_max = parse(self.qmax_edit, "max")
        if q_min is not None and q_max is not None and q_min > q_max:
            raise ExpressionError("Q range min must not exceed Q range max.")
        return q_min, q_max

    @safe_slot
    def action_fit_selected(self, *_):
        if not self.series:
            QMessageBox.information(
                self, "Fit",
                "Load a signal to fit to first (Load series... or drag & "
                "drop a file onto the Fit panel).",
            )
            return
        idx = self.signal_combo.currentIndex()
        if idx < 0 or idx >= len(self.series):
            QMessageBox.information(self, "Fit", "Choose a signal to fit to.")
            return
        expr = self.fit_expression_combo.currentText().strip()
        if not expr:
            QMessageBox.information(self, "Fit not initialized", "Choose or type a fit expression.")
            return
        q_min, q_max = self._get_q_range()
        entry = self.series[idx]
        fit_res, model = fit_expression_to_curve(
            expr, self.panels, self.variables, entry.curve,
            use_log=self.use_log_fitting, max_iterations=self.fit_iterations,
            q_min=q_min, q_max=q_max,
        )
        entry.model = model
        entry.rms = fit_res.rms
        entry.chi2 = fit_res.chi2
        entry.fit_values = dict(fit_res.values)
        entry.fit_expression = expr
        entry.fit_method = fit_res.method
        self._compute_fractions(entry, expr, self.variables)
        self._refresh_variables()
        self._refresh_series_table()
        self.refresh_plot()
        self.statusBar().showMessage(fit_res.message, 8000)

    def _compute_fractions(self, entry: SeriesEntry, expr: str, variables: dict):
        """Recover each referenced component's population (molar) fraction
        and, using its panel's subunit count, its mass fraction. Never lets
        a failure here affect the fit result itself -- worst case the
        fractions are just left empty."""
        try:
            weights = compute_population_weights(expr, self.panels, variables, entry.curve.q)
            entry.population_weights = weights
            entry.mass_fractions = compute_mass_fractions(weights, self.panels)
        except ExpressionError:
            entry.population_weights = {}
            entry.mass_fractions = {}

    @safe_slot
    def action_fit_all(self, *_):
        if not self.series:
            QMessageBox.information(
                self, "Fit",
                "Load a signal to fit to first (Load series... or drag & "
                "drop a file onto the Fit panel).",
            )
            return
        expr = self.fit_expression_combo.currentText().strip()
        if not expr:
            QMessageBox.information(self, "Fit not initialized", "Choose or type a fit expression.")
            return
        q_min, q_max = self._get_q_range()
        errors = []
        for entry in self.series:
            local_vars = copy.deepcopy(self.variables)
            try:
                fit_res, model = fit_expression_to_curve(
                    expr, self.panels, local_vars, entry.curve,
                    use_log=self.use_log_fitting, max_iterations=self.fit_iterations,
                    q_min=q_min, q_max=q_max,
                )
            except (ExpressionError, DataFileError) as exc:
                errors.append(f"{entry.designation}: {exc}")
                continue
            entry.model = model
            entry.rms = fit_res.rms
            entry.chi2 = fit_res.chi2
            entry.fit_values = dict(fit_res.values)
            entry.fit_expression = expr
            entry.fit_method = fit_res.method
            self._compute_fractions(entry, expr, local_vars)
        self._refresh_series_table()
        self.refresh_plot()
        if errors:
            QMessageBox.warning(self, "Some fits failed", "\n".join(errors))
        else:
            self.statusBar().showMessage("Fit complete for all series.", 8000)

    @safe_slot
    def action_export_series_table(self, *_):
        if not self.series:
            QMessageBox.information(self, "Export", "No series loaded to export.")
            return
        var_cols = sorted({name for entry in self.series for name in entry.fit_values})
        mass_cols = sorted({name for entry in self.series for name in entry.mass_fractions})
        headers = ["File Path"] + SERIES_BASE_COLUMNS + var_cols + [f"MassFrac:{n}" for n in mass_cols]
        rows = []
        for entry in self.series:
            fit_values = entry.fit_values
            row = [
                entry.filepath,
                os.path.basename(entry.filepath),
                entry.designation,
                f"{entry.rms:.6g}" if entry.rms is not None else "",
                f"{entry.chi2:.6g}" if entry.chi2 is not None else "",
            ]
            row += [f"{fit_values[n]:.6g}" if n in fit_values else "" for n in var_cols]
            row += [
                f"{entry.mass_fractions[n]:.6g}" if entry.mass_fractions.get(n) is not None else ""
                for n in mass_cols
            ]
            rows.append(row)
        comments = self._signal_path_comments()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output file as", "series_table.tsv",
            "Tab Separated Values (*.tsv);;All files (*.*)",
        )
        if not path:
            return
        export_table_tsv(path, headers, rows, comments=comments)
        self.statusBar().showMessage(f"Series table exported to {path}", 5000)

    @safe_slot
    def action_export_series_curves(self, *_):
        if not self.series:
            QMessageBox.information(self, "Export", "No series loaded to export.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose folder for exported curves")
        if not directory:
            return
        signal_paths = self._signal_path_comments()
        for entry in self.series:
            if entry.curve is not None:
                data_comments = [f"Source: {entry.filepath}"] + signal_paths
                export_curve_tsv(
                    os.path.join(directory, f"{entry.designation}_data.out"),
                    entry.curve.q, entry.curve.i,
                    comments=data_comments,
                )
            if entry.model is not None:
                comments = []
                if entry.fit_expression:
                    comments.append(f"Expression: {entry.fit_expression}")
                if entry.fit_method:
                    comments.append(f"Fit method: {entry.fit_method}")
                if entry.rms is not None:
                    comments.append(f"RMS: {entry.rms:.6g}")
                if entry.chi2 is not None:
                    comments.append(f"Chi2: {entry.chi2:.6g}")
                comments.extend(self._parameter_value_comments(entry.fit_values.keys()))
                comments.extend(self._mass_fraction_comments(entry))
                if entry.fit_expression:
                    comments.extend(self._component_path_comments(entry.fit_expression))
                comments.extend(signal_paths)
                export_curve_tsv(
                    os.path.join(directory, f"{entry.designation}_model.out"),
                    entry.model.q, entry.model.i,
                    comments=comments,
                )
        self.statusBar().showMessage(f"Series curves exported to {directory}", 5000)

    # --------------------------------------------------------- AI advisor
    @safe_slot
    def action_ai_settings(self, *_):
        settings = QSettings("PopulationFitter", "PopulationFitter")
        current_key = settings.value("ai/api_key", "", type=str)
        current_model = settings.value("ai/model", ai_advisor.DEFAULT_MODEL, type=str)
        dlg = AISettingsDialog(current_key, current_model, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            key, model = dlg.values()
            settings.setValue("ai/api_key", key)
            settings.setValue("ai/model", model)
            self.statusBar().showMessage("AI settings saved.", 4000)

    @safe_slot
    def action_ai_advisor(self, *_):
        if not ai_advisor.is_available():
            QMessageBox.information(
                self, "AI Fit Advisor",
                "The 'anthropic' package is not installed. Install it with "
                "'pip install anthropic' to enable this feature.",
            )
            return
        idx = self.signal_combo.currentIndex()
        if idx < 0 or idx >= len(self.series):
            QMessageBox.information(self, "AI Fit Advisor", "Choose a signal to fit to first.")
            return
        entry = self.series[idx]
        if entry.model is None or not entry.fit_expression:
            QMessageBox.information(
                self, "AI Fit Advisor",
                "Fit this signal first (Fit Selected Signal), then ask the advisor.",
            )
            return

        param_names = sorted(
            discover_expression_parameters(entry.fit_expression, self.panels, self.variables)
        )
        summary = ai_advisor.build_summary(
            expression=entry.fit_expression,
            variables=self.variables,
            param_names=param_names,
            signal_designation=entry.designation,
            q=entry.curve.q, data_i=entry.curve.i, model_i=entry.model.i,
            rms=entry.rms, chi2=entry.chi2,
            method=entry.fit_method or "",
            use_log=self.use_log_fitting,
        )

        settings = QSettings("PopulationFitter", "PopulationFitter")
        api_key = settings.value("ai/api_key", "", type=str)
        model = settings.value("ai/model", ai_advisor.DEFAULT_MODEL, type=str)

        self.ai_advisor_btn.setEnabled(False)
        self.statusBar().showMessage("Asking the AI advisor... (this may take a few seconds)")
        self._ai_worker = _AdvisorWorker(summary, api_key, model, parent=self)
        self._ai_worker.succeeded.connect(self._on_ai_advice_ready)
        self._ai_worker.failed.connect(self._on_ai_advice_failed)
        self._ai_worker.start()

    def _on_ai_advice_ready(self, text: str):
        self.ai_advisor_btn.setEnabled(True)
        self.statusBar().showMessage("AI advisor responded.", 5000)
        dlg = QDialog(self)
        dlg.setWindowTitle("AI Fit Advisor")
        dlg.resize(520, 420)
        layout = QVBoxLayout(dlg)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        layout.addWidget(view)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    def _on_ai_advice_failed(self, message: str):
        self.ai_advisor_btn.setEnabled(True)
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "AI Fit Advisor", message)

    # -------------------------------------------------------------- session
    @safe_slot
    def action_save_session(self, *_):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "session.pfsession",
            "Population Fitter Session (*.pfsession);;JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        settings = {
            "use_log_fitting": self.use_log_fitting,
            "fit_iterations": self.fit_iterations,
            "fit_expression": self.fit_expression_combo.currentText(),
            "signal_designation": self.signal_combo.currentText(),
            "logq": self.logq_check.isChecked(),
            "logi": self.logi_check.isChecked(),
            "qmin": self.qmin_edit.text(),
            "qmax": self.qmax_edit.text(),
        }
        session_io.save_session(path, self.panels, self.variables, self.series, settings)
        self.statusBar().showMessage(f"Session saved to {path}", 5000)

    @safe_slot
    def action_load_session(self, *_):
        if self.panels or self.series:
            reply = QMessageBox.question(
                self, "Load Session",
                "Loading a session replaces all current curves, expressions, "
                "and signals. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "Population Fitter Session (*.pfsession *.json);;All files (*.*)",
        )
        if not path:
            return

        panels, variables, series, settings = session_io.load_session(path)

        for widget in list(self.panel_widgets.values()):
            self.panel_list_layout.removeWidget(widget)
            widget.deleteLater()
        self.panel_widgets.clear()

        self.panels = panels
        self.variables = variables
        self.series = series
        for panel in self.panels:
            self._add_panel_widget(panel)

        self.use_log_fitting = bool(settings.get("use_log_fitting", False))
        self.log_fitting_act.blockSignals(True)
        self.log_fitting_act.setChecked(self.use_log_fitting)
        self.log_fitting_act.blockSignals(False)
        self.fit_iterations = int(settings.get("fit_iterations", 200))
        self.logq_check.blockSignals(True)
        self.logq_check.setChecked(bool(settings.get("logq", False)))
        self.logq_check.blockSignals(False)
        self.logi_check.blockSignals(True)
        self.logi_check.setChecked(bool(settings.get("logi", True)))
        self.logi_check.blockSignals(False)
        fit_expr = settings.get("fit_expression") or ""
        if fit_expr:
            self.fit_expression_combo.blockSignals(True)
            self.fit_expression_combo.setEditText(fit_expr)
            self.fit_expression_combo.blockSignals(False)
        self.qmin_edit.setText(str(settings.get("qmin") or ""))
        self.qmax_edit.setText(str(settings.get("qmax") or ""))

        self._on_panels_changed()

        signal_designation = settings.get("signal_designation") or ""
        if signal_designation:
            idx = self.signal_combo.findText(signal_designation)
            if idx >= 0:
                self.signal_combo.setCurrentIndex(idx)

        self.statusBar().showMessage(f"Session loaded from {path}", 5000)
