"""Qt widget representing one File/Expression panel (mirrors the original
FileExpressionPanel: a single panel that can be toggled between showing a
loaded data file or a typed expression)."""
from __future__ import annotations

import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox, QColorDialog, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from expr_eval import is_valid_designation
from io_utils import DataFileError, load_curve_file
from models import Panel


def _caption(text: str) -> QLabel:
    """A short field-label QLabel, tagged so the app-wide stylesheet
    (set in main_window.MainWindow._apply_header_styles) can make it
    larger/bold to match the other captions across the app."""
    label = QLabel(text)
    label.setObjectName("FieldCaption")
    return label


class FileExpressionPanelWidget(QGroupBox):
    changed = pyqtSignal()
    removed = pyqtSignal(int)
    export_requested = pyqtSignal(int)

    def __init__(self, panel: Panel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self._build_ui()
        self._load_from_panel()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        top = QHBoxLayout()
        self.file_radio = QRadioButton("File")
        self.expr_radio = QRadioButton("Expression")
        self.file_radio.toggled.connect(self._on_mode_toggled)
        self.expr_radio.toggled.connect(self._on_mode_toggled)
        top.addWidget(self.file_radio)
        top.addWidget(self.expr_radio)
        top.addWidget(_caption("Name:"))
        self.designation_edit = QLineEdit()
        self.designation_edit.setMaximumWidth(110)
        self.designation_edit.editingFinished.connect(self._on_designation_changed)
        top.addWidget(self.designation_edit)
        outer.addLayout(top)

        self.stack = QStackedWidget()

        file_widget = QWidget()
        file_layout = QHBoxLayout(file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.filename_edit = QLineEdit()
        self.filename_edit.setReadOnly(True)
        self.filename_edit.setPlaceholderText("(no file loaded)")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.filename_edit)
        file_layout.addWidget(browse_btn)
        self.stack.addWidget(file_widget)

        expr_widget = QWidget()
        expr_layout = QHBoxLayout(expr_widget)
        expr_layout.setContentsMargins(0, 0, 0, 0)
        self.expression_edit = QLineEdit()
        self.expression_edit.setPlaceholderText("e.g. a*Curve1 + (1-a)*Curve2")
        self.expression_edit.editingFinished.connect(self._on_expression_changed)
        expr_layout.addWidget(self.expression_edit)
        self.stack.addWidget(expr_widget)

        outer.addWidget(self.stack)

        bottom = QHBoxLayout()
        self.color_btn = QPushButton()
        self.color_btn.setFixedWidth(28)
        self.color_btn.setToolTip("Curve color")
        self.color_btn.clicked.connect(self._pick_color)
        bottom.addWidget(self.color_btn)
        self.visible_check = QCheckBox("Visible")
        self.visible_check.toggled.connect(self._on_visible_changed)
        bottom.addWidget(self.visible_check)
        bottom.addWidget(_caption("Subunits:"))
        self.subunits_spin = QDoubleSpinBox()
        self.subunits_spin.setDecimals(2)
        self.subunits_spin.setRange(0.01, 1_000_000.0)
        self.subunits_spin.setSingleStep(1.0)
        self.subunits_spin.setMaximumWidth(80)
        self.subunits_spin.setToolTip(
            "Number of subunits in this component (e.g. 1 for monomer, 2 for "
            "dimer, 4 for tetramer). Used to convert its fitted population "
            "(molar) fraction into a mass fraction."
        )
        self.subunits_spin.valueChanged.connect(self._on_subunits_changed)
        bottom.addWidget(self.subunits_spin)
        bottom.addStretch()
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(lambda: self.export_requested.emit(self.panel.id))
        bottom.addWidget(export_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self.removed.emit(self.panel.id))
        bottom.addWidget(remove_btn)
        outer.addLayout(bottom)

    def _load_from_panel(self):
        self.designation_edit.setText(self.panel.designation)
        # Block signals while setting both radios: toggling one before the
        # other is set can otherwise fire an intermediate _on_mode_toggled()
        # that reads stale state and silently flips panel.is_expression back.
        self.file_radio.blockSignals(True)
        self.expr_radio.blockSignals(True)
        self.file_radio.setChecked(not self.panel.is_expression)
        self.expr_radio.setChecked(self.panel.is_expression)
        self.file_radio.blockSignals(False)
        self.expr_radio.blockSignals(False)
        self.stack.setCurrentIndex(1 if self.panel.is_expression else 0)
        self.filename_edit.setText(self.panel.filepath or "")
        self.expression_edit.setText(self.panel.expression)
        self.visible_check.setChecked(self.panel.visible)
        self.subunits_spin.blockSignals(True)
        self.subunits_spin.setValue(self.panel.subunits)
        self.subunits_spin.blockSignals(False)
        self._update_color_swatch()
        self._update_title()

    def _update_color_swatch(self):
        self.color_btn.setStyleSheet(
            f"background-color: {self.panel.color}; border: 1px solid #444;"
        )

    def _update_title(self):
        self.setTitle(self.panel.designation or "(unnamed)")

    def _on_mode_toggled(self):
        is_expr = self.expr_radio.isChecked()
        if self.panel.is_expression == is_expr:
            return
        self.panel.is_expression = is_expr
        self.stack.setCurrentIndex(1 if is_expr else 0)
        self.changed.emit()

    def _on_designation_changed(self):
        name = self.designation_edit.text().strip()
        if name == self.panel.designation:
            return
        if not is_valid_designation(name):
            QMessageBox.warning(
                self, "Invalid name",
                "A panel name must start with a letter or underscore and "
                "contain only letters, digits, and underscores.",
            )
            self.designation_edit.setText(self.panel.designation)
            return
        self.panel.designation = name
        self._update_title()
        self.changed.emit()

    def _on_expression_changed(self):
        text = self.expression_edit.text()
        if text == self.panel.expression:
            return
        self.panel.expression = text
        self.changed.emit()

    def _on_visible_changed(self, checked: bool):
        self.panel.visible = checked
        self.changed.emit()

    def _on_subunits_changed(self, value: float):
        self.panel.subunits = value
        self.changed.emit()

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self.panel.color), self)
        if color.isValid():
            self.panel.color = color.name()
            self._update_color_swatch()
            self.changed.emit()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a signal output file", "",
            "Data Files (*.out *.dat *.chi *.txt);;All files (*.*)",
        )
        if path:
            self.set_file(path)

    def set_file(self, path: str):
        try:
            curve, warnings = load_curve_file(path)
        except DataFileError as exc:
            QMessageBox.warning(self, "Invalid data file", str(exc))
            return
        self.panel.filepath = path
        self.panel.curve = curve
        if not self.panel.designation or self.panel.designation.startswith("Panel"):
            base = os.path.splitext(os.path.basename(path))[0]
            base = "".join(c if (c.isalnum() or c == "_") else "_" for c in base)
            if base and base[0].isdigit():
                base = "_" + base
            if base and is_valid_designation(base):
                self.panel.designation = base
                self.designation_edit.setText(base)
        self.filename_edit.setText(path)
        self._update_title()
        if warnings:
            QMessageBox.information(self, "Data file loaded", "\n".join(warnings))
        self.changed.emit()
