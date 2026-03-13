"""
AmoraFiltering - Image Filtering Tools
=======================================
Interactive image filters: Gaussian, Median, CLAHE, Unsharp Mask, Non-Local Means.
Part of the AMORA-Digital Rock application.

LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin


# ===========================================================================
# FILTER DEFINITIONS
# ===========================================================================

FILTER_DEFS = {
    "gaussian": {
        "label": "Gaussian Blur",
        "desc":  "3D Gaussian smoothing (reduces noise, blurs edges)",
        "params": [
            {"name": "sigma", "label": "Sigma", "type": "float",
             "min": 0.1, "max": 20.0, "default": 1.0, "step": 0.1,
             "arg": "--sigma"},
        ],
    },
    "median": {
        "label": "Median Filter",
        "desc":  "3D median (removes salt-and-pepper noise, preserves edges)",
        "params": [
            {"name": "size", "label": "Kernel Size", "type": "int",
             "min": 3, "max": 15, "default": 3, "step": 2,
             "arg": "--size"},
        ],
    },
    "clahe": {
        "label": "CLAHE",
        "desc":  "Contrast Limited Adaptive Histogram Equalization (per-slice)",
        "params": [
            {"name": "clip_limit", "label": "Clip Limit", "type": "float",
             "min": 0.001, "max": 1.0, "default": 0.03, "step": 0.005,
             "arg": "--clip-limit"},
            {"name": "kernel_size", "label": "Kernel Size", "type": "int",
             "min": 4, "max": 128, "default": 8, "step": 2,
             "arg": "--kernel-size"},
        ],
    },
    "unsharp": {
        "label": "Unsharp Mask",
        "desc":  "Sharpens edges by subtracting a blurred copy",
        "params": [
            {"name": "sigma", "label": "Sigma", "type": "float",
             "min": 0.1, "max": 20.0, "default": 2.0, "step": 0.1,
             "arg": "--sigma"},
            {"name": "amount", "label": "Amount", "type": "float",
             "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1,
             "arg": "--amount"},
        ],
    },
    "nlm": {
        "label": "Non-Local Means",
        "desc":  "Advanced denoising (slow, high quality, per-slice)",
        "params": [
            {"name": "h", "label": "Strength (h)", "type": "float",
             "min": 0.01, "max": 2.0, "default": 0.1, "step": 0.01,
             "arg": "--h"},
            {"name": "patch_size", "label": "Patch Size", "type": "int",
             "min": 3, "max": 15, "default": 5, "step": 2,
             "arg": "--patch-size"},
            {"name": "patch_distance", "label": "Patch Distance", "type": "int",
             "min": 3, "max": 21, "default": 6, "step": 1,
             "arg": "--patch-distance"},
        ],
    },
    "ring_removal": {
        "label": "Ring Artifact Removal",
        "desc":  ("Removes ring artifacts from CT/microCT data. Converts each axial "
                   "slice to polar coordinates, filters radial stripes via median, "
                   "and converts back. Based on Avizo ring artifact correction method."),
        "params": [
            {"name": "max_radius", "label": "Max Radius (px)", "type": "int",
             "min": 10, "max": 2000, "default": 300, "step": 10,
             "arg": "--max-radius"},
            {"name": "ring_width", "label": "Ring Width (median kernel)", "type": "int",
             "min": 3, "max": 51, "default": 21, "step": 2,
             "arg": "--ring-width"},
            {"name": "center_x", "label": "Center X (0=auto)", "type": "int",
             "min": 0, "max": 10000, "default": 0, "step": 1,
             "arg": "--center-x"},
            {"name": "center_y", "label": "Center Y (0=auto)", "type": "int",
             "min": 0, "max": 10000, "default": 0, "step": 1,
             "arg": "--center-y"},
        ],
    },
}

FILTER_ORDER = ["gaussian", "median", "clahe", "unsharp", "nlm", "ring_removal"]


# ===========================================================================
# REGISTRATION
# ===========================================================================

class AmoraFiltering(ScriptedLoadableModule):

    def __init__(self, parent: Optional[qt.QWidget]):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Filtering"
        self.parent.categories = ["AMORA"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia",
        ]
        self.parent.helpText = (
            "Image filtering tools: Gaussian, Median, CLAHE, Unsharp Mask, "
            "Non-Local Means. Apply filters to the loaded volume data."
        )
        self.parent.acknowledgementText = "Projeto Catalisa ICT / CPGF / UFPA"
        self.parent.hidden = False


# ===========================================================================
# FILTERING TOOLS
# ===========================================================================

class AmoraFilteringWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent: Optional[qt.QWidget] = None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._process = None
        self._param_widgets = {}
        self._pipeline_steps = []

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # --- Header ---
        headerLabel = qt.QLabel("Image Filters")
        headerLabel.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #a78bfa; padding: 8px;"
        )
        self.layout.addWidget(headerLabel)

        descLabel = qt.QLabel(
            "Apply image filters to the loaded volume data. "
            "Filters can be stacked or applied from raw data."
        )
        descLabel.setStyleSheet(
            "font-size: 12px; color: #94a3b8; padding: 0 8px 8px 8px;"
        )
        descLabel.setWordWrap(True)
        self.layout.addWidget(descLabel)

        # --- Filter Selection ---
        filterCollapsible = ctk.ctkCollapsibleButton()
        filterCollapsible.text = "Filter Selection"
        self.layout.addWidget(filterCollapsible)
        filterLayout = qt.QVBoxLayout(filterCollapsible)

        self.comboFilter = qt.QComboBox()
        for key in FILTER_ORDER:
            self.comboFilter.addItem(FILTER_DEFS[key]["label"])
        self.comboFilter.currentIndexChanged.connect(self._onFilterChanged)
        filterLayout.addWidget(self.comboFilter)

        self.lblFilterDesc = qt.QLabel()
        self.lblFilterDesc.setWordWrap(True)
        self.lblFilterDesc.setStyleSheet("color: #94a3b8; font-size: 12px; padding: 4px;")
        filterLayout.addWidget(self.lblFilterDesc)

        # --- Parameters (dynamic) ---
        paramsCollapsible = ctk.ctkCollapsibleButton()
        paramsCollapsible.text = "Parameters"
        self.layout.addWidget(paramsCollapsible)
        self.paramsLayout = qt.QFormLayout(paramsCollapsible)

        # --- Actions ---
        actionsCollapsible = ctk.ctkCollapsibleButton()
        actionsCollapsible.text = "Actions"
        self.layout.addWidget(actionsCollapsible)
        actionsLayout = qt.QVBoxLayout(actionsCollapsible)

        self.chkFromRaw = qt.QCheckBox("Apply from raw (ignore previous filters)")
        self.chkFromRaw.setToolTip(
            "When checked, filter is applied to the original raw data.\n"
            "When unchecked, filter stacks on top of previously filtered data."
        )
        actionsLayout.addWidget(self.chkFromRaw)

        self.btnApply = qt.QPushButton("Apply Filter")
        self.btnApply.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #047857; }
            QPushButton:disabled { background-color: #374151; color: #6b7280; }
        """)
        self.btnApply.clicked.connect(self.onApplyFilter)
        actionsLayout.addWidget(self.btnApply)

        self.btnReset = qt.QPushButton("Reset to Raw")
        self.btnReset.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #b91c1c; }
            QPushButton:disabled { background-color: #374151; color: #6b7280; }
        """)
        self.btnReset.clicked.connect(self.onResetFilter)
        actionsLayout.addWidget(self.btnReset)

        # --- Pipeline ---
        pipelineCollapsible = ctk.ctkCollapsibleButton()
        pipelineCollapsible.text = "Applied Pipeline"
        pipelineCollapsible.collapsed = True
        self.layout.addWidget(pipelineCollapsible)
        pipelineLayout = qt.QVBoxLayout(pipelineCollapsible)

        self.pipelineList = qt.QLabel("No filters applied.")
        self.pipelineList.setWordWrap(True)
        self.pipelineList.setStyleSheet("color: #94a3b8; font-size: 12px;")
        pipelineLayout.addWidget(self.pipelineList)

        # --- Processing Log ---
        logCollapsible = ctk.ctkCollapsibleButton()
        logCollapsible.text = "Filter Log"
        self.layout.addWidget(logCollapsible)
        logLayout = qt.QVBoxLayout(logCollapsible)

        self.logOutput = qt.QTextEdit()
        self.logOutput.setReadOnly(True)
        self.logOutput.setMaximumHeight(250)
        logLayout.addWidget(self.logOutput)

        clearLogBtn = qt.QPushButton("Clear Log")
        clearLogBtn.clicked.connect(self.logOutput.clear)
        logLayout.addWidget(clearLogBtn)

        # --- Progress bar ---
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.setVisible(False)
        self.layout.addWidget(self.progressBar)

        self.layout.addStretch(1)
        self.logic = AmoraFilteringLogic()

        # Initialise filter UI
        self._onFilterChanged(0)

    # --- Lifecycle ---

    def enter(self):
        panel = slicer.util.mainWindow().findChild(qt.QDockWidget, "PanelDockWidget")
        if panel:
            panel.setVisible(True)

    def exit(self):
        pass

    def cleanup(self):
        pass

    # --- Dynamic filter parameters ---

    def _onFilterChanged(self, index):
        key = FILTER_ORDER[index] if 0 <= index < len(FILTER_ORDER) else None
        if not key or key not in FILTER_DEFS:
            return

        fdef = FILTER_DEFS[key]
        self.lblFilterDesc.setText(fdef["desc"])

        # Clear old param widgets
        while self.paramsLayout.rowCount() > 0:
            self.paramsLayout.removeRow(0)
        self._param_widgets.clear()

        # Create new param widgets
        for p in fdef["params"]:
            if p["type"] == "float":
                w = qt.QDoubleSpinBox()
                w.setRange(p["min"], p["max"])
                w.setValue(p["default"])
                w.setSingleStep(p["step"])
                w.setDecimals(3)
            elif p["type"] == "int":
                w = qt.QSpinBox()
                w.setRange(p["min"], p["max"])
                w.setValue(p["default"])
                w.setSingleStep(p["step"])
            else:
                continue

            self._param_widgets[p["name"]] = w
            self.paramsLayout.addRow(p["label"] + ":", w)

    # --- Helpers ---

    def _logAppend(self, msg):
        self.logOutput.append(msg)
        sb = self.logOutput.verticalScrollBar()
        sb.setValue(sb.maximum)

    def _cacheExists(self, name="_tensor_cache.npy"):
        tmp = Path(tempfile.gettempdir())
        return (tmp / name).exists()

    def _setButtonsEnabled(self, enabled):
        self.progressBar.setVisible(not enabled)
        self.btnApply.setEnabled(enabled)
        self.btnReset.setEnabled(enabled)

    def _getScriptsDir(self):
        """Get the path to the filtering scripts directory."""
        moduleDir = os.path.dirname(__file__)
        scriptsDir = os.path.join(moduleDir, "Resources", "Scripts")
        if os.path.isdir(scriptsDir):
            return scriptsDir
        # Fallback: try the source tree location
        scriptsDir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "AmoraFiltering", "Resources", "Scripts"
        )
        return scriptsDir

    def _runScript(self, scriptName, args=None):
        """Run a filtering script as a subprocess."""
        scriptsDir = self._getScriptsDir()
        scriptPath = os.path.join(scriptsDir, scriptName)

        if not os.path.isfile(scriptPath):
            self._logAppend(
                f"<span style='color:#ef4444;'>[ERROR] Script not found: {scriptPath}</span>"
            )
            self._setButtonsEnabled(True)
            return

        self._setButtonsEnabled(False)

        self._process = qt.QProcess()
        self._process.readyReadStandardOutput.connect(self._onStdout)
        self._process.readyReadStandardError.connect(self._onStderr)
        self._process.finished.connect(self._onFinished)

        cmdArgs = ["-u", scriptPath] + (args or [])
        self._process.start(sys.executable, cmdArgs)

    def _onStdout(self):
        if not self._process:
            return
        data = self._process.readAllStandardOutput()
        txt = data.data().decode("utf-8", errors="ignore").strip()
        if txt:
            self._logAppend(txt)

    def _onStderr(self):
        if not self._process:
            return
        data = self._process.readAllStandardError()
        txt = data.data().decode("utf-8", errors="ignore").strip()
        if txt:
            self._logAppend(f"<span style='color:#ef4444;'>{txt}</span>")

    def _onFinished(self, exitCode, exitStatus=None):
        if exitCode == 0:
            self._logAppend("<span style='color:#22c55e;'>[DONE] Success.</span>")
            if hasattr(self, '_currentFilterSummary') and self._currentFilterSummary:
                self._pipeline_steps.append({"name": self._currentFilterSummary})
                self._updatePipelineLabel()
        else:
            self._logAppend(
                f"<span style='color:#f59e0b;'>[DONE] Exit code {exitCode}</span>"
            )
        self._setButtonsEnabled(True)

    def _updatePipelineLabel(self):
        if not self._pipeline_steps:
            self.pipelineList.setText("No filters applied.")
            return
        lines = []
        for i, step in enumerate(self._pipeline_steps, 1):
            lines.append(f"{i}. {step['name']}")
        self.pipelineList.setText("\n".join(lines))

    # --- Filter Actions ---

    def onApplyFilter(self):
        if not self._cacheExists():
            slicer.util.warningDisplay(
                "No data loaded. Load data in Digital Rock first.",
                windowTitle="AMORA",
            )
            return

        index = self.comboFilter.currentIndex
        if index < 0 or index >= len(FILTER_ORDER):
            return
        key = FILTER_ORDER[index]

        fdef = FILTER_DEFS[key]
        args = ["--filter", key]

        # Collect param values
        param_summary = []
        for p in fdef["params"]:
            w = self._param_widgets.get(p["name"])
            if w is not None:
                val = w.value
                args.extend([p["arg"], str(val)])
                param_summary.append(f"{p['label']}={val}")

        if self.chkFromRaw.isChecked():
            args.append("--from-raw")

        summary = f"{fdef['label']} ({', '.join(param_summary)})"
        self._logAppend(f"[RUN] Applying: {summary}")
        self._currentFilterSummary = summary
        self._runScript("apply_filter.py", args)

    def onResetFilter(self):
        self._logAppend("[RUN] Resetting to raw data...")
        self._currentFilterSummary = None
        self._pipeline_steps.clear()
        self._updatePipelineLabel()
        self._runScript("apply_filter.py", ["--reset"])


# ===========================================================================
# LOGIC & TEST
# ===========================================================================

class AmoraFilteringLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class AmoraFilteringTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.delayDisplay("Testing filter definitions...")
        # Verify all filter defs are valid
        for key in FILTER_ORDER:
            self.assertTrue(key in FILTER_DEFS, f"Missing filter def: {key}")
            fdef = FILTER_DEFS[key]
            self.assertTrue("label" in fdef, f"Missing label for {key}")
            self.assertTrue("params" in fdef, f"Missing params for {key}")
        self.delayDisplay("Filter definitions test passed!")
