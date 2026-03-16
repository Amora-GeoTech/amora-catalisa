"""
AmoraSimulation - LBM Flow Simulation
==========================================
Lattice Boltzmann Method (D3Q19 MRT) flow simulation through
binarized rock samples. Based on taichi_LBM3D.

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
# REGISTRATION
# ===========================================================================

class AmoraSimulation(ScriptedLoadableModule):

    def __init__(self, parent: Optional[qt.QWidget]):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Simulation"
        self.parent.categories = [""]
        self.parent.dependencies = []
        self.parent.contributors = [
            "LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia"
        ]
        self.parent.helpText = (
            "LBM (Lattice Boltzmann) flow simulation through binarized rock samples. "
            "D3Q19 MRT solver with GPU acceleration via Taichi."
        )
        self.parent.acknowledgementText = "Projeto Catalisa ICT / CPGF / UFPA"
        self.parent.hidden = False


# ===========================================================================
# SIMULATION WIDGET
# ===========================================================================

class AmoraSimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent: Optional[qt.QWidget] = None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._process = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # --- Header ---
        headerLabel = qt.QLabel("LBM Flow Simulation")
        headerLabel.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #22d3ee; padding: 8px;"
        )
        self.layout.addWidget(headerLabel)

        descLabel = qt.QLabel(
            "Lattice Boltzmann D3Q19 MRT solver for porous media flow.\n"
            "Requires a binarized volume (0=pore, 1=solid)."
        )
        descLabel.setStyleSheet("font-size: 12px; color: #94a3b8; padding: 0 8px 8px 8px;")
        descLabel.setWordWrap(True)
        self.layout.addWidget(descLabel)

        # --- Dependencies ---
        depCollapsible = ctk.ctkCollapsibleButton()
        depCollapsible.text = "Dependencies"
        depCollapsible.collapsed = True
        self.layout.addWidget(depCollapsible)
        depLayout = qt.QVBoxLayout(depCollapsible)

        self.depStatus = qt.QLabel("Checking...")
        self.depStatus.setStyleSheet("color: #94a3b8; padding: 4px;")
        depLayout.addWidget(self.depStatus)

        self.installDepsBtn = qt.QPushButton("Install taichi + pyevtk")
        self.installDepsBtn.setToolTip("pip install taichi pyevtk into Slicer's Python")
        self.installDepsBtn.clicked.connect(self.onInstallDeps)
        depLayout.addWidget(self.installDepsBtn)

        # --- Geometry Source ---
        geoCollapsible = ctk.ctkCollapsibleButton()
        geoCollapsible.text = "Geometry (Binary Volume)"
        self.layout.addWidget(geoCollapsible)
        geoLayout = qt.QFormLayout(geoCollapsible)

        self.geoSourceCombo = qt.QComboBox()
        self.geoSourceCombo.addItems([
            "From binarized cache (Otsu result)",
            "From active Slicer volume",
        ])
        geoLayout.addRow("Source:", self.geoSourceCombo)

        self.solidValueCombo = qt.QComboBox()
        self.solidValueCombo.addItems(["1 = solid, 0 = pore (default)", "0 = solid, 1 = pore (inverted)"])
        geoLayout.addRow("Convention:", self.solidValueCombo)

        self.grayscaleCheck = qt.QCheckBox("Grayscale mode (partial bounce-back)")
        self.grayscaleCheck.setToolTip(
            "Use grayscale values as solid fraction (0=pore, 1=solid).\n"
            "Works directly on raw microCT data without binarization."
        )
        geoLayout.addRow("", self.grayscaleCheck)

        # --- Simulation Type ---
        typeCollapsible = ctk.ctkCollapsibleButton()
        typeCollapsible.text = "Simulation Type"
        self.layout.addWidget(typeCollapsible)
        typeLayout = qt.QFormLayout(typeCollapsible)

        self.simTypeCombo = qt.QComboBox()
        self.simTypeCombo.addItems([
            "Single-Phase Flow",
            "Two-Phase Flow (Color Gradient)",
        ])
        self.simTypeCombo.currentIndexChanged.connect(self._onSimTypeChanged)
        typeLayout.addRow("Type:", self.simTypeCombo)

        # --- Two-Phase Parameters (hidden by default) ---
        self.twoPhaseWidget = qt.QWidget()
        tpLayout = qt.QFormLayout(self.twoPhaseWidget)
        tpLayout.setContentsMargins(0, 0, 0, 0)

        self.niuLiquidSpin = qt.QDoubleSpinBox()
        self.niuLiquidSpin.setRange(0.001, 1.0)
        self.niuLiquidSpin.setValue(0.1)
        self.niuLiquidSpin.setDecimals(4)
        self.niuLiquidSpin.setSingleStep(0.01)
        self.niuLiquidSpin.setToolTip("Kinematic viscosity of liquid phase (psi > 0)")
        tpLayout.addRow("Viscosity liquid:", self.niuLiquidSpin)

        self.niuGasSpin = qt.QDoubleSpinBox()
        self.niuGasSpin.setRange(0.001, 1.0)
        self.niuGasSpin.setValue(0.1)
        self.niuGasSpin.setDecimals(4)
        self.niuGasSpin.setSingleStep(0.01)
        self.niuGasSpin.setToolTip("Kinematic viscosity of gas phase (psi < 0)")
        tpLayout.addRow("Viscosity gas:", self.niuGasSpin)

        self.surfaceTensionSpin = qt.QDoubleSpinBox()
        self.surfaceTensionSpin.setRange(0.0, 0.1)
        self.surfaceTensionSpin.setValue(0.005)
        self.surfaceTensionSpin.setDecimals(4)
        self.surfaceTensionSpin.setSingleStep(0.001)
        self.surfaceTensionSpin.setToolTip("Surface tension parameter (CapA)")
        tpLayout.addRow("Surface tension:", self.surfaceTensionSpin)

        self.psiSolidSpin = qt.QDoubleSpinBox()
        self.psiSolidSpin.setRange(-1.0, 1.0)
        self.psiSolidSpin.setValue(0.7)
        self.psiSolidSpin.setDecimals(2)
        self.psiSolidSpin.setSingleStep(0.1)
        self.psiSolidSpin.setToolTip("Wetting parameter at solid surfaces (-1=hydrophilic, 1=hydrophobic)")
        tpLayout.addRow("Wetting (psi_solid):", self.psiSolidSpin)

        self.phaseInitCombo = qt.QComboBox()
        self.phaseInitCombo.addItems([
            "Half-split along flow direction",
            "Random distribution",
        ])
        tpLayout.addRow("Phase init:", self.phaseInitCombo)

        self.twoPhaseWidget.setVisible(False)
        typeLayout.addRow("", self.twoPhaseWidget)

        # --- Simulation Parameters ---
        paramCollapsible = ctk.ctkCollapsibleButton()
        paramCollapsible.text = "Simulation Parameters"
        self.layout.addWidget(paramCollapsible)
        paramLayout = qt.QFormLayout(paramCollapsible)

        self.viscositySpin = qt.QDoubleSpinBox()
        self.viscositySpin.setRange(0.001, 1.0)
        self.viscositySpin.setValue(0.16667)
        self.viscositySpin.setDecimals(5)
        self.viscositySpin.setSingleStep(0.01)
        self.viscositySpin.setToolTip("Kinematic viscosity in lattice units (single-phase, default 1/6)")
        paramLayout.addRow("Viscosity (niu):", self.viscositySpin)

        self.timestepsSpin = qt.QSpinBox()
        self.timestepsSpin.setRange(100, 1000000)
        self.timestepsSpin.setValue(5000)
        self.timestepsSpin.setSingleStep(1000)
        self.timestepsSpin.setToolTip("Number of LBM timesteps to simulate")
        paramLayout.addRow("Timesteps:", self.timestepsSpin)

        self.saveIntervalSpin = qt.QSpinBox()
        self.saveIntervalSpin.setRange(100, 100000)
        self.saveIntervalSpin.setValue(1000)
        self.saveIntervalSpin.setSingleStep(500)
        self.saveIntervalSpin.setToolTip("Save results every N steps")
        paramLayout.addRow("Save interval:", self.saveIntervalSpin)

        # --- Boundary Conditions ---
        bcCollapsible = ctk.ctkCollapsibleButton()
        bcCollapsible.text = "Boundary Conditions"
        self.layout.addWidget(bcCollapsible)
        bcLayout = qt.QFormLayout(bcCollapsible)

        # Flow direction
        self.flowDirCombo = qt.QComboBox()
        self.flowDirCombo.addItems(["X (left→right)", "Y (front→back)", "Z (bottom→top)"])
        bcLayout.addRow("Flow direction:", self.flowDirCombo)

        # BC type
        self.bcTypeCombo = qt.QComboBox()
        self.bcTypeCombo.addItems([
            "Pressure-driven (ΔP)",
            "Velocity-driven (inlet velocity)",
            "Body force (gravity/pressure gradient)",
        ])
        self.bcTypeCombo.currentIndexChanged.connect(self._onBcTypeChanged)
        bcLayout.addRow("BC type:", self.bcTypeCombo)

        # Pressure BC
        self.rhoInSpin = qt.QDoubleSpinBox()
        self.rhoInSpin.setRange(0.5, 2.0)
        self.rhoInSpin.setValue(1.005)
        self.rhoInSpin.setDecimals(4)
        self.rhoInSpin.setSingleStep(0.001)
        bcLayout.addRow("Inlet density (ρ_in):", self.rhoInSpin)

        self.rhoOutSpin = qt.QDoubleSpinBox()
        self.rhoOutSpin.setRange(0.5, 2.0)
        self.rhoOutSpin.setValue(0.995)
        self.rhoOutSpin.setDecimals(4)
        self.rhoOutSpin.setSingleStep(0.001)
        bcLayout.addRow("Outlet density (ρ_out):", self.rhoOutSpin)

        # Velocity BC
        self.velInSpin = qt.QDoubleSpinBox()
        self.velInSpin.setRange(0.0, 0.2)
        self.velInSpin.setValue(0.01)
        self.velInSpin.setDecimals(5)
        self.velInSpin.setSingleStep(0.001)
        self.velInSpin.setVisible(False)
        bcLayout.addRow("Inlet velocity:", self.velInSpin)
        self._velInLabel = bcLayout.labelForField(self.velInSpin)

        # Force BC
        self.forceSpin = qt.QDoubleSpinBox()
        self.forceSpin.setRange(0.0, 0.001)
        self.forceSpin.setValue(1.0e-5)
        self.forceSpin.setDecimals(7)
        self.forceSpin.setSingleStep(1.0e-6)
        self.forceSpin.setVisible(False)
        bcLayout.addRow("Body force:", self.forceSpin)
        self._forceLabel = bcLayout.labelForField(self.forceSpin)

        self._onBcTypeChanged(0)

        # --- GPU Backend ---
        gpuCollapsible = ctk.ctkCollapsibleButton()
        gpuCollapsible.text = "Compute Backend"
        gpuCollapsible.collapsed = True
        self.layout.addWidget(gpuCollapsible)
        gpuLayout = qt.QFormLayout(gpuCollapsible)

        self.backendCombo = qt.QComboBox()
        self.backendCombo.addItems(["gpu (CUDA)", "cpu", "vulkan"])
        gpuLayout.addRow("Backend:", self.backendCombo)

        # --- Run ---
        runCollapsible = ctk.ctkCollapsibleButton()
        runCollapsible.text = "Run Simulation"
        self.layout.addWidget(runCollapsible)
        runLayout = qt.QVBoxLayout(runCollapsible)

        self.runBtn = qt.QPushButton("▶  Start LBM Simulation")
        self.runBtn.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 12px; "
            "background-color: #064e3b; color: #34d399; border: 1px solid #34d399; "
            "border-radius: 8px;"
        )
        self.runBtn.clicked.connect(self.onRunSimulation)
        runLayout.addWidget(self.runBtn)

        self.stopBtn = qt.QPushButton("Stop")
        self.stopBtn.setEnabled(False)
        self.stopBtn.clicked.connect(self.onStopSimulation)
        runLayout.addWidget(self.stopBtn)

        self.loadResultBtn = qt.QPushButton("Load Last Result into Slicer")
        self.loadResultBtn.clicked.connect(self.onLoadResult)
        runLayout.addWidget(self.loadResultBtn)

        # --- Flow Visualization ---
        vizCollapsible = ctk.ctkCollapsibleButton()
        vizCollapsible.text = "Flow Visualization"
        self.layout.addWidget(vizCollapsible)
        vizLayout = qt.QVBoxLayout(vizCollapsible)

        # View mode selection
        self.vizModeCombo = qt.QComboBox()
        self.vizModeCombo.addItems([
            "Internal (slice through Z - see fluid inside)",
            "External (3D volume rendering - see flow outside)",
            "Isosurface (3D velocity isosurfaces)",
        ])
        vizLayout.addWidget(self.vizModeCombo)

        # Isosurface threshold (fraction of max velocity)
        isoRow = qt.QHBoxLayout()
        isoRow.addWidget(qt.QLabel("Iso levels (%):"))
        self.isoLevelsSpin = qt.QSpinBox()
        self.isoLevelsSpin.setRange(1, 10)
        self.isoLevelsSpin.setValue(3)
        self.isoLevelsSpin.setToolTip("Number of isosurface levels to display")
        isoRow.addWidget(self.isoLevelsSpin)
        vizLayout.addLayout(isoRow)

        # Overlay rock structure
        self.overlayRockCheck = qt.QCheckBox("Overlay rock structure")
        self.overlayRockCheck.checked = True
        self.overlayRockCheck.setToolTip("Show rock grains alongside flow field")
        vizLayout.addWidget(self.overlayRockCheck)

        # Playback controls
        playRow = qt.QHBoxLayout()
        self.playBtn = qt.QPushButton("Play")
        self.playBtn.clicked.connect(self.onPlayAnimation)
        playRow.addWidget(self.playBtn)

        self.pauseBtn = qt.QPushButton("Pause")
        self.pauseBtn.setEnabled(False)
        self.pauseBtn.clicked.connect(self.onPauseAnimation)
        playRow.addWidget(self.pauseBtn)

        self.fpsLabel = qt.QLabel("FPS:")
        playRow.addWidget(self.fpsLabel)
        self.fpsSpin = qt.QSpinBox()
        self.fpsSpin.setRange(1, 30)
        self.fpsSpin.setValue(5)
        playRow.addWidget(self.fpsSpin)
        vizLayout.addLayout(playRow)

        # Frame slider
        sliderRow = qt.QHBoxLayout()
        self.frameLabel = qt.QLabel("Frame: 0/0")
        sliderRow.addWidget(self.frameLabel)
        self.frameSlider = qt.QSlider(qt.Qt.Horizontal)
        self.frameSlider.setRange(0, 0)
        self.frameSlider.valueChanged.connect(self._onFrameChanged)
        sliderRow.addWidget(self.frameSlider)
        vizLayout.addLayout(sliderRow)

        # Export video
        self.exportVideoBtn = qt.QPushButton("Export Video (MP4)")
        self.exportVideoBtn.clicked.connect(self.onExportVideo)
        vizLayout.addWidget(self.exportVideoBtn)

        # Animation state
        self._animTimer = qt.QTimer()
        self._animTimer.timeout.connect(self._onAnimTick)
        self._vizFrames = []
        self._vizNode = None
        self._vizRockNode = None
        self._vrNode = None

        # --- Permeability ---
        permCollapsible = ctk.ctkCollapsibleButton()
        permCollapsible.text = "Results / Permeability"
        permCollapsible.collapsed = True
        self.layout.addWidget(permCollapsible)
        permLayout = qt.QVBoxLayout(permCollapsible)

        self.permLabel = qt.QLabel("Run a simulation first.")
        self.permLabel.setStyleSheet("color: #94a3b8; padding: 8px;")
        self.permLabel.setWordWrap(True)
        permLayout.addWidget(self.permLabel)

        # --- Log ---
        logCollapsible = ctk.ctkCollapsibleButton()
        logCollapsible.text = "Simulation Log"
        self.layout.addWidget(logCollapsible)
        logLayout = qt.QVBoxLayout(logCollapsible)

        self.logOutput = qt.QTextEdit()
        self.logOutput.setReadOnly(True)
        self.logOutput.setMaximumHeight(300)
        logLayout.addWidget(self.logOutput)

        clearLogBtn = qt.QPushButton("Clear Log")
        clearLogBtn.clicked.connect(self.logOutput.clear)
        logLayout.addWidget(clearLogBtn)

        # --- Progress ---
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.setVisible(False)
        self.layout.addWidget(self.progressBar)

        self.layout.addStretch(1)
        self.logic = AmoraSimulationLogic()

        # Check deps on load
        qt.QTimer.singleShot(500, self._checkDeps)

    # --- Lifecycle ---

    def enter(self):
        panel = slicer.util.mainWindow().findChild(qt.QDockWidget, "PanelDockWidget")
        if panel:
            panel.setVisible(True)

    def exit(self):
        pass

    def cleanup(self):
        if self._process and self._process.state() != qt.QProcess.NotRunning:
            self._process.kill()

    # --- Helpers ---

    def _logAppend(self, msg):
        self.logOutput.append(msg)
        sb = self.logOutput.verticalScrollBar()
        sb.setValue(sb.maximum)

    def _findSystemPython(self):
        """Find system Python (not Slicer's bundled one, which lacks SSL)."""
        import shutil
        # Try common system Python paths on Windows
        candidates = [
            shutil.which("python3"),
            shutil.which("python"),
        ]
        # Also check common install locations
        for ver in ["312", "311", "310", "39"]:
            candidates.append(f"C:/Python{ver}/python.exe")
            candidates.append(f"C:/Users/{os.environ.get('USERNAME', '')}/AppData/Local/Programs/Python/Python{ver}/python.exe")

        for c in candidates:
            if c and os.path.isfile(c):
                # Make sure it's not Slicer's Python (check for SSL)
                try:
                    import subprocess
                    result = subprocess.run(
                        [c, "-c", "import ssl; print('ok')"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and "ok" in result.stdout:
                        return c
                except Exception:
                    continue
        return None

    def _checkDeps(self):
        """Check if taichi and pyevtk are installed in system Python."""
        self._systemPython = self._findSystemPython()
        if not self._systemPython:
            self.depStatus.setText(
                "<span style='color:#ef4444;'>System Python not found. "
                "Install Python 3.10+ from python.org</span>"
            )
            self.installDepsBtn.setEnabled(False)
            return

        import subprocess
        missing = []
        try:
            r = subprocess.run(
                [self._systemPython, "-c", "import taichi; print(taichi.__version__)"],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                ti_ver = r.stdout.strip()
            else:
                ti_ver = None
                missing.append("taichi")
        except Exception:
            ti_ver = None
            missing.append("taichi")

        try:
            r = subprocess.run(
                [self._systemPython, "-c", "from pyevtk.hl import gridToVTK; print('ok')"],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode != 0:
                missing.append("pyevtk")
        except Exception:
            missing.append("pyevtk")

        if not missing:
            self.depStatus.setText(
                f"<span style='color:#22c55e;'>OK - taichi {ti_ver} "
                f"({self._systemPython})</span>"
            )
            self.installDepsBtn.setEnabled(False)
        else:
            self.depStatus.setText(
                f"<span style='color:#ef4444;'>Missing: {', '.join(missing)} "
                f"(Python: {self._systemPython})</span>"
            )
            self.installDepsBtn.setEnabled(True)

    def _onSimTypeChanged(self, idx):
        """Show/hide two-phase parameters based on simulation type."""
        is_two_phase = (idx == 1)
        self.twoPhaseWidget.setVisible(is_two_phase)
        # Hide single-phase viscosity when two-phase is selected
        self.viscositySpin.setVisible(not is_two_phase)
        label = self.viscositySpin.parent().layout().labelForField(self.viscositySpin)
        if label:
            label.setVisible(not is_two_phase)
        # Hide grayscale option for two-phase
        self.grayscaleCheck.setVisible(not is_two_phase)

    def _onBcTypeChanged(self, idx):
        # 0=pressure, 1=velocity, 2=force
        self.rhoInSpin.setVisible(idx == 0)
        self.rhoOutSpin.setVisible(idx == 0)
        if hasattr(self, '_velInLabel') and self._velInLabel:
            self._velInLabel.setVisible(idx == 1)
        self.velInSpin.setVisible(idx == 1)
        if hasattr(self, '_forceLabel') and self._forceLabel:
            self._forceLabel.setVisible(idx == 2)
        self.forceSpin.setVisible(idx == 2)

    def _setButtonsEnabled(self, enabled):
        self.progressBar.setVisible(not enabled)
        self.runBtn.setEnabled(enabled)
        self.stopBtn.setEnabled(not enabled)
        self.installDepsBtn.setEnabled(enabled)

    def _getScriptsDir(self):
        moduleDir = os.path.dirname(__file__)
        scriptsDir = os.path.join(moduleDir, "Resources", "Scripts")
        if os.path.isdir(scriptsDir):
            return scriptsDir
        scriptsDir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "AmoraSimulation", "Resources", "Scripts"
        )
        return scriptsDir

    def _runScript(self, scriptName, args=None):
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

        pythonExe = getattr(self, '_systemPython', None) or sys.executable
        cmdArgs = ["-u", scriptPath] + (args or [])
        self._process.start(pythonExe, cmdArgs)

    def _onStdout(self):
        if not self._process:
            return
        txt = self._process.readAllStandardOutput().data().decode("utf-8", errors="ignore").strip()
        if txt:
            for line in txt.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Try to parse JSON results
                try:
                    if line.startswith("{"):
                        data = json.loads(line)
                        if "permeability" in data:
                            k_val = data["permeability"]
                            self.permLabel.setText(
                                f"<b style='color:#64ffda;'>Permeability (lattice): {k_val:.6e}</b><br>"
                                f"Direction: {data.get('direction', '?')}<br>"
                                f"Mean velocity: {data.get('mean_velocity', '?'):.6e}<br>"
                                f"Porosity: {data.get('porosity', '?'):.4f}"
                            )
                        if "step" in data:
                            step = data["step"]
                            total = data.get("total", self.timestepsSpin.value)
                            self.progressBar.setRange(0, total)
                            self.progressBar.setValue(step)
                        continue
                except Exception:
                    pass
                self._logAppend(line)

    def _onStderr(self):
        if not self._process:
            return
        txt = self._process.readAllStandardError().data().decode("utf-8", errors="ignore").strip()
        if txt:
            # Filter taichi compilation messages (not real errors)
            for line in txt.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if "[Taichi]" in line or "ti.init" in line:
                    self._logAppend(f"<span style='color:#57cbff;'>{line}</span>")
                else:
                    self._logAppend(f"<span style='color:#ef4444;'>{line}</span>")

    def _onFinished(self, exitCode, exitStatus=None):
        if exitCode == 0:
            self._logAppend("<span style='color:#22c55e;'>[DONE] Simulation finished.</span>")
        else:
            self._logAppend(
                f"<span style='color:#f59e0b;'>[DONE] Exit code {exitCode}</span>"
            )
        self._setButtonsEnabled(True)

    # --- Actions ---

    def onInstallDeps(self):
        pythonExe = getattr(self, '_systemPython', None)
        if not pythonExe:
            self._logAppend(
                "<span style='color:#ef4444;'>[ERROR] System Python not found. "
                "Install Python from python.org first.</span>"
            )
            return
        self._logAppend(f"[INFO] Installing taichi and pyevtk via {pythonExe}...")
        self._setButtonsEnabled(False)

        self._process = qt.QProcess()
        self._process.readyReadStandardOutput.connect(self._onStdout)
        self._process.readyReadStandardError.connect(self._onStderr)

        def onInstallFinished(exitCode, exitStatus=None):
            if exitCode == 0:
                self._logAppend("<span style='color:#22c55e;'>[DONE] Dependencies installed!</span>")
                self._checkDeps()
            else:
                self._logAppend("<span style='color:#ef4444;'>[ERROR] Install failed.</span>")
            self._setButtonsEnabled(True)

        self._process.finished.connect(onInstallFinished)
        self._process.start(pythonExe, ["-m", "pip", "install", "taichi", "pyevtk"])

    def onRunSimulation(self):
        tmp = Path(tempfile.gettempdir())
        is_grayscale = self.grayscaleCheck.checked

        # Check geometry source
        geoIdx = self.geoSourceCombo.currentIndex
        if geoIdx == 0:
            if is_grayscale:
                # Grayscale: use raw tensor cache
                if not (tmp / "_tensor_cache.npy").exists():
                    slicer.util.warningDisplay(
                        "No data loaded.\nLoad data in Digital Rock first.",
                        windowTitle="AMORA",
                    )
                    return
                geoPath = str(tmp / "_tensor_cache.npy")
            else:
                # Binary: use binarized cache
                if not (tmp / "_tensor_bin.npy").exists():
                    slicer.util.warningDisplay(
                        "No binarized data found.\n"
                        "Go to Processing → Otsu Segmentation first.",
                        windowTitle="AMORA",
                    )
                    return
                geoPath = str(tmp / "_tensor_bin.npy")
        else:
            # From active volume
            activeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if not activeNode:
                slicer.util.warningDisplay("No volume loaded.", windowTitle="AMORA")
                return
            arr = slicer.util.arrayFromVolume(activeNode)
            geoPath = str(tmp / "_lbm_geometry.npy")
            np.save(geoPath, arr)

        # Build args
        args = [
            "--geometry", geoPath,
            "--viscosity", str(self.viscositySpin.value),
            "--timesteps", str(self.timestepsSpin.value),
            "--save-interval", str(self.saveIntervalSpin.value),
            "--backend", self.backendCombo.currentText.split(" ")[0],
            "--output-dir", str(tmp / "lbm_results"),
        ]

        # Grayscale mode
        if is_grayscale:
            args.append("--grayscale")

        # Solid convention
        if self.solidValueCombo.currentIndex == 1:
            args.append("--invert-solid")

        # Flow direction
        flowDir = ["x", "y", "z"][self.flowDirCombo.currentIndex]
        args.extend(["--flow-direction", flowDir])

        # BC type
        bcIdx = self.bcTypeCombo.currentIndex
        if bcIdx == 0:
            args.extend([
                "--bc-type", "pressure",
                "--rho-inlet", str(self.rhoInSpin.value),
                "--rho-outlet", str(self.rhoOutSpin.value),
            ])
        elif bcIdx == 1:
            args.extend([
                "--bc-type", "velocity",
                "--velocity-inlet", str(self.velInSpin.value),
            ])
        else:
            args.extend([
                "--bc-type", "force",
                "--body-force", str(self.forceSpin.value),
            ])

        # Determine which solver to use
        simType = self.simTypeCombo.currentIndex
        if simType == 1:
            # Two-Phase Flow
            # Replace single-phase args with two-phase args
            args = [
                "--geometry", geoPath,
                "--timesteps", str(self.timestepsSpin.value),
                "--save-interval", str(self.saveIntervalSpin.value),
                "--backend", self.backendCombo.currentText.split(" ")[0],
                "--output-dir", str(tmp / "lbm_results"),
                "--niu-liquid", str(self.niuLiquidSpin.value),
                "--niu-gas", str(self.niuGasSpin.value),
                "--surface-tension", str(self.surfaceTensionSpin.value),
                "--psi-solid", str(self.psiSolidSpin.value),
                "--flow-direction", flowDir,
            ]
            # Phase init
            phaseIdx = self.phaseInitCombo.currentIndex
            if phaseIdx == 0:
                args.extend(["--phase-init", "half"])
            else:
                args.extend(["--phase-init", "random"])
            # Solid convention
            if self.solidValueCombo.currentIndex == 1:
                args.append("--invert-solid")
            # BC type
            if bcIdx == 0:
                args.extend([
                    "--bc-type", "pressure",
                    "--rho-inlet", str(self.rhoInSpin.value),
                    "--rho-outlet", str(self.rhoOutSpin.value),
                ])
            else:
                args.extend([
                    "--bc-type", "force",
                    "--body-force", str(self.forceSpin.value),
                ])

            self._logAppend("[RUN] Starting Two-Phase LBM simulation...")
            self._logAppend(f"[INFO] Geometry: {geoPath}")
            self._logAppend(f"[INFO] niu_l={self.niuLiquidSpin.value}, niu_g={self.niuGasSpin.value}, CapA={self.surfaceTensionSpin.value}")
            self._runScript("lbm_two_phase.py", args)
        else:
            # Single-Phase Flow
            self._logAppend("[RUN] Starting LBM simulation...")
            self._logAppend(f"[INFO] Geometry: {geoPath}")
            self._logAppend(f"[INFO] Steps: {self.timestepsSpin.value}, niu={self.viscositySpin.value}")
            self._runScript("lbm_single_phase.py", args)

    def onStopSimulation(self):
        if self._process and self._process.state() != qt.QProcess.NotRunning:
            self._process.kill()
            self._logAppend("<span style='color:#f59e0b;'>[STOPPED] Simulation killed.</span>")
            self._setButtonsEnabled(True)

    def onLoadResult(self):
        """Load the last simulation velocity magnitude into Slicer."""
        tmp = Path(tempfile.gettempdir()) / "lbm_results"
        if not tmp.exists():
            slicer.util.warningDisplay(
                "No simulation results found.\nRun a simulation first.",
                windowTitle="AMORA",
            )
            return

        npy_files = sorted(tmp.glob("velocity_magnitude_*.npy"))
        if not npy_files:
            slicer.util.warningDisplay("No velocity result files found.", windowTitle="AMORA")
            return

        latest = npy_files[-1]
        self._logAppend(f"[LOAD] Loading {latest.name}...")

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            arr = np.load(str(latest))

            volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            volumeNode.SetName(f"LBM_Velocity_{latest.stem}")
            slicer.util.updateVolumeFromArray(volumeNode, arr)

            displayNode = volumeNode.GetDisplayNode()
            if displayNode:
                displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileColdToHotRainbow.txt")
                displayNode.SetAutoWindowLevel(True)

            slicer.util.setSliceViewerLayers(background=volumeNode)
            self._logAppend(
                f"<span style='color:#22c55e;'>[LOADED] {latest.name} "
                f"({arr.shape}) into Slicer</span>"
            )
        except Exception as e:
            self._logAppend(f"<span style='color:#ef4444;'>[ERROR] {e}</span>")
        finally:
            slicer.app.restoreOverrideCursor()

    # --- Flow Visualization ---

    def _loadVizFrames(self):
        """Discover all saved velocity magnitude snapshots."""
        tmp = Path(tempfile.gettempdir()) / "lbm_results"
        if not tmp.exists():
            return []
        return sorted(tmp.glob("velocity_magnitude_*.npy"))

    def _loadGeometry(self):
        """Load the saved geometry for overlay."""
        geoPath = Path(tempfile.gettempdir()) / "lbm_results" / "geometry.npy"
        if geoPath.exists():
            return np.load(str(geoPath))
        return None

    def _getSourceVolumeSpacing(self):
        """Get spacing and origin from the source volume node so results align."""
        # Try to find the original loaded volume
        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            name = node.GetName()
            if name and not name.startswith("LBM_"):
                return node.GetSpacing(), node.GetOrigin()
        return (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)

    def _applyVolumeGeometry(self, node):
        """Copy spacing and origin from source volume to result volume."""
        spacing, origin = self._getSourceVolumeSpacing()
        node.SetSpacing(*spacing)
        node.SetOrigin(*origin)

    def _setupRockOverlay(self):
        """Create/update rock structure volume node for overlay."""
        if not self.overlayRockCheck.checked:
            return
        geo = self._loadGeometry()
        if geo is None:
            return

        if self._vizRockNode is None or slicer.mrmlScene.GetNodeByID(
            self._vizRockNode.GetID() if self._vizRockNode else ""
        ) is None:
            self._vizRockNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            self._vizRockNode.SetName("LBM_RockStructure")

        # Solid=255, pore=0 for good contrast
        rock_vis = (geo.astype(np.uint8) * 255)
        slicer.util.updateVolumeFromArray(self._vizRockNode, rock_vis)
        self._applyVolumeGeometry(self._vizRockNode)

        displayNode = self._vizRockNode.GetDisplayNode()
        if displayNode:
            displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
            displayNode.SetAutoWindowLevel(False)
            displayNode.SetWindow(255)
            displayNode.SetLevel(128)

    def _ensureVizNode(self, arr):
        """Create or reuse flow volume node for animation playback."""
        if self._vizNode is None or slicer.mrmlScene.GetNodeByID(
            self._vizNode.GetID() if self._vizNode else ""
        ) is None:
            self._vizNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            self._vizNode.SetName("LBM_FlowAnimation")

        slicer.util.updateVolumeFromArray(self._vizNode, arr)
        self._applyVolumeGeometry(self._vizNode)

        displayNode = self._vizNode.GetDisplayNode()
        if displayNode:
            displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileColdToHotRainbow.txt")
            if not hasattr(self, '_vizWL') or self._vizWL is None:
                displayNode.SetAutoWindowLevel(True)
                # Force compute so we can read back
                displayNode.AutoWindowLevelOff()
                displayNode.AutoWindowLevelOn()
                self._vizWL = (displayNode.GetWindow(), displayNode.GetLevel())
            else:
                displayNode.SetAutoWindowLevel(False)
                displayNode.SetWindow(self._vizWL[0])
                displayNode.SetLevel(self._vizWL[1])

        return self._vizNode

    def _setupInternalView(self):
        """Internal view: slice viewers showing flow inside the rock."""
        # Set up foreground (flow) + background (rock) overlay in slice views
        if self.overlayRockCheck.checked and self._vizRockNode:
            slicer.util.setSliceViewerLayers(
                background=self._vizRockNode,
                foreground=self._vizNode,
                foregroundOpacity=0.6,
            )
        else:
            slicer.util.setSliceViewerLayers(background=self._vizNode)

        # Make sure 3D view is hidden for internal mode
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutThreeOverThreeView)

    def _setupExternalView(self):
        """External view: 3D volume rendering showing flow around/through rock."""
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalWidescreenView)

        # Enable volume rendering for the flow node
        try:
            volRenLogic = slicer.modules.volumerendering.logic()
        except Exception as e:
            self._logAppend(
                "<span style='color:#ef4444;'>[ERROR] Volume Rendering module not available. "
                "A C++ rebuild may be needed: cmake --build C:/W/AR/Slicer-build "
                "--target qSlicerVolumeRenderingModule --config Release</span>"
            )
            # Fallback to internal view
            self._setupInternalView()
            return

        # Flow volume rendering
        if self._vizNode:
            displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(self._vizNode)
            if displayNode is None:
                displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(self._vizNode)

            if displayNode:
                displayNode.SetVisibility(True)

                # Set transfer function: transparent at 0, colored for flow
                volPropNode = displayNode.GetVolumePropertyNode()
                if volPropNode:
                    volProp = volPropNode.GetVolumeProperty()
                    # Get data range
                    arr = slicer.util.arrayFromVolume(self._vizNode)
                    vmax = float(arr.max()) if arr.max() > 0 else 1.0

                    # Opacity: 0 at zero (transparent), ramp up for flow
                    otf = volProp.GetScalarOpacity()
                    otf.RemoveAllPoints()
                    otf.AddPoint(0.0, 0.0)          # zero velocity = transparent
                    otf.AddPoint(vmax * 0.01, 0.0)   # near-zero = still transparent
                    otf.AddPoint(vmax * 0.05, 0.15)   # low flow = slightly visible
                    otf.AddPoint(vmax * 0.3, 0.4)    # medium flow
                    otf.AddPoint(vmax, 0.8)           # high flow = opaque

                    # Color: blue (low) -> cyan -> yellow -> red (high)
                    ctf = volProp.GetRGBTransferFunction()
                    ctf.RemoveAllPoints()
                    ctf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                    ctf.AddRGBPoint(vmax * 0.1, 0.0, 0.0, 0.8)    # blue
                    ctf.AddRGBPoint(vmax * 0.25, 0.0, 0.7, 1.0)   # cyan
                    ctf.AddRGBPoint(vmax * 0.5, 1.0, 1.0, 0.0)    # yellow
                    ctf.AddRGBPoint(vmax * 0.75, 1.0, 0.4, 0.0)   # orange
                    ctf.AddRGBPoint(vmax, 1.0, 0.0, 0.0)          # red

        # Rock volume rendering (semi-transparent gray)
        if self.overlayRockCheck.checked and self._vizRockNode:
            rockDisplay = volRenLogic.GetFirstVolumeRenderingDisplayNode(self._vizRockNode)
            if rockDisplay is None:
                rockDisplay = volRenLogic.CreateDefaultVolumeRenderingNodes(self._vizRockNode)

            if rockDisplay:
                rockDisplay.SetVisibility(True)
                volPropNode = rockDisplay.GetVolumePropertyNode()
                if volPropNode:
                    volProp = volPropNode.GetVolumeProperty()
                    otf = volProp.GetScalarOpacity()
                    otf.RemoveAllPoints()
                    otf.AddPoint(0, 0.0)       # pore = transparent
                    otf.AddPoint(127, 0.0)
                    otf.AddPoint(128, 0.08)    # solid = slightly visible
                    otf.AddPoint(255, 0.12)

                    ctf = volProp.GetRGBTransferFunction()
                    ctf.RemoveAllPoints()
                    ctf.AddRGBPoint(0, 0.0, 0.0, 0.0)
                    ctf.AddRGBPoint(255, 0.7, 0.7, 0.7)  # light gray

    def _setupIsosurfaceView(self):
        """Isosurface view: 3D contour surfaces of velocity magnitude (like Paraview)."""
        import vtk

        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalWidescreenView)

        arr = slicer.util.arrayFromVolume(self._vizNode)
        vmax = float(arr.max()) if arr.max() > 0 else 1.0

        # Clean up previous isosurface models
        for oldNode in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if oldNode.GetName().startswith("LBM_Iso_"):
                slicer.mrmlScene.RemoveNode(oldNode)

        # Get spacing from source volume
        spacing, origin = self._getSourceVolumeSpacing()

        # Create VTK image data from numpy array
        imageData = vtk.vtkImageData()
        dims = arr.shape  # ZYX
        imageData.SetDimensions(dims[2], dims[1], dims[0])
        imageData.SetSpacing(spacing[0], spacing[1], spacing[2])
        imageData.SetOrigin(origin[0], origin[1], origin[2])

        from vtk.util.numpy_support import numpy_to_vtk
        flat = arr.flatten(order='C').astype(np.float32)
        vtkArr = numpy_to_vtk(flat, deep=True)
        vtkArr.SetName("velocity")
        imageData.GetPointData().SetScalars(vtkArr)

        # Color map: blue -> cyan -> yellow -> red
        nLevels = self.isoLevelsSpin.value
        colors = []
        for i in range(nLevels):
            t = (i + 1) / (nLevels + 1)
            if t < 0.33:
                r, g, b = 0.0, t / 0.33, 1.0
            elif t < 0.66:
                r, g, b = (t - 0.33) / 0.33, 1.0, 1.0 - (t - 0.33) / 0.33
            else:
                r, g, b = 1.0, 1.0 - (t - 0.66) / 0.34, 0.0
            colors.append((r, g, b))

        # Create isosurfaces at different velocity levels
        for i in range(nLevels):
            frac = (i + 1) / (nLevels + 1)
            isoValue = vmax * frac

            contour = vtk.vtkContourFilter()
            contour.SetInputData(imageData)
            contour.SetValue(0, isoValue)
            contour.Update()

            if contour.GetOutput().GetNumberOfCells() == 0:
                continue

            # Smooth the isosurface
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(contour.GetOutputPort())
            smoother.SetNumberOfIterations(15)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetPassBand(0.1)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()

            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(smoother.GetOutputPort())
            normals.ComputePointNormalsOn()
            normals.Update()

            # Create model node in Slicer
            modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            modelNode.SetName(f"LBM_Iso_{i}_{isoValue:.4f}")
            modelNode.SetAndObservePolyData(normals.GetOutput())
            modelNode.CreateDefaultDisplayNodes()

            displayNode = modelNode.GetDisplayNode()
            r, g, b = colors[i]
            displayNode.SetColor(r, g, b)
            displayNode.SetOpacity(0.6 if nLevels > 1 else 0.8)
            displayNode.SetEdgeVisibility(False)
            displayNode.SetBackfaceCulling(True)

        # Also show rock as semi-transparent model if enabled
        if self.overlayRockCheck.checked and self._vizRockNode:
            geo = self._loadGeometry()
            if geo is not None:
                # Clean old rock iso
                for oldNode in slicer.util.getNodesByClass("vtkMRMLModelNode"):
                    if oldNode.GetName() == "LBM_Iso_Rock":
                        slicer.mrmlScene.RemoveNode(oldNode)

                geoImage = vtk.vtkImageData()
                geoImage.SetDimensions(dims[2], dims[1], dims[0])
                geoImage.SetSpacing(spacing[0], spacing[1], spacing[2])
                geoImage.SetOrigin(origin[0], origin[1], origin[2])

                geoFlat = geo.flatten(order='C').astype(np.float32)
                geoVtk = numpy_to_vtk(geoFlat, deep=True)
                geoVtk.SetName("solid")
                geoImage.GetPointData().SetScalars(geoVtk)

                rockContour = vtk.vtkContourFilter()
                rockContour.SetInputData(geoImage)
                rockContour.SetValue(0, 0.5)  # solid boundary
                rockContour.Update()

                if rockContour.GetOutput().GetNumberOfCells() > 0:
                    rockModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
                    rockModel.SetName("LBM_Iso_Rock")
                    rockModel.SetAndObservePolyData(rockContour.GetOutput())
                    rockModel.CreateDefaultDisplayNodes()
                    rockDisp = rockModel.GetDisplayNode()
                    rockDisp.SetColor(0.75, 0.7, 0.65)  # sandstone color
                    rockDisp.SetOpacity(0.15)
                    rockDisp.SetBackfaceCulling(False)

        self._logAppend(f"[VIZ] Isosurface view: {nLevels} levels, vmax={vmax:.6f}")

    def _showFrame(self, idx):
        """Load and display frame at index idx."""
        if idx < 0 or idx >= len(self._vizFrames):
            return
        arr = np.load(str(self._vizFrames[idx]))
        self._ensureVizNode(arr)
        self.frameLabel.setText(f"Frame: {idx + 1}/{len(self._vizFrames)}")

        vizMode = self.vizModeCombo.currentIndex
        if vizMode == 0:
            self._setupInternalView()
        elif vizMode == 1:
            self._setupExternalView()
        else:
            self._setupIsosurfaceView()

    def _onFrameChanged(self, val):
        """Slider moved manually."""
        if self._vizFrames:
            self._showFrame(val)

    def onPlayAnimation(self):
        """Start playback of velocity snapshots."""
        self._vizFrames = self._loadVizFrames()
        if not self._vizFrames:
            slicer.util.warningDisplay(
                "No simulation snapshots found.\nRun a simulation first.",
                windowTitle="AMORA",
            )
            return

        self._vizWL = None  # Reset W/L on first frame
        self._setupRockOverlay()
        self.frameSlider.setRange(0, len(self._vizFrames) - 1)
        self.frameSlider.setValue(0)
        self._showFrame(0)

        interval_ms = int(1000.0 / self.fpsSpin.value)
        self._animTimer.start(interval_ms)
        self.playBtn.setEnabled(False)
        self.pauseBtn.setEnabled(True)
        self._logAppend(
            f"[PLAY] Animating {len(self._vizFrames)} frames at {self.fpsSpin.value} FPS"
        )

    def onPauseAnimation(self):
        """Pause playback."""
        self._animTimer.stop()
        self.playBtn.setEnabled(True)
        self.pauseBtn.setEnabled(False)

    def _onAnimTick(self):
        """Advance to next frame."""
        idx = self.frameSlider.value + 1
        if idx >= len(self._vizFrames):
            idx = 0  # Loop
        self.frameSlider.setValue(idx)

    def onExportVideo(self):
        """Export animation frames as MP4 video."""
        self._vizFrames = self._loadVizFrames()
        if not self._vizFrames:
            slicer.util.warningDisplay(
                "No simulation snapshots found.\nRun a simulation first.",
                windowTitle="AMORA",
            )
            return

        savePath = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save Flow Animation",
            str(Path.home() / "lbm_flow.mp4"),
            "MP4 Video (*.mp4);;GIF Animation (*.gif)",
        )
        if not savePath:
            return

        self._logAppend(f"[EXPORT] Rendering {len(self._vizFrames)} frames...")
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

        try:
            try:
                import matplotlib
            except ImportError:
                slicer.util.pip_install("matplotlib")
                import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            geo = self._loadGeometry()
            frames_np = []
            vmax_global = 0.0
            for f in self._vizFrames:
                arr = np.load(str(f))
                vmax_global = max(vmax_global, float(arr.max()))

            sample = np.load(str(self._vizFrames[0]))
            nz = sample.shape[0]
            slice_indices = [nz // 4, nz // 2, 3 * nz // 4]

            for fi, fpath in enumerate(self._vizFrames):
                arr = np.load(str(fpath))

                fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=120)
                for ax, si in zip(axes, slice_indices):
                    if geo is not None:
                        ax.imshow(geo[si], cmap="gray", vmin=0, vmax=1, alpha=0.4)
                    flow = np.ma.masked_where(arr[si] < vmax_global * 0.005, arr[si])
                    im = ax.imshow(
                        flow, cmap="inferno",
                        vmin=0, vmax=vmax_global,
                        interpolation="bilinear",
                    )
                    ax.set_title(f"Z = {si}", fontsize=10)
                    ax.axis("off")

                fig.suptitle(
                    f"LBM Flow - Frame {fi + 1}/{len(self._vizFrames)}",
                    fontsize=13, fontweight="bold",
                )
                fig.colorbar(im, ax=axes, shrink=0.8, label="Velocity (LU)")
                fig.tight_layout()

                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
                frames_np.append(buf.copy())
                plt.close(fig)

            if not frames_np:
                self._logAppend("<span style='color:#ef4444;'>[ERROR] No frames rendered</span>")
                return

            if savePath.endswith(".gif"):
                # Use matplotlib to save animated GIF (no PIL needed)
                fig_anim, ax_anim = plt.subplots(figsize=(14, 4), dpi=120)
                ax_anim.axis("off")
                ims = []
                from matplotlib.animation import ArtistAnimation
                for buf in frames_np:
                    im_art = ax_anim.imshow(buf)
                    ims.append([im_art])
                anim = ArtistAnimation(fig_anim, ims, interval=int(1000 / self.fpsSpin.value))
                anim.save(savePath, writer="pillow")
                plt.close(fig_anim)
            else:
                try:
                    import imageio
                except ImportError:
                    slicer.util.pip_install("imageio[ffmpeg]")
                    import imageio
                writer = imageio.get_writer(savePath, fps=self.fpsSpin.value)
                for frame in frames_np:
                    writer.append_data(frame)
                writer.close()

            self._logAppend(
                f"<span style='color:#22c55e;'>[DONE] Exported to {savePath}</span>"
            )

        except Exception as e:
            self._logAppend(f"<span style='color:#ef4444;'>[ERROR] {e}</span>")
        finally:
            slicer.app.restoreOverrideCursor()


# ===========================================================================
# LOGIC & TEST
# ===========================================================================

class AmoraSimulationLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class AmoraSimulationTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.delayDisplay("AmoraSimulation test passed (placeholder).")
