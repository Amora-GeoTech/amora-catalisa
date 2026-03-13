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
        self.parent.categories = ["AMORA"]
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
        self.viscositySpin.setToolTip("Kinematic viscosity in lattice units (default 1/6)")
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
        is_grayscale = self.grayscaleCheck.isChecked

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

        # Find latest velocity file
        npy_files = sorted(tmp.glob("velocity_magnitude_*.npy"))
        if not npy_files:
            slicer.util.warningDisplay(
                "No velocity result files found.",
                windowTitle="AMORA",
            )
            return

        latest = npy_files[-1]
        self._logAppend(f"[LOAD] Loading {latest.name}...")

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            arr = np.load(str(latest))

            # Create volume node
            volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            volumeNode.SetName(f"LBM_Velocity_{latest.stem}")
            slicer.util.updateVolumeFromArray(volumeNode, arr)

            # Set colormap to a flow-friendly one
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
