"""
AmoraProcessing - Processing Tools
=======================================
Histogram, segmentation, ROI generation, and GIF export.
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
import vtk
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

class AmoraProcessing(ScriptedLoadableModule):

    def __init__(self, parent: Optional[qt.QWidget]):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Processing"
        self.parent.categories = [""]
        self.parent.dependencies = []
        self.parent.contributors = [
            "LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia"
        ]
        self.parent.helpText = "Histogram, segmentation, ROI generation, GIF export."
        self.parent.acknowledgementText = "Projeto Catalisa ICT / CPGF / UFPA"
        self.parent.hidden = False


# ===========================================================================
# PROCESSING TOOLS
# ===========================================================================

class AmoraProcessingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent: Optional[qt.QWidget] = None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._process = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # --- Header ---
        headerLabel = qt.QLabel("Processing Tools")
        headerLabel.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #22d3ee; padding: 8px;"
        )
        self.layout.addWidget(headerLabel)

        descLabel = qt.QLabel("Histogram, segmentation, ROI generation, GIF export.")
        descLabel.setStyleSheet("font-size: 12px; color: #94a3b8; padding: 0 8px 8px 8px;")
        self.layout.addWidget(descLabel)

        # --- Segmentation ---
        segCollapsible = ctk.ctkCollapsibleButton()
        segCollapsible.text = "Segmentation"
        self.layout.addWidget(segCollapsible)
        segLayout = qt.QFormLayout(segCollapsible)

        self.histBtn = qt.QPushButton("Compute Histogram")
        self.histBtn.setToolTip("Compute and display the histogram of the loaded volume")
        self.histBtn.clicked.connect(self.onComputeHistogram)
        segLayout.addRow("", self.histBtn)

        # Method selection
        self.segMethodCombo = qt.QComboBox()
        self.segMethodCombo.addItems([
            "Otsu (binary: pore vs solid)",
            "Multi-Otsu (3 classes)",
            "Multi-Otsu (4 classes)",
            "Hysteresis (noise-robust)",
        ])
        self.segMethodCombo.currentIndexChanged.connect(self._onSegMethodChanged)
        segLayout.addRow("Method:", self.segMethodCombo)

        # Hysteresis params (hidden by default)
        self.hystWidget = qt.QWidget()
        hystLayout = qt.QFormLayout(self.hystWidget)
        hystLayout.setContentsMargins(0, 0, 0, 0)
        self.hystLowSpin = qt.QDoubleSpinBox()
        self.hystLowSpin.setRange(0.0, 1.0)
        self.hystLowSpin.setValue(0.3)
        self.hystLowSpin.setDecimals(2)
        self.hystLowSpin.setSingleStep(0.05)
        self.hystLowSpin.setToolTip("Low threshold fraction (0-1). Voxels above this are candidates.")
        hystLayout.addRow("Low fraction:", self.hystLowSpin)
        self.hystHighSpin = qt.QDoubleSpinBox()
        self.hystHighSpin.setRange(0.0, 1.0)
        self.hystHighSpin.setValue(0.7)
        self.hystHighSpin.setDecimals(2)
        self.hystHighSpin.setSingleStep(0.05)
        self.hystHighSpin.setToolTip("High threshold fraction (0-1). Seeds for connected regions.")
        hystLayout.addRow("High fraction:", self.hystHighSpin)
        self.hystWidget.setVisible(False)
        segLayout.addRow("", self.hystWidget)

        # Auto-load result
        self.autoLoadCheck = qt.QCheckBox("Auto-load result into Slicer")
        self.autoLoadCheck.checked = True
        segLayout.addRow("", self.autoLoadCheck)

        # Save location (folder)
        self.segSavePath = ctk.ctkPathLineEdit()
        self.segSavePath.filters = ctk.ctkPathLineEdit.Dirs
        self.segSavePath.setToolTip("Select folder to save segmented volume (leave empty for temp dir)")
        segLayout.addRow("Save folder:", self.segSavePath)

        # Run button
        self.segRunBtn = qt.QPushButton("Run Segmentation")
        self.segRunBtn.setToolTip("Run the selected segmentation method")
        self.segRunBtn.setStyleSheet(
            "font-weight: bold; padding: 8px; "
            "background-color: #1e3a5f; color: #60a5fa; "
            "border: 1px solid #60a5fa; border-radius: 6px;"
        )
        self.segRunBtn.clicked.connect(self.onRunSegmentation)
        segLayout.addRow("", self.segRunBtn)

        self.applyWLBtn = qt.QPushButton("Apply Window/Level to Volume")
        self.applyWLBtn.setToolTip(
            "Make the current brightness/contrast (Window/Level) permanent on the volume data"
        )
        self.applyWLBtn.clicked.connect(self.onApplyWindowLevel)
        segLayout.addRow("", self.applyWLBtn)

        # --- Regions of Interest ---
        roiCollapsible = ctk.ctkCollapsibleButton()
        roiCollapsible.text = "Regions of Interest"
        self.layout.addWidget(roiCollapsible)
        roiLayout = qt.QFormLayout(roiCollapsible)

        self.roiSizeSpin = qt.QSpinBox()
        self.roiSizeSpin.setRange(8, 512)
        self.roiSizeSpin.setValue(64)
        self.roiSizeSpin.setToolTip("Size of each cubic ROI (voxels)")
        roiLayout.addRow("ROI Size:", self.roiSizeSpin)

        self.numRoisSpin = qt.QSpinBox()
        self.numRoisSpin.setRange(1, 100)
        self.numRoisSpin.setValue(5)
        self.numRoisSpin.setToolTip("Number of ROIs to generate")
        roiLayout.addRow("Number of ROIs:", self.numRoisSpin)

        self.genRoisBtn = qt.QPushButton("Generate ROIs")
        self.genRoisBtn.setToolTip("Generate random cubic ROIs from the volume")
        self.genRoisBtn.clicked.connect(self.onGenerateROIs)
        roiLayout.addRow("", self.genRoisBtn)

        self.roiIdSpin = qt.QSpinBox()
        self.roiIdSpin.setRange(0, 999)
        self.roiIdSpin.setValue(0)
        roiLayout.addRow("ROI ID:", self.roiIdSpin)

        self.plotRoiBtn = qt.QPushButton("Plot ROI by ID")
        self.plotRoiBtn.clicked.connect(self.onPlotROI)
        roiLayout.addRow("", self.plotRoiBtn)

        # --- GIF Export ---
        gifCollapsible = ctk.ctkCollapsibleButton()
        gifCollapsible.text = "GIF Export"
        gifCollapsible.collapsed = True
        self.layout.addWidget(gifCollapsible)
        gifLayout = qt.QVBoxLayout(gifCollapsible)

        self.gifTensorBtn = qt.QPushButton("Export Tensor GIF")
        self.gifTensorBtn.clicked.connect(self.onExportTensorGif)
        gifLayout.addWidget(self.gifTensorBtn)

        self.gifBinBtn = qt.QPushButton("Export Binarized GIF")
        self.gifBinBtn.clicked.connect(self.onExportBinarizedGif)
        gifLayout.addWidget(self.gifBinBtn)

        self.gifRoiBtn = qt.QPushButton("Export ROI GIF")
        self.gifRoiBtn.clicked.connect(self.onExportROIGif)
        gifLayout.addWidget(self.gifRoiBtn)

        # --- Session ---
        sessionCollapsible = ctk.ctkCollapsibleButton()
        sessionCollapsible.text = "Session"
        sessionCollapsible.collapsed = True
        self.layout.addWidget(sessionCollapsible)
        sessionLayout = qt.QVBoxLayout(sessionCollapsible)

        self.clearRamBtn = qt.QPushButton("Clear RAM Cache")
        self.clearRamBtn.clicked.connect(self.onClearRAM)
        sessionLayout.addWidget(self.clearRamBtn)

        # --- Processing Log ---
        logCollapsible = ctk.ctkCollapsibleButton()
        logCollapsible.text = "Processing Log"
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
        self.logic = AmoraProcessingLogic()

    # --- Lifecycle ---

    def enter(self):
        panel = slicer.util.mainWindow().findChild(qt.QDockWidget, "PanelDockWidget")
        if panel:
            panel.setVisible(True)

    def exit(self):
        pass

    def cleanup(self):
        pass

    # --- Helpers ---

    def _logAppend(self, msg):
        self.logOutput.append(msg)
        sb = self.logOutput.verticalScrollBar()
        sb.setValue(sb.maximum)

    def _cacheExists(self, name="_tensor_cache.npy"):
        tmp = Path(tempfile.gettempdir())
        return (tmp / name).exists() or (tmp / f"{name}.ok").exists()

    def _setButtonsEnabled(self, enabled):
        self.progressBar.setVisible(not enabled)
        for btn in [self.histBtn, self.segRunBtn, self.applyWLBtn,
                     self.genRoisBtn, self.plotRoiBtn, self.gifTensorBtn,
                     self.gifBinBtn, self.gifRoiBtn, self.clearRamBtn]:
            btn.setEnabled(enabled)

    def _getScriptsDir(self):
        """Get the path to the processing scripts directory."""
        # In the built app, resources are installed alongside the module .py
        moduleDir = os.path.dirname(__file__)
        scriptsDir = os.path.join(moduleDir, "Resources", "Scripts")
        if os.path.isdir(scriptsDir):
            return scriptsDir
        # Fallback: try the source tree location
        scriptsDir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "AmoraProcessing", "Resources", "Scripts"
        )
        return scriptsDir

    def _runScript(self, scriptName, args=None):
        """Run a processing script as a subprocess."""
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
        txt = self._process.readAllStandardOutput().data().decode("utf-8", errors="ignore").strip()
        if txt:
            for line in txt.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    if line.startswith("{"):
                        data = json.loads(line)
                        if "png" in data:
                            self._logAppend(
                                f"<span style='color:#22c55e;'>[RESULT] {data['png']}</span>"
                            )
                        if data.get("ok") and "output" in data:
                            self._lastSegResult = data
                            self._logAppend(
                                f"<span style='color:#22c55e;'>[SEG] {data.get('info', 'Done')}</span>"
                            )
                        continue
                except Exception:
                    pass
                self._logAppend(line)

    def _onStderr(self):
        if not self._process:
            return
        txt = self._process.readAllStandardError().data().decode("utf-8", errors="ignore").strip()
        if txt:
            self._logAppend(f"<span style='color:#ef4444;'>{txt}</span>")

    def _onFinished(self, exitCode, exitStatus=None):
        if exitCode == 0:
            self._logAppend("<span style='color:#22c55e;'>[DONE] Success.</span>")
            # Auto-load segmentation result if requested
            if getattr(self, '_segAutoLoad', False) and hasattr(self, '_lastSegResult'):
                self._autoLoadSegResult()
                self._segAutoLoad = False
        else:
            self._logAppend(
                f"<span style='color:#f59e0b;'>[DONE] Exit code {exitCode}</span>"
            )
        self._setButtonsEnabled(True)

    def _autoLoadSegResult(self):
        """Load segmentation result into Slicer as a new volume."""
        data = getattr(self, '_lastSegResult', None)
        if not data:
            return
        outPath = data.get("output", "")
        if not outPath or not os.path.exists(outPath):
            return

        try:
            arr = np.load(outPath)
            nodeName = f"Segmented_{data.get('method', 'result')}"
            volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            volumeNode.SetName(nodeName)
            slicer.util.updateVolumeFromArray(volumeNode, arr)

            # Copy geometry from original volume if available
            origNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if origNode and origNode != volumeNode:
                volumeNode.SetOrigin(origNode.GetOrigin())
                volumeNode.SetSpacing(origNode.GetSpacing())
                ijkToRas = vtk.vtkMatrix4x4()
                origNode.GetIJKToRASMatrix(ijkToRas)
                volumeNode.SetIJKToRASMatrix(ijkToRas)

            displayNode = volumeNode.GetDisplayNode()
            if displayNode:
                nLabels = data.get("unique_labels", 2)
                if nLabels <= 2:
                    # Binary: grayscale with good contrast
                    displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
                    displayNode.SetAutoWindowLevel(False)
                    displayNode.SetWindow(1)
                    displayNode.SetLevel(0.5)
                else:
                    # Multi-class: use label colormap
                    displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels")
                    displayNode.SetAutoWindowLevel(True)

            slicer.util.setSliceViewerLayers(background=volumeNode)
            slicer.util.resetSliceViews()
            self._logAppend(
                f"<span style='color:#22c55e;'>[LOADED] {nodeName} "
                f"({arr.shape}) into Slicer</span>"
            )
        except Exception as e:
            self._logAppend(f"<span style='color:#ef4444;'>[ERROR] Auto-load: {e}</span>")

    # --- Processing Actions ---

    def onComputeHistogram(self):
        if not self._cacheExists():
            slicer.util.warningDisplay(
                "No data loaded. Load data in Digital Rock first.",
                windowTitle="AMORA",
            )
            return
        self._logAppend("[RUN] Computing histogram...")

        try:
            # Load data from cache
            cachePath = os.path.join(tempfile.gettempdir(), "_tensor_cache.npy")
            arr = np.load(cachePath, mmap_mode="r")
            data = np.asarray(arr).ravel()

            bins = 256
            if np.issubdtype(data.dtype, np.integer):
                bins = min(256, int(data.max() - data.min() + 1))
                bins = max(bins, 2)

            counts, edges = np.histogram(data, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])

            # Create table node with histogram data
            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Histogram")
            table = tableNode.GetTable()

            arrX = vtk.vtkFloatArray()
            arrX.SetName("Voxel Intensity")
            arrX.SetNumberOfTuples(len(centers))
            for i, v in enumerate(centers):
                arrX.SetValue(i, float(v))
            table.AddColumn(arrX)

            arrY = vtk.vtkFloatArray()
            arrY.SetName("Frequency")
            arrY.SetNumberOfTuples(len(counts))
            for i, v in enumerate(counts):
                arrY.SetValue(i, float(v))
            table.AddColumn(arrY)

            # Create plot series
            plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLPlotSeriesNode", "Histogram"
            )
            plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
            plotSeriesNode.SetXColumnName("Voxel Intensity")
            plotSeriesNode.SetYColumnName("Frequency")
            plotSeriesNode.SetPlotType(plotSeriesNode.PlotTypeLine)
            plotSeriesNode.SetColor(0.2, 0.6, 1.0)

            # Create plot chart
            plotChartNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLPlotChartNode", "Histogram"
            )
            plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())
            plotChartNode.SetTitle("Voxel Intensity Histogram")
            plotChartNode.SetXAxisTitle("Intensity")
            plotChartNode.SetYAxisTitle("Frequency")

            # Show in plot view
            layoutManager = slicer.app.layoutManager()
            # Switch to a layout that includes the plot view
            layoutManager.setLayout(
                slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpPlotView
            )
            plotWidget = layoutManager.plotWidget(0)
            if plotWidget:
                plotView = plotWidget.mrmlPlotViewNode()
                plotView.SetPlotChartNodeID(plotChartNode.GetID())

            self._logAppend(
                f"<span style='color:#22c55e;'>[DONE] Histogram displayed in Plot view "
                f"({len(centers)} bins, {data.size} voxels)</span>"
            )
        except Exception as e:
            self._logAppend(
                f"<span style='color:#ef4444;'>[ERROR] Histogram: {e}</span>"
            )

    def _onSegMethodChanged(self, idx):
        """Show/hide hysteresis params based on selected method."""
        self.hystWidget.setVisible(idx == 3)

    def onRunSegmentation(self):
        if not self._cacheExists():
            slicer.util.warningDisplay("No data loaded.", windowTitle="AMORA")
            return

        methodIdx = self.segMethodCombo.currentIndex
        args = []
        if methodIdx == 0:
            args = ["--method", "otsu"]
            self._logAppend("[RUN] Otsu segmentation...")
        elif methodIdx == 1:
            args = ["--method", "multi_otsu", "--n-classes", "3"]
            self._logAppend("[RUN] Multi-Otsu (3 classes)...")
        elif methodIdx == 2:
            args = ["--method", "multi_otsu", "--n-classes", "4"]
            self._logAppend("[RUN] Multi-Otsu (4 classes)...")
        elif methodIdx == 3:
            args = [
                "--method", "hysteresis",
                "--low-frac", str(self.hystLowSpin.value),
                "--high-frac", str(self.hystHighSpin.value),
            ]
            self._logAppend("[RUN] Hysteresis segmentation...")

        # Custom output folder
        saveFolder = self.segSavePath.currentPath
        if saveFolder and os.path.isdir(saveFolder):
            methodName = ["otsu", "multi_otsu_3", "multi_otsu_4", "hysteresis"][methodIdx]
            outFile = os.path.join(saveFolder, f"segmented_{methodName}.npy")
            args.extend(["--output", outFile])

        # Auto-load flag
        self._segAutoLoad = self.autoLoadCheck.checked
        self._runScript("segment_otsu.py", args)

    def onApplyWindowLevel(self):
        """Make the current Window/Level adjustment permanent on the volume data."""
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        if not volumeNode:
            slicer.util.warningDisplay("No volume loaded.", windowTitle="AMORA")
            return

        displayNode = volumeNode.GetDisplayNode()
        if not displayNode:
            slicer.util.warningDisplay("No display node found.", windowTitle="AMORA")
            return

        window = displayNode.GetWindow()
        level = displayNode.GetLevel()
        vmin = level - window / 2.0
        vmax = level + window / 2.0

        if window <= 0 or vmin == vmax:
            slicer.util.warningDisplay(
                "Window/Level values are invalid. Adjust W/L on the slice view first.",
                windowTitle="AMORA",
            )
            return

        self._logAppend(
            f"[RUN] Applying Window/Level: W={window:.1f}, L={level:.1f} "
            f"(range [{vmin:.1f}, {vmax:.1f}])"
        )

        try:
            arr = slicer.util.arrayFromVolume(volumeNode)
            originalDtype = arr.dtype

            # Rescale: map [vmin, vmax] -> [0, output_max]
            if np.issubdtype(originalDtype, np.integer):
                info = np.iinfo(originalDtype)
                output_max = float(info.max)
            else:
                output_max = 1.0

            arrFloat = arr.astype(np.float64)
            rescaled = (arrFloat - vmin) / (vmax - vmin) * output_max
            rescaled = np.clip(rescaled, 0, output_max)
            arr[:] = rescaled.astype(originalDtype)

            slicer.util.arrayFromVolumeModified(volumeNode)

            # Reset W/L to full range so display matches the new data
            displayNode.AutoWindowLevelOn()

            # Update the temp cache so downstream tools use adjusted data
            tmp = Path(tempfile.gettempdir())
            cachePath = tmp / "_tensor_cache.npy"
            if cachePath.exists():
                np.save(str(cachePath), arr)
                self._logAppend("[INFO] Updated _tensor_cache.npy with adjusted data.")

            self._logAppend(
                "<span style='color:#22c55e;'>[DONE] Window/Level applied permanently.</span>"
            )

        except Exception as e:
            self._logAppend(
                f"<span style='color:#ef4444;'>[ERROR] {e}</span>"
            )

    def onGenerateROIs(self):
        if not self._cacheExists():
            slicer.util.warningDisplay("No data loaded.", windowTitle="AMORA")
            return
        size = str(self.roiSizeSpin.value)
        nrois = str(self.numRoisSpin.value)
        self._logAppend(f"[RUN] Generating {nrois} ROIs ({size}^3)...")
        self._runScript("generate_rois.py", [
            "--roi-size-z", size, "--roi-size-y", size, "--roi-size-x", size,
            "--num-rois", nrois,
        ])

    def onPlotROI(self):
        tmp = Path(tempfile.gettempdir())
        if not (tmp / "_rois.json.ok").exists():
            slicer.util.warningDisplay(
                "No ROIs found. Generate ROIs first.", windowTitle="AMORA",
            )
            return
        roiId = str(self.roiIdSpin.value)
        self._logAppend(f"[RUN] Plotting ROI #{roiId}...")
        self._runScript("plot_roi_vtk.py", ["--roi-id", roiId])

    def onExportTensorGif(self):
        if not self._cacheExists():
            slicer.util.warningDisplay("No data loaded.", windowTitle="AMORA")
            return
        self._logAppend("[RUN] Exporting tensor GIF...")
        self._runScript("export_gif.py", ["--basename", "_tensor_cache.npy"])

    def onExportBinarizedGif(self):
        tmp = Path(tempfile.gettempdir())
        if not (tmp / "_tensor_bin.ok").exists():
            slicer.util.warningDisplay(
                "No binarized data. Run Otsu segmentation first.",
                windowTitle="AMORA",
            )
            return
        self._logAppend("[RUN] Exporting binarized GIF...")
        self._runScript("export_gif.py", ["--basename", "_tensor_bin.npy"])

    def onExportROIGif(self):
        tmp = Path(tempfile.gettempdir())
        if not (tmp / "_rois.json.ok").exists():
            slicer.util.warningDisplay(
                "No ROIs found. Generate ROIs first.", windowTitle="AMORA",
            )
            return
        roiId = self.roiIdSpin.value
        name = f"_roi_{roiId:03d}.npy"
        self._logAppend(f"[RUN] Exporting ROI #{roiId} GIF...")
        self._runScript("export_gif.py", ["--basename", name])

    def onClearRAM(self):
        self._logAppend("[INFO] Clearing RAM cache...")
        tmp = Path(tempfile.gettempdir())
        count = 0
        for pattern in ["_tensor_cache*", "_tensor_bin*", "_roi_*", "_rois.json*"]:
            for f in tmp.glob(pattern):
                try:
                    f.unlink()
                    count += 1
                except Exception:
                    pass
        self._logAppend(f"[INFO] Removed {count} cached files.")


# ===========================================================================
# LOGIC & TEST
# ===========================================================================

class AmoraProcessingLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class AmoraProcessingTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.delayDisplay("Testing cache detection...")
        tmp = Path(tempfile.gettempdir())
        testFile = tmp / "_amora_test_cache.npy"
        if testFile.exists():
            testFile.unlink()
        self.assertFalse(testFile.exists())
        np.save(str(testFile), np.zeros((4, 4, 4), dtype=np.float32))
        self.assertTrue(testFile.exists())
        testFile.unlink()
        self.delayDisplay("Cache detection test passed!")
