"""
AmoraDigitalRock - Digital Rock Analysis Tool
==================================================
Volume loading, 2D slice browsing, and 3D rendering.
Uses amora_io for file I/O.

LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia
"""

import os
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

class AmoraDigitalRock(ScriptedLoadableModule):

    def __init__(self, parent: Optional[qt.QWidget]):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Digital Rock"
        self.parent.categories = [""]
        self.parent.dependencies = []
        self.parent.contributors = [
            "LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia"
        ]
        self.parent.helpText = "Load volume data, browse 2D slices, render 3D volumes."
        self.parent.acknowledgementText = "Projeto Catalisa ICT / CPGF / UFPA"
        self.parent.hidden = False


# ===========================================================================
# DIGITAL ROCK TOOL
# ===========================================================================

class AmoraDigitalRockWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent: Optional[qt.QWidget] = None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._volumeNode = None
        self._cacheReady = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # --- Header ---
        headerLabel = qt.QLabel("Digital Rock Analysis")
        headerLabel.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #34d399; padding: 8px;"
        )
        self.layout.addWidget(headerLabel)

        descLabel = qt.QLabel("Load volume data, browse 2D slices, render 3D volumes.")
        descLabel.setStyleSheet("font-size: 12px; color: #94a3b8; padding: 0 8px 8px 8px;")
        self.layout.addWidget(descLabel)

        # --- Data Loading ---
        loadCollapsible = ctk.ctkCollapsibleButton()
        loadCollapsible.text = "Data Loading"
        self.layout.addWidget(loadCollapsible)
        loadLayout = qt.QFormLayout(loadCollapsible)

        self.fileSelector = ctk.ctkPathLineEdit()
        self.fileSelector.filters = ctk.ctkPathLineEdit.Files | ctk.ctkPathLineEdit.Dirs
        self.fileSelector.nameFilters = [
            "All supported (*.nc *.h5 *.hdf5 *.npy *.raw)",
            "NetCDF / HDF5 (*.nc *.h5 *.hdf5)",
            "NumPy arrays (*.npy)",
            "Raw binary (*.raw)",
        ]
        self.fileSelector.setToolTip("Select a volume file or directory")
        self.fileSelector.currentPathChanged.connect(self.onPathChanged)
        loadLayout.addRow("Input:", self.fileSelector)

        # --- Raw Parameters (Hidden by default) ---
        self.rawWidget = qt.QWidget()
        rawLayout = qt.QFormLayout(self.rawWidget)
        rawLayout.setContentsMargins(0, 0, 0, 0)

        self.rawDimX = qt.QSpinBox()
        self.rawDimX.setRange(1, 10000)
        self.rawDimX.setValue(512)
        self.rawDimY = qt.QSpinBox()
        self.rawDimY.setRange(1, 10000)
        self.rawDimY.setValue(512)
        self.rawDimZ = qt.QSpinBox()
        self.rawDimZ.setRange(1, 10000)
        self.rawDimZ.setValue(512)

        dimLayout = qt.QHBoxLayout()
        dimLayout.addWidget(qt.QLabel("X:"))
        dimLayout.addWidget(self.rawDimX)
        dimLayout.addWidget(qt.QLabel("Y:"))
        dimLayout.addWidget(self.rawDimY)
        dimLayout.addWidget(qt.QLabel("Z:"))
        dimLayout.addWidget(self.rawDimZ)
        rawLayout.addRow("Dimensions:", dimLayout)

        self.rawDtype = qt.QComboBox()
        self.rawDtype.addItems(["uint8", "uint16", "uint32", "int8", "int16", "int32", "float32", "float64"])
        self.rawDtype.setCurrentText("uint16")
        rawLayout.addRow("Data Type:", self.rawDtype)

        self.rawWidget.setVisible(False)
        loadLayout.addRow("", self.rawWidget)

        # --- Projection Parameters (Hidden by default) ---
        self.projWidget = qt.QWidget()
        projLayout = qt.QFormLayout(self.projWidget)
        projLayout.setContentsMargins(0, 0, 0, 0)

        self.projInfoLabel = qt.QLabel("")
        self.projInfoLabel.setWordWrap(True)
        self.projInfoLabel.setStyleSheet("color: #60a5fa; font-size: 11px; padding: 4px;")
        projLayout.addRow("", self.projInfoLabel)

        self.projMaxSpin = qt.QSpinBox()
        self.projMaxSpin.setRange(0, 100000)
        self.projMaxSpin.setValue(0)
        self.projMaxSpin.setToolTip("Max projections to load (0 = all). Use to limit memory.")
        projLayout.addRow("Max projections:", self.projMaxSpin)

        self.projWidget.setVisible(False)
        loadLayout.addRow("", self.projWidget)

        # --- Load Button ---
        self.loadBtn = qt.QPushButton("Load Volume")
        self.loadBtn.setToolTip("Load the selected file or directory")
        self.loadBtn.clicked.connect(self.onLoadClicked)
        loadLayout.addRow("", self.loadBtn)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.setStyleSheet("font-size: 11px; color: #94a3b8; padding: 4px 8px;")
        loadLayout.addRow("", self.statusLabel)

        # --- Volume Info ---
        infoCollapsible = ctk.ctkCollapsibleButton()
        infoCollapsible.text = "Volume Information"
        infoCollapsible.collapsed = True
        self.layout.addWidget(infoCollapsible)
        infoLayout = qt.QVBoxLayout(infoCollapsible)

        self.infoText = qt.QTextEdit()
        self.infoText.setReadOnly(True)
        self.infoText.setMaximumHeight(200)
        self.infoText.setPlainText("No volume loaded.")
        infoLayout.addWidget(self.infoText)

        # --- Visualization Controls ---
        vizCollapsible = ctk.ctkCollapsibleButton()
        vizCollapsible.text = "Visualization"
        vizCollapsible.collapsed = True
        self.layout.addWidget(vizCollapsible)
        vizLayout = qt.QFormLayout(vizCollapsible)

        # Colormap selector
        self.cmapCombo = qt.QComboBox()
        self._cmapEntries = [
            ("Grey", "vtkMRMLColorTableNodeGrey"),
            ("Viridis", "vtkMRMLColorTableNodeFileViridis.txt"),
            ("Magma", "vtkMRMLColorTableNodeFileMagma.txt"),
            ("Inferno", "vtkMRMLColorTableNodeFileInferno.txt"),
            ("Plasma", "vtkMRMLColorTableNodeFilePlasma.txt"),
            ("Warm 1", "vtkMRMLColorTableNodeWarm1"),
            ("Warm 2", "vtkMRMLColorTableNodeWarm2"),
            ("Cool 1", "vtkMRMLColorTableNodeCool1"),
            ("Cool 2", "vtkMRMLColorTableNodeCool2"),
            ("Rainbow", "vtkMRMLColorTableNodeRainbow"),
            ("Ocean", "vtkMRMLColorTableNodeOcean"),
            ("Red", "vtkMRMLColorTableNodeRed"),
            ("Green", "vtkMRMLColorTableNodeGreen"),
            ("Blue", "vtkMRMLColorTableNodeBlue"),
            ("Yellow", "vtkMRMLColorTableNodeYellow"),
            ("Hot to Cold Rainbow", "vtkMRMLColorTableNodeFileColdToHotRainbow.txt"),
        ]
        for name, _ in self._cmapEntries:
            self.cmapCombo.addItem(name)
        self.cmapCombo.currentIndexChanged.connect(self.onColormapChanged)
        vizLayout.addRow("Colormap:", self.cmapCombo)

        self.planeCombo = qt.QComboBox()
        self.planeCombo.addItems([
            "XY (Axial)", "XZ (Coronal)", "YZ (Sagittal)", "All 3 Planes"
        ])
        self.planeCombo.currentIndexChanged.connect(self.onPlaneChanged)
        vizLayout.addRow("Slice Plane:", self.planeCombo)

        self.sliceSlider = ctk.ctkSliderWidget()
        self.sliceSlider.singleStep = 1
        self.sliceSlider.minimum = 0
        self.sliceSlider.maximum = 100
        self.sliceSlider.value = 50
        self.sliceSlider.decimals = 0
        self.sliceSlider.setToolTip("Browse through volume slices")
        self.sliceSlider.valueChanged.connect(self.onSliceChanged)
        vizLayout.addRow("Slice Index:", self.sliceSlider)

        # --- Quick Actions ---
        actionsCollapsible = ctk.ctkCollapsibleButton()
        actionsCollapsible.text = "Quick Actions"
        actionsCollapsible.collapsed = True
        self.layout.addWidget(actionsCollapsible)
        actionsLayout = qt.QVBoxLayout(actionsCollapsible)

        self.saveNrrdBtn = qt.QPushButton("Save as NRRD")
        self.saveNrrdBtn.clicked.connect(self.onSaveNrrd)
        actionsLayout.addWidget(self.saveNrrdBtn)

        self.centerViewBtn = qt.QPushButton("Center 3D View")
        self.centerViewBtn.clicked.connect(self.onCenterView)
        actionsLayout.addWidget(self.centerViewBtn)

        self.layout.addStretch(1)
        self.logic = AmoraDigitalRockLogic()

    # --- Lifecycle ---

    def enter(self):
        panel = slicer.util.mainWindow().findChild(qt.QDockWidget, "PanelDockWidget")
        if panel:
            panel.setVisible(True)
        # Auto-detect volume if loaded externally (e.g. via toolbar Load Data)
        if self._volumeNode is None:
            node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if node:
                self._onVolumeLoaded(node)
                self.statusLabel.setText(f"Active: {node.GetName()}")

    def exit(self):
        pass

    def cleanup(self):
        pass

    # --- Data Loading (synchronous - safe for PythonQt) ---

    def onPathChanged(self, path):
        is_raw = str(path).lower().endswith(".raw")
        self.rawWidget.setVisible(is_raw)

        # Detect projection folder
        is_proj = False
        if path and os.path.isdir(path):
            from AmoraDigitalRockLib.amora_io import _detect_projection_folder, _parse_geometry_yaml
            detected = _detect_projection_folder(Path(path))
            if detected is not None:
                is_proj = True
                proj_dir, geom_path, _ = detected
                geom = _parse_geometry_yaml(geom_path)
                n_det = geom.get('nDetector', [0, 0])
                n_raw = len(list(proj_dir.glob("*.raw")))
                mode = geom.get('mode', '?')
                mem_gb = n_raw * n_det[0] * n_det[1] * 4 / (1024**3)
                self.projInfoLabel.setText(
                    f"CT Projections detected: {n_raw} files\n"
                    f"Detector: {n_det[0]} x {n_det[1]}, Mode: {mode}\n"
                    f"Estimated memory: {mem_gb:.1f} GB"
                )
        self.projWidget.setVisible(is_proj)
        if is_proj:
            self.rawWidget.setVisible(False)

    def onLoadClicked(self):
        filePath = self.fileSelector.currentPath
        if not filePath:
            slicer.util.errorDisplay("Please select a valid file.")
            return

        p = Path(filePath)
        if not p.exists():
            slicer.util.errorDisplay(f"Path does not exist:\n{filePath}")
            return

        kwargs = {}
        if self.rawWidget.visible:
            kwargs["shape"] = (self.rawDimZ.value, self.rawDimY.value, self.rawDimX.value)
            kwargs["dtype"] = self.rawDtype.currentText
        if self.projWidget.visible:
            max_proj = self.projMaxSpin.value
            if max_proj > 0:
                kwargs["max_projections"] = max_proj

        self.loadBtn.setEnabled(False)
        self.statusLabel.setText("Loading...")
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()

        try:
            from AmoraDigitalRockLib.amora_io import load_volume
            nodes = load_volume(filePath, **kwargs)

            if nodes:
                scalar_node = None
                for n in nodes:
                    if n is not None and n.IsA("vtkMRMLScalarVolumeNode"):
                        scalar_node = n
                        break
                node = scalar_node or nodes[0]

                if node is not None:
                    self._onVolumeLoaded(node)
                    self.statusLabel.setText(
                        f"Loaded: {node.GetName()} ({len(nodes)} volume(s))"
                    )
                else:
                    self.statusLabel.setText("No displayable volumes found.")
            else:
                self.statusLabel.setText("No volumes found in file.")

        except Exception as e:
            logging.error(f"[AMORA] Load failed: {e}", exc_info=True)
            slicer.util.errorDisplay(f"Failed to load:\n{e}")
            self.statusLabel.setText(f"Error: {e}")
        finally:
            slicer.app.restoreOverrideCursor()
            self.loadBtn.setEnabled(True)

    def _onVolumeLoaded(self, volumeNode):
        self._volumeNode = volumeNode
        imageData = volumeNode.GetImageData()
        if not imageData:
            return

        dims = imageData.GetDimensions()
        self.sliceSlider.maximum = max(dims) - 1
        self.sliceSlider.value = dims[2] // 2
        self._updateVolumeInfo(volumeNode)
        self._cacheForProcessing(volumeNode)

        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView
        )
        slicer.util.resetSliceViews()

    def _updateVolumeInfo(self, volumeNode):
        imageData = volumeNode.GetImageData()
        if not imageData:
            self.infoText.setPlainText("No image data.")
            return

        dims = imageData.GetDimensions()
        spacing = volumeNode.GetSpacing()
        origin = volumeNode.GetOrigin()
        arr = slicer.util.arrayFromVolume(volumeNode)

        # Compute stats synchronously
        vmin = float(arr.min())
        vmax = float(arr.max())
        vmean = float(arr.mean())
        vstd = float(arr.std())

        info = (
            f"Name: {volumeNode.GetName()}\n"
            f"Dimensions: {dims[0]} x {dims[1]} x {dims[2]}\n"
            f"Array shape: {arr.shape}\n"
            f"Dtype: {arr.dtype}\n"
            f"Voxels: {arr.size:,}\n"
            f"Spacing: ({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f})\n"
            f"Origin: ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})\n"
            f"Memory: {arr.nbytes / 1024 / 1024:.1f} MB\n"
            f"\nMin: {vmin:.4f}\n"
            f"Max: {vmax:.4f}\n"
            f"Mean: {vmean:.4f}\n"
            f"Std: {vstd:.4f}"
        )

        # Add CT projection info if available
        ct_mode = volumeNode.GetAttribute("AMORA.ct_mode")
        if ct_mode:
            n_proj = volumeNode.GetAttribute("AMORA.nProjections") or "?"
            n_det = volumeNode.GetAttribute("AMORA.nDetector") or "?"
            dso = volumeNode.GetAttribute("AMORA.DSO") or "?"
            dsd = volumeNode.GetAttribute("AMORA.DSD") or "?"
            n_voxel = volumeNode.GetAttribute("AMORA.nVoxel") or "?"
            d_voxel = volumeNode.GetAttribute("AMORA.dVoxel") or "?"
            info += (
                f"\n\n--- CT Acquisition ---\n"
                f"Mode: {ct_mode}\n"
                f"Projections: {n_proj}\n"
                f"Detector: {n_det}\n"
                f"DSO: {dso} mm\n"
                f"DSD: {dsd} mm\n"
                f"Recon voxels: {n_voxel}\n"
                f"Voxel size: {d_voxel} mm"
            )

        self.infoText.setPlainText(info)

    def _cacheForProcessing(self, volumeNode):
        """Save tensor cache synchronously for filtering/processing modules."""
        self._cacheReady = False
        try:
            arr = slicer.util.arrayFromVolume(volumeNode)
            cache_path = Path(tempfile.gettempdir()) / "_tensor_cache.npy"
            ok_path = Path(tempfile.gettempdir()) / "_tensor_cache.npy.ok"
            if ok_path.exists():
                ok_path.unlink()
            np.save(str(cache_path), arr.astype(np.float32))
            ok_path.touch()
            self._cacheReady = True
            logging.info("[AMORA] Cache ready.")
        except Exception as e:
            logging.warning(f"[AMORA] Cache failed: {e}")

    # --- Visualization ---

    def onColormapChanged(self, index):
        if not self._volumeNode:
            return
        if index < 0 or index >= len(self._cmapEntries):
            return
        _, colorNodeID = self._cmapEntries[index]
        displayNode = self._volumeNode.GetDisplayNode()
        if displayNode:
            displayNode.SetAndObserveColorNodeID(colorNodeID)

    def onPlaneChanged(self, index):
        layouts = [
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView,
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpGreenSliceView,
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView,
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView,
        ]
        slicer.app.layoutManager().setLayout(layouts[index])

    def onSliceChanged(self, value):
        if self._volumeNode:
            sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
            if sliceWidget:
                sliceWidget.mrmlSliceNode().SetSliceOffset(value)

    # --- Quick Actions ---

    def onSaveNrrd(self):
        if not self._volumeNode:
            slicer.util.warningDisplay("No volume loaded.")
            return
        savePath = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(), "Save Volume", "",
            "NRRD files (*.nrrd);;All files (*)",
        )
        if savePath:
            slicer.util.saveNode(self._volumeNode, savePath)
            slicer.util.infoDisplay(f"Saved to:\n{savePath}")

    def onCenterView(self):
        slicer.util.resetSliceViews()
        threeDWidget = slicer.app.layoutManager().threeDWidget(0)
        if threeDWidget:
            threeDWidget.threeDView().resetFocalPoint()


# ===========================================================================
# LOGIC & TEST
# ===========================================================================

class AmoraDigitalRockLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class AmoraDigitalRockTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.delayDisplay("Testing amora_io import...")
        from AmoraDigitalRockLib.amora_io import load_volume
        self.delayDisplay("amora_io imported OK")

        tensor = np.random.rand(32, 32, 32).astype(np.float32)
        tmpPath = os.path.join(tempfile.gettempdir(), "_amora_test_vol.npy")
        np.save(tmpPath, tensor)
        nodes = load_volume(tmpPath)
        self.assertTrue(len(nodes) == 1)
        self.delayDisplay(f"NPY load OK: {nodes[0].GetName()}")
        os.remove(tmpPath)
        self.delayDisplay("All tests passed!")
