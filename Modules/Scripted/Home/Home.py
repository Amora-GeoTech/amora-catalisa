"""
Home - AMORA-Digital Rock Application Controller
==================================================
Takes over the Slicer UI completely. This is NOT a plugin -
it IS the application. Slicer is just the rendering engine underneath.

The toolbar is the primary interface. The side panel only appears
when the user opens a specific tool (Digital Rock, Processing).

LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia
CPGF - Programa de Pos-graduacao em Geofisica
Projeto Catalisa ICT
"""

import os
import logging
from typing import Optional

import qt
import slicer
import SlicerCustomAppUtilities
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

# Import to ensure the files are available through the Qt resource system
from Resources import HomeResources  # noqa: F401


# ===========================================================================
# THEME CONFIGURATION
# ===========================================================================

THEME_NAMES = ["ocean", "dark", "alloyguard", "light"]
THEME_LABELS = {
    "dark": "AMORA Dark",
    "ocean": "Ocean",
    "alloyguard": "AlloyGuard",
    "light": "Light",
}


# ===========================================================================
# MODULE METADATA
# ===========================================================================

class Home(ScriptedLoadableModule):
    """The Home module orchestrates and styles the overall AMORA application."""

    def __init__(self, parent: Optional[qt.QWidget]):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Home"
        self.parent.categories = [""]
        self.parent.dependencies = []
        self.parent.contributors = [
            "LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia",
        ]
        self.parent.helpText = """AMORA-Digital Rock application controller."""
        self.parent.acknowledgementText = "Projeto Catalisa ICT / CPGF / UFPA"


# ===========================================================================
# APPLICATION CONTROLLER
# ===========================================================================

class HomeWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    Main application controller. Owns the AMORA toolbar, manages navigation,
    themes, and controls the side panel visibility.
    """

    _currentThemeIndex = 0
    _customStyleApplied = False

    @property
    def toolbarNames(self):
        return [str(k) for k in self._toolbars]

    _toolbars = {}

    def __init__(self, parent: Optional[qt.QWidget] = None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer)
        self.uiWidget = slicer.util.loadUI(self.resourcePath("UI/Home.ui"))
        self.layout.addWidget(self.uiWidget)
        self.ui = slicer.util.childWidgetVariables(self.uiWidget)

        # Create logic class
        self.logic = HomeLogic()

        # Dark palette propagation fix
        self.uiWidget.setPalette(slicer.util.mainWindow().style().standardPalette())

        # Remove unneeded UI elements and set up AMORA UI
        self.modifyWindowUI()

        # Apply default AMORA theme
        self._applyTheme(THEME_NAMES[HomeWidget._currentThemeIndex])

        # Home page content (shown in side panel when on Home)
        msg = qt.QLabel("AMORA-Digital Rock")
        msg.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #34d399; "
            "padding: 16px 8px 4px 8px;"
        )
        msg.setAlignment(qt.Qt.AlignHCenter)
        self.layout.addWidget(msg)

        sub = qt.QLabel("Select a tool from the toolbar above.")
        sub.setStyleSheet("font-size: 13px; color: #64748b; padding: 4px 8px;")
        sub.setAlignment(qt.Qt.AlignHCenter)
        self.layout.addWidget(sub)

        self.layout.addStretch()

    def cleanup(self):
        pass

    # --- Window UI Customization ---

    def modifyWindowUI(self):
        """Build AMORA toolbar."""
        self._buildToolbar()

    def _buildToolbar(self):
        """Create the AMORA toolbar - the ONLY navigation interface."""
        mw = slicer.util.mainWindow()
        existing = mw.findChild(qt.QToolBar, "AmoraToolBar")
        if existing:
            self.AmoraToolBar = existing
            return

        mainTB = slicer.util.findChild(mw, "MainToolBar")

        self.AmoraToolBar = qt.QToolBar("AmoraToolBar")
        self.AmoraToolBar.name = "AmoraToolBar"
        self.AmoraToolBar.setMovable(False)
        self.AmoraToolBar.setIconSize(qt.QSize(28, 28))
        mw.insertToolBar(mainTB, self.AmoraToolBar)
        self._toolbars["AmoraToolBar"] = self.AmoraToolBar

        # --- Logo (unclickable brand marker) ---
        logoPath = self.resourcePath("Icons/AMORA_logo.png")
        if os.path.isfile(logoPath):
            logoAction = self.AmoraToolBar.addAction(qt.QIcon(logoPath), "")
            logoAction.setEnabled(False)
            logoAction.setToolTip("AMORA-Digital Rock")

        self.AmoraToolBar.addSeparator()

        # --- Home button ---
        homeIcon = qt.QApplication.style().standardIcon(qt.QStyle.SP_DesktopIcon)
        homeAction = self.AmoraToolBar.addAction(homeIcon, " Home")
        homeAction.setToolTip("Back to 3D view (hide side panel)")
        homeAction.triggered.connect(self._goHome)

        self.AmoraToolBar.addSeparator()

        # --- Load Data (opens file dialog, loads via amora_io) ---
        loadIcon = qt.QApplication.style().standardIcon(qt.QStyle.SP_DialogOpenButton)
        loadAction = self.AmoraToolBar.addAction(loadIcon, " Load Data")
        loadAction.setToolTip("Load .nc, .npy, .h5 volume files")
        loadAction.triggered.connect(self._loadData)

        # --- Save ---
        saveIcon = qt.QApplication.style().standardIcon(qt.QStyle.SP_DialogSaveButton)
        saveAction = self.AmoraToolBar.addAction(saveIcon, " Save")
        saveAction.setToolTip("Save the active volume to file")
        saveAction.triggered.connect(self._saveData)

        self.AmoraToolBar.addSeparator()

        # --- Theme toggle ---
        restyleIcon = qt.QApplication.style().standardIcon(qt.QStyle.SP_BrowserReload)
        themeAction = self.AmoraToolBar.addAction(restyleIcon, "")
        themeAction.setToolTip("Change theme")
        themeAction.triggered.connect(self._cycleTheme)

        # --- Spacer + version ---
        spacer = qt.QWidget()
        spacer.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
        self.AmoraToolBar.addWidget(spacer)

        versionLabel = qt.QLabel(" AMORA v2.0 ")
        versionLabel.setStyleSheet(
            "color: #64748b; font-size: 10px; padding: 2px 8px; "
            "background: rgba(30, 41, 59, 120); border-radius: 6px;"
        )
        self.AmoraToolBar.addWidget(versionLabel)

    def insertToolBar(self, beforeToolBarName, name, title=None):
        """Helper to insert a new toolbar between existing ones."""
        beforeToolBar = slicer.util.findChild(slicer.util.mainWindow(), beforeToolBarName)
        if title is None:
            title = name
        toolBar = qt.QToolBar(title)
        toolBar.name = name
        slicer.util.mainWindow().insertToolBar(beforeToolBar, toolBar)
        self._toolbars[name] = toolBar
        return toolBar

    def toggleStyle(self, visible):
        if visible:
            self._applyTheme(THEME_NAMES[HomeWidget._currentThemeIndex])
        else:
            slicer.app.styleSheet = ""

    # --- Navigation ---

    def _goHome(self):
        """Switch to a clean FourUp view."""
        try:
            slicer.app.layoutManager().setLayout(
                slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalWidescreenView
            )
        except Exception:
            pass

    def _openTool(self, toolName):
        """Open a tool - show side panel with the tool's interface."""
        try:
            logging.info(f"[AMORA] Opening module: {toolName}")
            module = getattr(slicer.modules, toolName.lower(), None)
            if module is None:
                slicer.util.warningDisplay(
                    f"Module '{toolName}' not found.\n"
                    "Check that it is listed in CMakeLists.txt and was built.",
                    windowTitle="AMORA",
                )
                return
            slicer.util.selectModule(toolName)
            panel = slicer.util.mainWindow().findChild(qt.QDockWidget, "PanelDockWidget")
            if panel:
                panel.setVisible(True)
        except Exception as e:
            logging.error(f"[AMORA] Could not open {toolName}: {e}", exc_info=True)
            slicer.util.warningDisplay(
                f"Module '{toolName}' error:\n{e}",
                windowTitle="AMORA",
            )

    # --- Load / Save ---

    def _loadData(self):
        """Open file dialog and load volume via amora_io (bypasses ITK)."""
        filePath = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Open Volume File",
            "",
            "All supported (*.nc *.h5 *.hdf5 *.npy *.raw *.nrrd *.nii *.nii.gz);;"
            "NetCDF / HDF5 (*.nc *.h5 *.hdf5);;"
            "NumPy (*.npy);;"
            "Raw binary (*.raw);;"
            "NRRD (*.nrrd);;"
            "NIfTI (*.nii *.nii.gz);;"
            "All files (*)",
        )
        if not filePath:
            return

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            ext = filePath.lower().rsplit(".", 1)[-1]

            if ext == "raw":
                # Raw files need dimensions — prompt the user
                from AmoraDigitalRockLib.amora_io import load_volume
                dims_ok, kwargs = self._askRawParams()
                if not dims_ok:
                    return
                nodes = load_volume(filePath, **kwargs)
            elif ext in ("nc", "h5", "hdf5", "npy"):
                from AmoraDigitalRockLib.amora_io import load_volume
                nodes = load_volume(filePath)
            else:
                # Slicer native formats (nrrd, nii, etc.)
                node = slicer.util.loadVolume(filePath)
                nodes = [node] if node else []

            if nodes:
                logging.info(f"[AMORA] Loaded {len(nodes)} volume(s)")
                # Switch to Digital Rock module to show the loaded data
                self._openTool("AmoraDigitalRock")
            else:
                slicer.util.warningDisplay("No volumes found in file.")

        except Exception as e:
            logging.error(f"[AMORA] Load failed: {e}", exc_info=True)
            slicer.util.errorDisplay(f"Failed to load:\n{e}")
        finally:
            slicer.app.restoreOverrideCursor()

    def _askRawParams(self):
        """Show a dialog to get .raw file dimensions and dtype."""
        slicer.app.restoreOverrideCursor()  # Restore cursor so dialog is usable
        
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle("Raw File Parameters")
        dialog.setMinimumWidth(350)
        formLayout = qt.QFormLayout(dialog)

        dimX = qt.QSpinBox(); dimX.setRange(1, 10000); dimX.setValue(512)
        dimY = qt.QSpinBox(); dimY.setRange(1, 10000); dimY.setValue(512)
        dimZ = qt.QSpinBox(); dimZ.setRange(1, 10000); dimZ.setValue(512)

        dimRow = qt.QHBoxLayout()
        dimRow.addWidget(qt.QLabel("X:")); dimRow.addWidget(dimX)
        dimRow.addWidget(qt.QLabel("Y:")); dimRow.addWidget(dimY)
        dimRow.addWidget(qt.QLabel("Z:")); dimRow.addWidget(dimZ)
        formLayout.addRow("Dimensions:", dimRow)

        dtypeBox = qt.QComboBox()
        dtypeBox.addItems(["uint8", "uint16", "uint32", "int16", "float32", "float64"])
        dtypeBox.setCurrentText("uint16")
        formLayout.addRow("Data Type:", dtypeBox)

        # Explicit OK / Cancel buttons (QDialogButtonBox unreliable in this Qt)
        btnRow = qt.QHBoxLayout()
        okBtn = qt.QPushButton("OK")
        okBtn.setDefault(True)
        cancelBtn = qt.QPushButton("Cancel")
        okBtn.clicked.connect(lambda: dialog.accept())
        cancelBtn.clicked.connect(lambda: dialog.reject())
        btnRow.addWidget(okBtn)
        btnRow.addWidget(cancelBtn)
        formLayout.addRow("", btnRow)

        if dialog.exec_() == qt.QDialog.Accepted:
            kwargs = {
                "shape": (dimZ.value, dimY.value, dimX.value),
                "dtype": dtypeBox.currentText,
            }
            return True, kwargs
        return False, {}

    def _saveData(self):
        """Save active volume to file."""
        activeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        if not activeNode:
            slicer.util.warningDisplay("No volume loaded to save.")
            return

        savePath = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save Volume",
            activeNode.GetName() + ".nrrd",
            "NRRD (*.nrrd);;NumPy (*.npy);;All files (*)",
        )
        if not savePath:
            return

        try:
            if savePath.lower().endswith(".npy"):
                import numpy as np
                arr = slicer.util.arrayFromVolume(activeNode)
                np.save(savePath, arr)
            else:
                slicer.util.saveNode(activeNode, savePath)
            slicer.util.infoDisplay(f"Saved to:\n{savePath}")
        except Exception as e:
            slicer.util.errorDisplay(f"Save failed:\n{e}")

    # --- Theme Management ---

    def _cycleTheme(self):
        """Cycle through available themes."""
        HomeWidget._currentThemeIndex = (
            HomeWidget._currentThemeIndex + 1
        ) % len(THEME_NAMES)

        themeName = THEME_NAMES[HomeWidget._currentThemeIndex]

        # Reset Slicer's default first when cycling back to dark
        if HomeWidget._currentThemeIndex == 0:
            slicer.app.styleSheet = ""

        self._applyTheme(themeName)

    def _applyTheme(self, themeName):
        """Load and apply a QSS stylesheet globally."""
        try:
            stylesheetPath = self.resourcePath(f"Stylesheets/amora_{themeName}.qss")
            if not os.path.isfile(stylesheetPath):
                logging.warning(f"[AMORA] Stylesheet not found: {stylesheetPath}")
                return

            with open(stylesheetPath, "r", encoding="utf-8") as fh:
                style = fh.read()

            slicer.app.styleSheet = style
            HomeWidget._customStyleApplied = True
            logging.info(f"[AMORA] Theme: {THEME_LABELS.get(themeName, themeName)}")
        except Exception as e:
            logging.warning(f"[AMORA] Theme '{themeName}' failed: {e}")

    def applyApplicationStyle(self):
        """Apply the default AMORA style using SlicerCustomAppUtilities."""
        self._applyTheme(THEME_NAMES[HomeWidget._currentThemeIndex])
        self.styleThreeDWidget()
        self.styleSliceWidgets()

    # --- Styling Helpers ---

    def styleThreeDWidget(self):
        try:
            viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
        except Exception:
            pass

    def styleSliceWidgets(self):
        try:
            for name in slicer.app.layoutManager().sliceViewNames():
                sliceWidget = slicer.app.layoutManager().sliceWidget(name)
                self.styleSliceWidget(sliceWidget)
        except Exception:
            pass

    def styleSliceWidget(self, sliceWidget):
        controller = sliceWidget.sliceController()

    # --- Lifecycle ---

    def enter(self):
        """When user navigates to Home, hide the panel (clean 3D view)."""
        self._goHome()

    def exit(self):
        pass


# ===========================================================================
# LOGIC
# ===========================================================================

class HomeLogic(ScriptedLoadableModuleLogic):
    pass
