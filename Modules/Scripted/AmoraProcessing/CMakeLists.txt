#-----------------------------------------------------------------------------
set(MODULE_NAME AmoraProcessing)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Scripts/compute_histogram.py
  Resources/Scripts/segment_otsu.py
  Resources/Scripts/generate_rois.py
  Resources/Scripts/export_gif.py
  Resources/Scripts/plot_roi_vtk.py
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )
