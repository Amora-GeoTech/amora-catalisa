#-----------------------------------------------------------------------------
set(MODULE_NAME Home)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/Gears.png
  Resources/Icons/AMORA_logo.png
  Resources/Icons/dr_ico.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/UI/Settings.ui
  Resources/Home.qss
  Resources/Stylesheets/amora_dark.qss
  Resources/Stylesheets/amora_ocean.qss
  Resources/Stylesheets/amora_alloyguard.qss
  Resources/Stylesheets/amora_light.qss
  )

slicerFunctionAddPythonQtResources(MODULE_PYTHON_QRC_RESOURCES
  Resources/${MODULE_NAME}.qrc
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS} ${MODULE_PYTHON_QRC_RESOURCES}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )
