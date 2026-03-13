#-----------------------------------------------------------------------------
set(MODULE_NAME AmoraDigitalRock)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/amora_io.py
  ${MODULE_NAME}Lib/amora_file_reader.py
  ${MODULE_NAME}Lib/AmoraNetCDFReader.py
  ${MODULE_NAME}Lib/AmoraNpyReader.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )
