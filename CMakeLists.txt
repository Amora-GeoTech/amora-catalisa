cmake_minimum_required(VERSION 3.16.3)

# Enable C++14
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set (EP_GIT_PROTOCOL "https")

# Slicer sources
include(FetchContent)
if(NOT DEFINED slicersources_SOURCE_DIR)
  # Download Slicer sources and set variables slicersources_SOURCE_DIR and slicersources_BINARY_DIR
  FetchContent_Populate(slicersources
    GIT_REPOSITORY ${EP_GIT_PROTOCOL}://github.com/Slicer/Slicer
    GIT_TAG        c792b1fb8445a8c91282b33bcaa037091130eb85
    GIT_PROGRESS   1
    )
else()
  set(slicersources_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/slicersources-subbuild)
endif()

# macOS initialization
set(CMAKE_MODULE_PATH ${slicersources_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})
include(SlicerInitializeOSXVariables)

project(AmoraDigitalRock)

# Configure Application
set(Slicer_APPLICATIONS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Applications)
set(Slicer_MAIN_PROJECT "AmoraDigitalRockApp")

# Set organization
set(Slicer_ORGANIZATION_DOMAIN "ufpa.br")
set(Slicer_ORGANIZATION_NAME   "LPSA/UFPA - Laboratorio de Petrossismica Sustentavel da Amazonia")

# Default home and favorite modules
set(Slicer_DEFAULT_HOME_MODULE "Home")
set(Slicer_DEFAULT_FAVORITE_MODULES "")

# Configure SuperBuild
set(SUPERBUILD_TOPLEVEL_PROJECT Slicer)
set(EXTERNAL_PROJECT_ADDITIONAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild")
include(ExternalProjectDependency)

# Additional Slicer dependencies looked up in EXTERNAL_PROJECT_ADDITIONAL_DIR
set(Slicer_ADDITIONAL_DEPENDENCIES
  )

#  Enable listed remote modules from ITK
set(Slicer_ITK_ADDITIONAL_MODULES
  )

if(NOT CMAKE_CONFIGURATION_TYPES)
  set(Slicer_DEFAULT_BUILD_TYPE "Release")
endif()
include(SlicerInitializeBuildType)
include(SlicerInitializeReleaseType)

# Set application bundle identifier for macOS
if(APPLE)
  set(Slicer_MACOSX_BUNDLE_GUI_IDENTIFIER "br.ufpa.amoradigitalrock")
endif()

# Installation folder and admin account requirement for Windows
if(WIN32)
  # Note: To avoid escaping issue, make sure to use forward slash when setting
  #       "Slicer_CPACK_NSIS_INSTALL_ROOT". It is replaced by "\\\\" in SlicerCPack.
  set(Slicer_CPACK_NSIS_INSTALL_REQUIRES_ADMIN_ACCOUNT ON)
  if(Slicer_CPACK_NSIS_INSTALL_REQUIRES_ADMIN_ACCOUNT)
    # User has administrative privileges, therefore we can install to shared folder
    # "C:\Program Files" or "C:\Program Files (x86)".
    if(CMAKE_CL_64)
      set(Slicer_CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
    else()
      set(Slicer_CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
    endif()
  else()
    # We do not require administrative privileges, therefore we install to user folder
    # "C:\Users\<username>\AppData\Local".
    set(Slicer_CPACK_NSIS_INSTALL_ROOT "$LOCALAPPDATA/${Slicer_ORGANIZATION_NAME}")
  endif()
endif()

# Slicer options
option(BUILD_TESTING                            "Build application test suite"                        OFF)
option(Slicer_BUILD_APPLICATIONUPDATE_SUPPORT   "Build application update support"                    OFF)
option(Slicer_BUILD_DOCUMENTATION               "Build documentation (Doxygen, sphinx, ...)"          OFF)
if(WIN32)
  option(Slicer_BUILD_WIN32_CONSOLE_LAUNCHER    "Build ${PROJECT_NAME} launcher executable as a console app on windows (displays console at application start)" OFF)
  option(Slicer_BUILD_WIN32_CONSOLE             "Build application executable as a console app (allows capturing and piping console output)" OFF)
endif()

option(Slicer_BUILD_DICOM_SUPPORT               "Build application with DICOM support"                OFF)
option(Slicer_BUILD_DIFFUSION_SUPPORT           "Build application with Diffusion support"            OFF)
option(Slicer_BUILD_EXTENSIONMANAGER_SUPPORT    "Build application with ExtensionManager support"     OFF)
option(Slicer_BUILD_MULTIVOLUME_SUPPORT         "Build application with MultiVolume support"          OFF)
option(Slicer_BUILD_PARAMETERSERIALIZER_SUPPORT "Build application with parameter serializer support" OFF)
option(Slicer_USE_PYTHONQT                      "Build application with Python support"               ON)
option(Slicer_USE_QtTesting                     "Build application with QtTesting support"            OFF)
option(Slicer_USE_SimpleITK                     "Build application with SimpleITK support"            OFF)

option(Slicer_BUILD_BRAINSTOOLS                 "Build application with BRAINSTools module"           OFF)
option(Slicer_BUILD_DataStore                   "Build application with DataStore module"             OFF)
option(Slicer_BUILD_CompareVolumes              "Build application with ChangeTrackerPy module"       OFF)
option(Slicer_BUILD_LandmarkRegistration        "Build application with LandmarkRegistration module"  ON)
option(Slicer_BUILD_SurfaceToolbox              "Build application with SurfaceToolbox module"        OFF)

# Enable Slicer built-in modules
set(Slicer_CLIMODULES_ENABLED
  ResampleDTIVolume             # Needed by ResampleScalarVectorDWIVolume
  ResampleScalarVectorDWIVolume # Depends on DiffusionApplications, needed by CropVolume
  )
set(Slicer_QTLOADABLEMODULES_ENABLED
  )
set(Slicer_QTSCRIPTEDMODULES_ENABLED
  )

# Disable Slicer built-in modules
set(Slicer_CLIMODULES_DISABLED
  )
set(Slicer_QTLOADABLEMODULES_DISABLED
  SceneViews
  SlicerWelcome
  ViewControllers
  )
set(Slicer_QTSCRIPTEDMODULES_DISABLED
  DataProbe
  DMRIInstall
  Endoscopy
  LabelStatistics
  PerformanceTests
  SampleData
  VectorToScalarVolume
  ScreenCapture
  SegmentEditor
  SegmentStatistics
  SurfaceToolbox
  )

# Enable/Disable Slicer custom modules: To create a new module, use the SlicerExtensionWizard.
set(Slicer_EXTENSION_SOURCE_DIRS
  ${AmoraDigitalRock_SOURCE_DIR}/Modules/Scripted/Home
  ${AmoraDigitalRock_SOURCE_DIR}/Modules/Scripted/AmoraDigitalRock
  ${AmoraDigitalRock_SOURCE_DIR}/Modules/Scripted/AmoraProcessing
  ${AmoraDigitalRock_SOURCE_DIR}/Modules/Scripted/AmoraFiltering
  ${AmoraDigitalRock_SOURCE_DIR}/Modules/Scripted/AmoraSimulation
  )

# Add remote extension source directories

# SlicerCustomAppUtilities
set(extension_name "SlicerCustomAppUtilities")
set(${extension_name}_SOURCE_DIR "${CMAKE_BINARY_DIR}/${extension_name}")
FetchContent_Populate(${extension_name}
  SOURCE_DIR     ${${extension_name}_SOURCE_DIR}
  GIT_REPOSITORY ${EP_GIT_PROTOCOL}://github.com/KitwareMedical/SlicerCustomAppUtilities.git
  GIT_TAG        1d984a2c9143e2617ff1ffa9d86c51e07dc6321e
  GIT_PROGRESS   1
  QUIET
  )
message(STATUS "Remote - ${extension_name} [OK]")
list(APPEND Slicer_EXTENSION_SOURCE_DIRS ${${extension_name}_SOURCE_DIR}/Modules/Scripted/SlicerCustomAppUtilities)

# SlicerOpenIGTLink
#set(extension_name "SlicerOpenIGTLink")
#set(${extension_name}_SOURCE_DIR "${CMAKE_BINARY_DIR}/${extension_name}")
#FetchContent_Populate(${extension_name}
#  SOURCE_DIR     ${${extension_name}_SOURCE_DIR}
#  GIT_REPOSITORY ${EP_GIT_PROTOCOL}://github.com/openigtlink/SlicerOpenIGTLink.git
#  GIT_TAG        2b92f7b1ffe02403109b671f28424e8770e902a0
#  GIT_PROGRESS   1
#  QUIET
#  )
#message(STATUS "Remote - ${extension_name} [OK]")
#list(APPEND Slicer_EXTENSION_SOURCE_DIRS ${${extension_name}_SOURCE_DIR})


# Override Slicer's medical volume rendering presets with rock-focused presets
set(_vr_custom "${AmoraDigitalRock_SOURCE_DIR}/Resources/VolumeRendering")
set(_vr_slicer "${slicersources_SOURCE_DIR}/Modules/Loadable/VolumeRendering/Resources")
if(EXISTS "${_vr_custom}/presets.xml")
  message(STATUS "AMORA: Overriding VolumeRendering presets with rock-focused presets")
  configure_file("${_vr_custom}/presets.xml" "${_vr_slicer}/presets.xml" COPYONLY)
  configure_file("${_vr_custom}/qSlicerVolumeRenderingModule.qrc" "${_vr_slicer}/qSlicerVolumeRenderingModule.qrc" COPYONLY)
  file(GLOB _rock_icons "${_vr_custom}/Icons/*.png")
  foreach(_icon ${_rock_icons})
    get_filename_component(_icon_name "${_icon}" NAME)
    configure_file("${_icon}" "${_vr_slicer}/Icons/${_icon_name}" COPYONLY)
  endforeach()
endif()

# Add Slicer sources
add_subdirectory(${slicersources_SOURCE_DIR} ${slicersources_BINARY_DIR})
