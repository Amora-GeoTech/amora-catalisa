# Build and Package AmoraDigitalRock

This document summarizes how to build and package AmoraDigitalRock on Linux.

AmoraDigitalRock is a custom Slicer application. Reading the [3D Slicer Developer Documentation](https://slicer.readthedocs.io/en/latest/developer_guide/index.html) may help answer additional questions.

The initial source files were created using [KitwareMedical/SlicerCustomAppTemplate](https://github.com/KitwareMedical/SlicerCustomAppTemplate).

## Clone

```sh
git clone https://github.com/LPSA-UFPA/AmoraDigitalRock.git
```

## Prerequisites

Install the prerequisites as described in the [Slicer documentation for building on Linux](https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/linux.html). This includes development tools, support libraries, and Qt.

## Build

Note: The build process can take hours.

Build:

```sh
cmake \
  -DCMAKE_BUILD_TYPE:STRING=Release \
  -DQt5_DIR:PATH=/opt/qt/5.15.2/gcc_64/lib/cmake/Qt5 \
  -S AmoraDigitalRock \
  -B AmoraDigitalRock-SuperBuild-Release
cd AmoraDigitalRock-SuperBuild-Release
make -j<N>
```

where `<N>` is the number of parallel builds. As a rule of thumb, many use the `number of CPU threads - 1` as the number of parallel builds.
On Ubuntu 20.04, the default Qt5 packages are too old and so the Slicer documentation linked above should have suggested a method of installing Qt 5.15.2;
if you installed it to `/opt/qt`, for example, then an extra option like `-DQt5_DIR:PATH=/opt/qt/5.15.2/gcc_64/lib/cmake/Qt5` would be needed in the `cmake` command above.

Once the application is built, there will be an _inner build_ inside the _superbuild_ folder, located at `AmoraDigitalRock-SuperBuild-Release/Slicer-build`. The application executable is contained in this _inner build_ folder.

## Package

From the _inner build_ folder:

```sh
make package
```
