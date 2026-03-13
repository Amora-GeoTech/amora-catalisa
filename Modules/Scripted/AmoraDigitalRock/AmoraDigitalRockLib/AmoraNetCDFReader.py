"""
AmoraNetCDFReader - Slicer ScriptedFileReader for NetCDF/HDF5 files
====================================================================
Standalone file loaded via qSlicerScriptedFileReader.setPythonSource().
Routes .nc/.h5/.hdf5 through amora_io instead of Slicer's ITK reader.

IMPORTANT: Class name MUST match filename (without .py extension).
"""

import logging
from pathlib import Path


class AmoraNetCDFReader:

    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return "AMORA NetCDF/HDF5 Volume"

    def fileType(self):
        return "AmoraNetCDFVolume"

    def extensions(self):
        return [
            "NetCDF (*.nc)",
            "HDF5 (*.h5)",
            "HDF5 (*.hdf5)",
        ]

    def canLoadFile(self, filePath):
        ext = filePath.lower()
        return ext.endswith(".nc") or ext.endswith(".h5") or ext.endswith(".hdf5")

    def load(self, properties):
        try:
            filePath = properties.get("fileName", "")
            if isinstance(filePath, list):
                filePath = filePath[0] if filePath else ""
            if not filePath:
                return False

            logging.info(f"[AMORA] NetCDF reader intercepted: {filePath}")

            path = Path(filePath)

            from AmoraDigitalRockLib.amora_io import import_file, import_directory

            if path.is_dir():
                nodes = import_directory(path)
            else:
                nodes = import_file(path)

            if nodes:
                logging.info(f"[AMORA] Loaded {len(nodes)} volume(s) from {filePath}")
                return True
            else:
                logging.warning(f"[AMORA] No volumes found in {filePath}")
                return False

        except Exception as e:
            logging.error(f"[AMORA] NetCDF load failed: {e}", exc_info=True)
            return False
