"""
amora_file_reader.py - Standalone Scripted File Reader for AMORA
================================================================
Routes .nc, .npy, .h5, .hdf5 files through amora_io.
Must be loaded via qSlicerScriptedFileReader.setPythonSource().
"""

import logging
from pathlib import Path


class AmoraFileReaderPlugin:

    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return "AMORA Volume"

    def fileType(self):
        return "AmoraVolume"

    def extensions(self):
        return [
            "NetCDF (*.nc)",
            "HDF5 (*.h5)",
            "HDF5 (*.hdf5)",
            "NumPy array (*.npy)",
        ]

    def canLoadFile(self, filePath):
        ext = filePath.lower()
        return (
            ext.endswith(".nc")
            or ext.endswith(".h5")
            or ext.endswith(".hdf5")
            or ext.endswith(".npy")
        )

    def load(self, properties):
        try:
            # Slicer scripted readers put the filename in properties
            filePath = properties.get("fileName", "")
            if isinstance(filePath, list):
                filePath = filePath[0] if filePath else ""
            if not filePath:
                return False

            logging.info(f"[AMORA] File reader intercepted: {filePath}")

            path = Path(filePath)

            import slicer
            try:
                # Need to use the full path to avoid import errors inside Slicer scripted env
                from AmoraDigitalRockLib.amora_io import import_directory, load_volume
            except ImportError:
                # Fallback if AmoraDigitalRockLib isn't cleanly in sys.path
                import sys
                import os
                mod_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if mod_path not in sys.path:
                    sys.path.append(mod_path)
                from AmoraDigitalRockLib.amora_io import import_directory, load_volume

            # Check if it's a directory (multi-file dataset)
            if path.is_dir():
                nodes = import_directory(path)
            else:
                nodes = load_volume(filePath)

            if nodes:
                logging.info(f"[AMORA] Loaded {len(nodes)} volume(s) from {filePath}")
                # Store loaded node IDs for Slicer to track (if caller checks this)
                self.parent.loadedNodes = [
                    node.GetID() for node in nodes if node is not None
                ]
                return True
            else:
                logging.warning(f"[AMORA] No volumes found in {filePath}")
                return False

        except Exception as e:
            logging.error(f"[AMORA] Reader failed: {e}", exc_info=True)
            return False
