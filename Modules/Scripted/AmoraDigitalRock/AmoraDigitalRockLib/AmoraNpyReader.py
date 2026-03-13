"""
AmoraNpyReader - Slicer ScriptedFileReader for NumPy .npy files
================================================================
Standalone file loaded via qSlicerScriptedFileReader.setPythonSource().
Routes .npy through amora_io instead of failing on Slicer's native reader.

IMPORTANT: Class name MUST match filename (without .py extension).
"""

import logging


class AmoraNpyReader:

    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return "AMORA NumPy Volume"

    def fileType(self):
        return "AmoraNpyVolume"

    def extensions(self):
        return ["NumPy array (*.npy)"]

    def canLoadFile(self, filePath):
        return filePath.lower().endswith(".npy")

    def load(self, properties):
        try:
            filePath = properties.get("fileName", "")
            if isinstance(filePath, list):
                filePath = filePath[0] if filePath else ""
            if not filePath:
                return False

            logging.info(f"[AMORA] NPY reader intercepted: {filePath}")

            from AmoraDigitalRockLib.amora_io import import_npy_file
            node = import_npy_file(filePath)

            if node:
                logging.info(f"[AMORA] Loaded NPY: {node.GetName()}")
                return True
            return False

        except Exception as e:
            logging.error(f"[AMORA] NPY load failed: {e}", exc_info=True)
            return False
