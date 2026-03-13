"""
amora_io.py - Volume I/O for AMORA-Digital Rock
=================================================
Ported from GeoSlicer's ltrace.slicer.netcdf pipeline.
Handles .nc, .npy, and .raw file loading into Slicer MRML scene.

Supports:
  - Single .nc / .h5 / .hdf5 file loading  (import_file)
  - Directory of .nc files with auto-concat (import_directory)
  - .npy 3D arrays
  - .raw headerless volumes

No ltrace dependency - standalone.
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import vtk
import slicer

logger = logging.getLogger("[AMORA-IO]")

NETCDF_EXTENSIONS = {".nc", ".h5", ".hdf5"}


# ===========================================================================
# XARRAY / NETCDF UTILITIES  (cloned from GeoSlicer ltrace.slicer.netcdf)
# ===========================================================================

def get_dims(array) -> List[str]:
    """Return the first 3 dimension names from an xarray DataArray."""
    return array.dims[:3]


def get_spacing(array) -> List[float]:
    """Extract voxel spacing [x, y, z] from xarray coordinate diffs."""
    import xarray as xr
    z, y, x = get_dims(array)
    spacing = [
        array[dim][1] - array[dim][0] if len(array[dim]) > 1 else 1
        for dim in (x, y, z)
    ]
    spacing = [
        val.data.item() if isinstance(val, xr.DataArray) else val
        for val in spacing
    ]
    return spacing


def get_origin(array) -> List[float]:
    """Extract origin [-x0, -y0, z0] from xarray coordinates (RAS convention)."""
    import xarray as xr
    z, y, x = get_dims(array)
    origin = [-array[x][0], -array[y][0], array[z][0]]
    origin = [
        val.data.item() if isinstance(val, xr.DataArray) else val
        for val in origin
    ]
    return origin


def array_to_node(array, node, spacing=None):
    """
    Write xarray DataArray data into a Slicer vtkMRMLVolumeNode.
    Sets spacing, origin, and IJK-to-RAS transform.

    Cloned from GeoSlicer: ltrace.slicer.netcdf._array_to_node()
    """
    if spacing is None:
        spacing = get_spacing(array)

    ijk_to_ras = array.attrs.get("transform")
    origin = get_origin(array)

    slicer.util.updateVolumeFromArray(node, array.data)
    node.SetOrigin(*origin)
    node.SetSpacing(*spacing)

    if ijk_to_ras is not None:
        vtk_matrix = vtk.vtkMatrix4x4()
        vtk_matrix.DeepCopy(ijk_to_ras)
        node.SetIJKToRASMatrix(vtk_matrix)
    else:
        node.SetIJKToRASDirections(-1, 0, 0, 0, -1, 0, 0, 0, 1)


def _sanitize_name(name: str) -> str:
    """Clean variable name for Slicer node naming."""
    if not name or not (name[0].isalnum() or name[0] == "_"):
        name = "_" + name
    name = name.rstrip()
    name = name.replace("/", "\u2215")
    name = re.sub("_+", "_", name)
    result = "".join(c for c in name if ord(c) > 31 and ord(c) != 127)
    return result or "_"


# ===========================================================================
# DIRECTORY LOADING  (ported from GeoSlicer netcdf._open_dataset_from_directory)
# ===========================================================================

def _open_dataset_from_directory(path: Path) -> list:
    """
    Open all .nc files in a directory and concatenate variables.
    Each variable is concatenated along the z-axis (preferred) or the first dim.

    Ported from GeoSlicer: ltrace.slicer.netcdf._open_dataset_from_directory()
    """
    import xarray as xr

    files = sorted(list(path.glob("*.nc")))
    if not files:
        return []

    # Discover which variables are in which files
    var_to_files = defaultdict(list)
    all_vars = set()
    for f in files:
        with xr.open_dataset(f) as ds:
            for var_name in ds.data_vars:
                var_to_files[var_name].append(f)
                all_vars.add(var_name)

    final_datasets = []
    for var_name in sorted(list(all_vars)):
        files_with_var = var_to_files[var_name]

        if not files_with_var:
            continue

        if len(files_with_var) == 1:
            with xr.open_dataset(files_with_var[0]) as ds:
                final_datasets.append(ds[[var_name]])
            continue

        # Determine which dimension to concatenate ('z' is preferred)
        concat_dim = None
        with xr.open_dataset(files_with_var[0]) as ds:
            first_da = ds[var_name]
            if "z" in first_da.dims:
                concat_dim = "z"
            elif first_da.dims:
                concat_dim = first_da.dims[0]

        if concat_dim:
            try:
                def select_var_preprocess(ds, _var=var_name):
                    return ds[[_var]]

                combined_ds = xr.open_mfdataset(
                    files_with_var,
                    preprocess=select_var_preprocess,
                    concat_dim=concat_dim,
                    combine="nested",
                    chunks=256,
                )
                final_datasets.append(combined_ds)

            except Exception as e:
                logger.warning(
                    f"Could not concatenate variable '{var_name}': {e}. "
                    f"Loading files for this data array separately."
                )
                for f in files_with_var:
                    with xr.open_dataset(f) as ds:
                        final_datasets.append(ds[[var_name]])
        else:
            logger.warning(
                f"No concatenation dimension found for '{var_name}'. "
                f"Loading files separately."
            )
            for f in files_with_var:
                with xr.open_dataset(f) as ds:
                    final_datasets.append(ds[[var_name]])

    return final_datasets


# ===========================================================================
# DATASET IMPORT  (ported from GeoSlicer import_dataset + _handle_dataset)
# ===========================================================================

def import_dataset(dataset, images="all"):
    """
    Convert every 3D+ variable in an xarray Dataset into Slicer volume nodes.
    Yields (node, name) tuples.

    Ported from GeoSlicer: ltrace.slicer.netcdf.import_dataset()
    Simplified – no label maps, segmentations, or microtom support.
    """
    for name, array in dataset.items():
        if images != "all" and name not in images:
            continue
        if array.ndim < 3:
            logger.info(f"Skipping '{name}': ndim={array.ndim} < 3")
            continue

        clean_name = _sanitize_name(name)
        spacing = get_spacing(array)

        # Determine node class
        has_color = "c" in array.dims
        node_class = "vtkMRMLVectorVolumeNode" if has_color else "vtkMRMLScalarVolumeNode"

        node = slicer.mrmlScene.AddNewNodeByClass(node_class, clean_name)
        node.CreateDefaultDisplayNodes()
        array_to_node(array, node, spacing)

        logger.info(f"Loaded '{clean_name}': shape={array.shape}, dtype={array.dtype}")
        yield (node, clean_name)


def _handle_dataset(dataset, path):
    """
    Import a dataset and organize nodes in the subject hierarchy.
    After importing, sets up display (window/level, slice viewers).

    Ported from GeoSlicer: ltrace.slicer.netcdf._handle_dataset()
    """
    if isinstance(path, str):
        path = Path(path)

    dataset_name = path.with_suffix("").name if path.is_file() else path.name
    sh = slicer.mrmlScene.GetSubjectHierarchyNode()
    scene_id = sh.GetSceneItemID()

    existing = sh.GetItemByName(dataset_name)
    folder_id = existing if existing else sh.CreateFolderItem(scene_id, dataset_name)
    sh.SetItemAttribute(folder_id, "source_path", path.as_posix())

    nodes = []
    for node, name in import_dataset(dataset):
        sh.CreateItem(folder_id, node)
        nodes.append(node)

    # Set up display for the first scalar volume
    _setup_display(nodes)

    logger.info(f"Imported {len(nodes)} volumes from {dataset_name}")
    return nodes


def _subsample(arr, step=4):
    """Return a subsampled view of a 3D array for fast percentile estimation."""
    if arr.size < 1_000_000:
        return arr
    return arr[::step, ::step, ::step]


def _setup_display(nodes):
    """Configure slice viewers and window/level for loaded volumes."""
    first_scalar = None
    for n in nodes:
        if n is not None and n.IsA("vtkMRMLScalarVolumeNode"):
            first_scalar = n
            break

    if first_scalar:
        # Show in all slice viewers
        slicer.util.setSliceViewerLayers(background=first_scalar, fit=True)

        # Compute window/level from subsampled data percentiles
        # (AutoWindowLevelOn often fails for uint16 micro-CT data)
        display = first_scalar.GetDisplayNode()
        if display:
            arr = slicer.util.arrayFromVolume(first_scalar)
            sub = _subsample(arr)
            p2 = float(np.percentile(sub, 2))
            p98 = float(np.percentile(sub, 98))
            window = p98 - p2
            level = (p98 + p2) / 2.0
            if window < 1:
                window = float(sub.max() - sub.min())
                level = float((sub.max() + sub.min()) / 2.0)
            display.AutoWindowLevelOff()
            display.SetWindow(window)
            display.SetLevel(level)
            logger.info(f"Display: window={window:.1f}, level={level:.1f} (p2={p2}, p98={p98})")

        # Reset slice views to fit the data
        slicer.util.resetSliceViews()


def import_file(path: Union[str, Path]) -> list:
    """
    Load a single .nc / .h5 / .hdf5 file into Slicer's MRML scene.
    Returns list of created volume nodes.

    Ported from GeoSlicer: ltrace.slicer.netcdf.import_file()
    """
    import xarray as xr

    path = Path(path)
    dataset = xr.open_dataset(str(path))
    return _handle_dataset(dataset, path)


def import_directory(path: Union[str, Path]) -> list:
    """
    Load multiple .nc files from a directory, concatenating variables.
    Returns list of created volume nodes.

    Ported from GeoSlicer: ltrace.slicer.netcdf.import_directory()
    """
    path = Path(path)
    dataset_list = _open_dataset_from_directory(path)
    all_nodes = []
    for dataset in dataset_list:
        nodes = _handle_dataset(dataset, path)
        all_nodes.extend(nodes)
    return all_nodes


# Keep backward-compat alias
import_nc_file = import_file


# ===========================================================================
# NPY LOADER
# ===========================================================================

def read_npy_data(path: Union[str, Path]) -> dict:
    """
    Read a .npy file using memory-mapping (thread-safe, no Slicer calls).
    Returns dict with 'array', 'name', 'window', 'level' for scene creation.
    Percentiles are computed on a subsample of the mmap (~1/64th of voxels).
    """
    path = Path(path)
    mmap = np.load(str(path), mmap_mode='r')

    if mmap.ndim < 3:
        raise ValueError(f"Expected 3D array, got {mmap.ndim}D: shape={mmap.shape}")

    # Compute percentiles on subsampled mmap BEFORE materializing
    sub = _subsample(mmap)
    p2 = float(np.percentile(sub, 2))
    p98 = float(np.percentile(sub, 98))
    window = max(p98 - p2, 1.0)
    level = (p98 + p2) / 2.0

    # Now materialize into contiguous float32
    if mmap.dtype not in (np.float32, np.float64):
        tensor = np.ascontiguousarray(mmap, dtype=np.float32)
    else:
        tensor = np.ascontiguousarray(mmap)
    del mmap

    return {
        "array": tensor,
        "name": f"AMORA_{path.stem}",
        "window": window,
        "level": level,
    }


def import_npy_file(path: Union[str, Path]) -> slicer.vtkMRMLScalarVolumeNode:
    """
    Load a .npy volume file into Slicer.
    Assumes array is 3D (Z, Y, X).
    Uses memory-mapping to avoid loading the entire file into RAM upfront.
    """
    data = read_npy_data(path)
    return _create_scalar_node(data)


def read_raw_data(
    path: Union[str, Path],
    shape: Tuple[int, int, int],
    dtype: str = "uint8",
    byte_order: str = "little",
    header_offset: int = 0,
) -> dict:
    """
    Read a .raw file using memory-mapping (thread-safe, no Slicer calls).
    Returns dict with 'array', 'name', 'window', 'level'.
    """
    path = Path(path)
    np_dtype = np.dtype(dtype)
    if byte_order == "big":
        np_dtype = np_dtype.newbyteorder(">")

    expected = shape[0] * shape[1] * shape[2]
    mmap = np.memmap(str(path), dtype=np_dtype, mode='r', offset=header_offset,
                     shape=(expected,))
    mmap = mmap.reshape(shape)

    # Percentiles on subsample before materializing
    sub = _subsample(mmap)
    p2 = float(np.percentile(sub, 2))
    p98 = float(np.percentile(sub, 98))
    window = max(p98 - p2, 1.0)
    level = (p98 + p2) / 2.0

    if mmap.dtype not in (np.float32, np.float64):
        tensor = np.ascontiguousarray(mmap, dtype=np.float32)
    else:
        tensor = np.ascontiguousarray(mmap)
    del mmap

    return {
        "array": tensor,
        "name": f"AMORA_{path.stem}",
        "window": window,
        "level": level,
    }


def _create_scalar_node(data: dict) -> slicer.vtkMRMLScalarVolumeNode:
    """Create a Slicer volume node from read data dict. MUST run on main thread."""
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", data["name"])
    node.CreateDefaultDisplayNodes()

    slicer.util.updateVolumeFromArray(node, data["array"])
    node.SetIJKToRASDirections(-1, 0, 0, 0, -1, 0, 0, 0, 1)

    slicer.util.setSliceViewerLayers(background=node, fit=True)
    display = node.GetDisplayNode()
    if display:
        display.AutoWindowLevelOff()
        display.SetWindow(data["window"])
        display.SetLevel(data["level"])
    slicer.util.resetSliceViews()

    logger.info(f"Loaded '{data['name']}': shape={data['array'].shape}")
    return node


# ===========================================================================
# RAW LOADER
# ===========================================================================

def import_raw_file(
    path: Union[str, Path],
    shape: Tuple[int, int, int],
    dtype: str = "uint8",
    byte_order: str = "little",
    header_offset: int = 0,
) -> slicer.vtkMRMLScalarVolumeNode:
    """
    Load a headerless .raw volume file.
    Uses memory-mapping for efficient I/O.
    """
    data = read_raw_data(path, shape, dtype, byte_order, header_offset)
    return _create_scalar_node(data)


# ===========================================================================
# UNIFIED LOADER  (inspired by GeoSlicer microct.checkNetCdfFiles)
# ===========================================================================

def read_volume_data(path: Union[str, Path], **kwargs) -> list:
    """
    Thread-safe: read volume data from disk without touching Slicer scene.
    Returns list of dicts with 'array', 'name', 'window', 'level'.
    For .nc/.h5 files returns None (must use synchronous load_volume).
    """
    path = Path(path)

    if path.is_dir():
        return None  # netcdf directories need Slicer — use load_volume

    ext = path.suffix.lower()

    if ext == ".npy":
        return [read_npy_data(path)]
    elif ext == ".raw":
        if "shape" not in kwargs:
            raise ValueError(".raw files require 'shape' parameter: (Z, Y, X)")
        return [read_raw_data(path, **kwargs)]
    elif ext in NETCDF_EXTENSIONS:
        return None  # xarray + Slicer nodes — use load_volume
    else:
        return None  # fallback to synchronous


def load_volume(path: Union[str, Path], **kwargs) -> list:
    """
    Load any supported volume file or directory. Auto-detects format.
    Returns list of volume nodes.

    - .nc / .h5 / .hdf5 file  → import_file()
    - Directory with .nc files → import_directory()
    - .npy file               → import_npy_file()
    - .raw file               → import_raw_file() (requires shape kwarg)
    - Everything else         → Slicer's native loader as fallback
    """
    path = Path(path)

    # Directory: check for .nc files inside
    if path.is_dir():
        nc_files = list(path.glob("*.nc"))
        if nc_files:
            return import_directory(path)
        else:
            raise ValueError(f"No .nc files found in directory: {path}")

    ext = path.suffix.lower()

    if ext == ".npy":
        return [import_npy_file(path)]
    elif ext in NETCDF_EXTENSIONS:
        return import_file(path)
    elif ext == ".raw":
        if "shape" not in kwargs:
            raise ValueError(".raw files require 'shape' parameter: (Z, Y, X)")
        return [import_raw_file(path, **kwargs)]
    else:
        # Try Slicer's native loader as fallback
        node = slicer.util.loadVolume(str(path))
        if node:
            return [node]
        raise ValueError(f"Unsupported file format: {ext}")
