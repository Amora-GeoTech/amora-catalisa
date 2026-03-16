"""
amora_io.py - Volume I/O for AMORA-Digital Rock
=================================================
Full-featured NetCDF / HDF5 / NPY / RAW / CT projection I/O for Slicer.

Supports:
  - Single .nc / .h5 / .hdf5 file loading with label maps, segmentations,
    color tables, attribute preservation, and table import
  - Directory of .nc files with auto-concat
  - .npy 3D arrays
  - .raw headerless volumes
  - CT projection folders (.raw projections + geometry.yaml)
  - NetCDF export with compression, transforms, chunking, and label export
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import vtk
import slicer

logger = logging.getLogger("[AMORA-IO]")

NETCDF_EXTENSIONS = {".nc", ".h5", ".hdf5"}

MIN_CHUNKING_SIZE_BYTES = 2**33  # 8 GiB
CHUNK_SIZE_BYTES = 2**21  # 2 MiB

try:
    import yaml
except ImportError:
    yaml = None


# ===========================================================================
# DATACLASSES
# ===========================================================================

@dataclass
class DataArrayTransform:
    ijk_to_ras: np.ndarray
    transform: np.ndarray
    ras_min: np.ndarray
    ras_max: np.ndarray


# ===========================================================================
# NAME SANITIZATION
# ===========================================================================

def _sanitize_var_name(name: str) -> str:
    """Clean variable name for Slicer node naming."""
    if not name or not (name[0].isalnum() or name[0] == "_"):
        name = "_" + name
    name = name.rstrip()
    name = name.replace("/", "\u2215")
    # '__' is reserved for table columns (e.g. table_name__column_name)
    name = re.sub("_+", "_", name)
    result = "".join(c for c in name if ord(c) > 31 and ord(c) != 127)
    return result or "_"


def _deduplicate_names(names):
    """Ensure all names in list are unique by appending _N suffixes."""
    unique_names = []
    name_counts = defaultdict(int)
    for name in names:
        if name in unique_names:
            name_counts[name] += 1
            name = f"{name}_{name_counts[name]}"
        unique_names.append(name)
    return unique_names


def _sanitize_var_names(names):
    """Sanitize and deduplicate a list of variable names."""
    names = [_sanitize_var_name(name) for name in names]
    return _deduplicate_names(names)


# ===========================================================================
# XARRAY / NETCDF UTILITIES
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


def _array_to_node(array, node, spacing=None):
    """
    Write xarray DataArray data into a Slicer vtkMRMLVolumeNode.
    Sets spacing, origin, and IJK-to-RAS transform.
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


def _crop_value(array, value):
    """Crop an xarray DataArray by removing border regions equal to value."""
    crop_where = array == value
    slice_dict = {}
    for dim in array.dims:
        if dim == "c":
            continue
        crop_over_dim = crop_where.all(dim=tuple(set(array.dims) - {dim}))
        first = crop_over_dim.argmin()
        last = array[dim].size - crop_over_dim[::-1].argmin()
        slice_dict[dim] = slice(int(first), int(last))
    return array[slice_dict]


# ===========================================================================
# COLOR TABLE / LABEL MAP UTILITIES
# ===========================================================================

def create_color_table(name, colors, color_names=None, add_background=True):
    """Create a vtkMRMLColorTableNode with the given colors."""
    color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
    n = len(colors) + (1 if add_background else 0)
    color_node.SetNumberOfColors(n)
    color_node.SetNamesInitialised(True)

    idx = 0
    if add_background:
        color_node.SetColor(0, "Background", 0.0, 0.0, 0.0, 0.0)
        idx = 1

    for i, color in enumerate(colors):
        r, g, b = color[:3]
        a = color[3] if len(color) > 3 else 1.0
        cname = color_names[i] if color_names and i < len(color_names) else f"Label_{i + idx}"
        color_node.SetColor(i + idx, cname, r, g, b, a)

    return color_node


def _parse_hex_color(color_str):
    """Parse a hex color string (#RRGGBB or #RGB) to (r, g, b) tuple in [0, 1]."""
    color_str = color_str.strip()
    if color_str.startswith("#"):
        color_str = color_str[1:]
    if len(color_str) == 3:
        color_str = "".join(c * 2 for c in color_str)
    if len(color_str) == 6:
        r = int(color_str[0:2], 16) / 255.0
        g = int(color_str[2:4], 16) / 255.0
        b = int(color_str[4:6], 16) / 255.0
        return (r, g, b)
    return (0.5, 0.5, 0.5)


def nc_labels_to_color_node(labels, name="nc_labels"):
    """Create a color table node from NetCDF label attributes."""
    colors = []
    colorNames = []
    if isinstance(labels, str):
        labels = [labels]
    for label in labels[1:]:
        try:
            seg_name, index, color = label.split(",")
        except Exception as error:
            logger.info(f"Failed to parse label string: {label}. Error: {error}")
            continue
        colorNames.append(seg_name)
        colors.append(_parse_hex_color(color))

    color_table = create_color_table(
        f"{name}_ColorTable", colors=colors, color_names=colorNames, add_background=True
    )
    return color_table


def _get_label_map_labels_csv(label_map_node, with_color=True):
    """Get label map labels as CSV strings (Name,Index,Color)."""
    display = label_map_node.GetDisplayNode()
    if not display:
        return []
    color_node = display.GetColorNode()
    if not color_node:
        return []

    labels = []
    for i in range(color_node.GetNumberOfColors()):
        name = color_node.GetColorName(i)
        if name == "Background" and i == 0:
            continue
        color = [0, 0, 0, 0]
        color_node.GetColor(i, color)
        if with_color:
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            labels.append(f"{name},{i},{hex_color}")
        else:
            labels.append(f"{name},{i}")
    return labels


# ===========================================================================
# TEMPORARY NODE HELPERS
# ===========================================================================

_temporary_nodes = []


def _create_temporary_volume_node(node_class, name, unique_name=False):
    """Create a temporary volume node that can be cleaned up later."""
    if unique_name:
        node = slicer.mrmlScene.AddNewNodeByClass(node_class.__name__)
    else:
        node = slicer.mrmlScene.AddNewNodeByClass(node_class.__name__, name)
    node.CreateDefaultDisplayNodes()
    node.SetAttribute("_temporary", "1")
    _temporary_nodes.append(node)
    return node


def _make_temporary_node_permanent(node, show=True):
    """Remove temporary flag and optionally show in viewers."""
    node.RemoveAttribute("_temporary")
    if node in _temporary_nodes:
        _temporary_nodes.remove(node)
    if show and node.IsA("vtkMRMLScalarVolumeNode"):
        slicer.util.setSliceViewerLayers(background=node)


def _remove_temporary_nodes():
    """Remove all temporary nodes from the scene."""
    global _temporary_nodes
    for node in _temporary_nodes:
        if node and slicer.mrmlScene.IsNodePresent(node):
            slicer.mrmlScene.RemoveNode(node)
    _temporary_nodes = []


def _update_segmentation_from_label_map(segmentation_node, label_map_node, include_empty=True):
    """Convert a label map to a segmentation node."""
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        label_map_node, segmentation_node
    )


def _get_source_volume(node):
    """Get the source/reference volume for a segmentation or label map node."""
    if hasattr(node, "GetNodeReference"):
        role = slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()
        ref = node.GetNodeReference(role)
        if ref:
            return ref
    return None


def _safe_convert_array(array, dtype):
    """Safely convert array to target dtype, clipping if needed."""
    target = np.dtype(dtype)
    if np.issubdtype(target, np.integer):
        info = np.iinfo(target)
        array = np.clip(array, info.min, info.max)
    return array.astype(target)


# ===========================================================================
# ATTRIBUTE STORAGE (TOML-based)
# ===========================================================================

def _convert_numpy(obj):
    """Recursively convert numpy types to native Python for serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating, np.ndarray)):
        return obj.tolist()
    else:
        return obj


def _create_attr_text_node(name, key, value):
    """Create a text node storing a single attribute key-value pair."""
    text_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", name)
    text_node.SetText(value)
    text_node.SetAttribute("IsNcAttrs", "1")
    text_node.SetAttribute("AttrKey", key)
    return text_node


def _create_attrs_toml_node(name, attrs):
    """Create a text node storing attributes as TOML."""
    try:
        import toml
        text_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", name)
        text_node.SetText(toml.dumps(_convert_numpy(attrs)))
        text_node.SetAttribute("IsNcAttrs", "1")
        return text_node
    except ImportError:
        # Fallback: store as repr string
        import json
        text_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", name)
        text_node.SetText(json.dumps(_convert_numpy(attrs), indent=2))
        text_node.SetAttribute("IsNcAttrs", "1")
        return text_node


def _is_attr_node(node):
    """Check if a node is a NetCDF attribute text node."""
    if not isinstance(node, slicer.vtkMRMLTextNode):
        return False
    return node.GetAttribute("IsNcAttrs") == "1"


def _attrs_from_node(node):
    """Extract attributes dict from a text node."""
    if not _is_attr_node(node):
        raise ValueError(f"Node {node.GetName()} is not a valid attributes node.")
    key = node.GetAttribute("AttrKey")
    text = node.GetText()
    if key:
        return {key: text}
    else:
        try:
            import toml
            return toml.loads(text)
        except ImportError:
            import json
            try:
                return json.loads(text)
            except Exception:
                return {"_raw": text}


def _create_text_nodes_for_attrs(attrs_dict, base_name):
    """Create text nodes for all attributes, grouping small ones together."""
    text_nodes = {}
    if not attrs_dict:
        return text_nodes

    small_attrs = {}
    for key, value in attrs_dict.items():
        try:
            import toml
            attr_toml = toml.dumps({key: value})
            is_large = len(attr_toml) > 500 or key.lower() == "pcr"
        except ImportError:
            import json
            attr_json = json.dumps({key: _convert_numpy(value)})
            is_large = len(attr_json) > 500 or key.lower() == "pcr"

        if is_large:
            node_name = f"{base_name}_attr_{key}"
            if isinstance(value, str):
                text_node = _create_attr_text_node(node_name, key, value)
            else:
                text_node = _create_attrs_toml_node(node_name, {key: value})
            text_nodes[key] = text_node
        else:
            small_attrs[key] = value

    if small_attrs:
        text_node = _create_attrs_toml_node(f"{base_name}_attrs", small_attrs)
        text_nodes[...] = text_node

    return text_nodes


def _get_attrs_from_text_nodes(folder_id, sh):
    """Retrieve attributes stored in text nodes under a subject hierarchy folder."""
    attrs = {}
    children = vtk.vtkIdList()
    sh.GetItemChildren(folder_id, children)

    for i in range(children.GetNumberOfIds()):
        child_id = children.GetId(i)
        if sh.GetItemOwnerPluginName(child_id) == "Folder":
            item_name = sh.GetItemName(child_id)
            if not item_name.endswith("_attrs"):
                continue
            sub_children = vtk.vtkIdList()
            sh.GetItemChildren(child_id, sub_children)
            for j in range(sub_children.GetNumberOfIds()):
                sub_child_id = sub_children.GetId(j)
                sub_child_node = sh.GetItemDataNode(sub_child_id)
                if _is_attr_node(sub_child_node):
                    attrs.update(_attrs_from_node(sub_child_node))
    return attrs


# ===========================================================================
# HDF5 UTILITIES
# ===========================================================================

def extract_pixel_sizes_from_hdf5(file_path):
    """Extract pixel sizes from HDF5 beamline metadata (returns x_mm, y_mm, z_mm)."""
    try:
        import h5py
    except ImportError:
        logger.warning("h5py not available; cannot extract pixel sizes from HDF5")
        return None

    base_path = "Beamline Parameters/snapshot/after/beamline-state/beam-optics/measured"

    try:
        with h5py.File(file_path, "r") as f:
            def get_param_as_mm(param_name):
                param_path = f"{base_path}/{param_name}"
                try:
                    value = f[f"{param_path}/value"][()]
                except KeyError:
                    return 1
                try:
                    units = f[f"{param_path}/units"][()]
                    units = units.decode("utf-8") if isinstance(units, bytes) else units
                except KeyError:
                    units = "mm"

                # Simple unit conversion without pint dependency
                unit_factors = {
                    "mm": 1.0, "m": 1000.0, "cm": 10.0, "um": 0.001,
                    "µm": 0.001, "nm": 1e-6, "in": 25.4,
                }
                factor = unit_factors.get(units, 1.0)
                return value * factor

            x_mm = get_param_as_mm("pixel-size-x")
            y_mm = get_param_as_mm("pixel-size-y")
            z_mm = x_mm
            return x_mm, y_mm, z_mm
    except Exception as e:
        logger.warning(f"Failed to extract pixel sizes from HDF5: {e}")
        return None


# ===========================================================================
# TRANSFORM UTILITIES
# ===========================================================================

def _vtk_4x4_to_numpy(matrix_vtk):
    """Convert a vtkMatrix4x4 to a 4x4 numpy array."""
    matrix_np = np.empty(16)
    matrix_vtk.DeepCopy(matrix_np, matrix_vtk)
    return matrix_np.reshape(4, 4)


def _add_color_dim(transform):
    """Add a color dimension to a 4x4 transform, making it 5x5."""
    assert transform.shape == (4, 4)
    ret = np.insert(transform, 3, 0, axis=0)
    ret = np.insert(ret, 3, 0, axis=1)
    ret[3, 3] = 1
    return ret


def _get_transform(node, array_shape):
    """
    Compute the full DataArrayTransform for a volume node.
    Includes IJK-to-RAS matrix, internal transform, and RAS bounding box.
    """
    default_transform = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    transform = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(transform)
    transform = _vtk_4x4_to_numpy(transform)
    ijk_to_ras = transform.copy()

    transform = transform @ default_transform
    transform[:2, 3] *= -1

    # Convert XYZ to ZYX
    transform[:3, :3] = np.flip(transform[:3, :3], axis=(0, 1))
    transform[:3, 3] = np.flip(transform[:3, 3], axis=0)

    pre_shape = np.array(array_shape[:3]) - 1

    # Corners of a unit cube
    unit_corners = [
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ]
    ijk_corners = unit_corners * pre_shape
    ras_corners = np.array([(transform @ np.concatenate([ijk, [1]]))[:3] for ijk in ijk_corners])

    ras_min, ras_max = ras_corners.min(axis=0), ras_corners.max(axis=0)

    if len(array_shape) == 4:
        transform = _add_color_dim(transform)

    return DataArrayTransform(ijk_to_ras, transform, ras_min, ras_max)


def _get_dataset_main_dims(dataset):
    """Use a heuristic to find likely zyxc dimensions of a dataset."""
    for var in dataset:
        array = dataset[var]
        dims = array.dims
        if len(dims) >= 3:
            return dims[:4]


# ===========================================================================
# CHUNKING
# ===========================================================================

def _recommended_chunksizes(img):
    """Compute recommended chunk sizes for NetCDF compression."""
    chunk_size = round((CHUNK_SIZE_BYTES // img.dtype.itemsize) ** (1 / 3))
    if (
        img.nbytes >= MIN_CHUNKING_SIZE_BYTES
        and img.ndim >= 3
        and all(size >= chunk_size * 4 for size in img.shape[:3])
    ):
        return (chunk_size,) * 3
    return None


# ===========================================================================
# TABLE / DATAFRAME UTILITIES
# ===========================================================================

def _dataframe_to_table_node(df):
    """Convert a pandas DataFrame to a Slicer vtkMRMLTableNode."""
    import pandas as pd
    table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
    table = table_node.GetTable()

    for col_name in df.columns:
        col_data = df[col_name]
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                arr = vtk.vtkIntArray()
            else:
                arr = vtk.vtkDoubleArray()
        else:
            arr = vtk.vtkStringArray()

        arr.SetName(str(col_name))
        arr.SetNumberOfTuples(len(col_data))

        for i, val in enumerate(col_data):
            if isinstance(arr, vtk.vtkStringArray):
                arr.SetValue(i, str(val))
            elif isinstance(arr, vtk.vtkIntArray):
                arr.SetValue(i, int(val) if not pd.isna(val) else 0)
            else:
                arr.SetValue(i, float(val) if not pd.isna(val) else 0.0)

        table.AddColumn(arr)

    table_node.Modified()
    return table_node


def _table_node_to_dataframe(table_node):
    """Convert a Slicer vtkMRMLTableNode to a pandas DataFrame."""
    import pandas as pd
    table = table_node.GetTable()
    data = {}
    for i in range(table.GetNumberOfColumns()):
        col = table.GetColumn(i)
        col_name = col.GetName()
        values = []
        for j in range(col.GetNumberOfTuples()):
            if isinstance(col, vtk.vtkStringArray):
                values.append(col.GetValue(j))
            else:
                values.append(col.GetValue(j))
        data[col_name] = values
    return pd.DataFrame(data)


def _auto_detect_column_type(table_node):
    """Auto-detect and set column types for a table node."""
    pass  # Slicer handles column types automatically


# ===========================================================================
# DIRECTORY LOADING
# ===========================================================================

def _open_dataset_from_directory(path: Path) -> list:
    """
    Open all .nc files in a directory and concatenate variables.
    Each variable is concatenated along the z-axis (preferred) or the first dim.
    """
    import xarray as xr

    if isinstance(path, str):
        path = Path(path)

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
# DATASET IMPORT (full-featured)
# ===========================================================================

def import_dataset(dataset, images="all"):
    """
    Convert every 3D+ variable in an xarray Dataset into Slicer volume nodes.
    Supports scalar volumes, vector volumes, label maps, segmentations,
    color tables, reference nodes, and table data.
    Yields (main_node, aux_text_nodes) tuples.
    """
    import pandas as pd

    has_reference = []
    other = []
    column_items = []

    for name, array in dataset.items():
        if array.dims[0].startswith("table__"):
            column_items.append((name, array))
            continue
        add_to = has_reference if "reference" in array.attrs else other
        add_to.append((name, array))

    # Import nodes with references last so the referenced nodes are already loaded
    array_items = other + has_reference

    role = slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()
    imported = {}
    first_scalar = None
    first_label_map = None

    for name, array in array_items:
        if images != "all" and name not in images:
            continue

        if array.ndim < 3:
            logger.info(f"Skipping '{name}': ndim={array.ndim} < 3")
            continue

        clean_name = _sanitize_var_name(name)
        has_labels = "labels" in array.attrs
        is_labelmap = False if "type" not in array.attrs else array.attrs["type"] == "labelmap"
        ijk_to_ras = array.attrs.get("transform")
        single_coords = ijk_to_ras is None

        # Get spacing before cropping in case some dimension is size 1
        spacing = get_spacing(array)
        if single_coords:
            fill_value = 0 if has_labels else 255
            array = _crop_value(array, fill_value)

        if has_labels:
            # Create label map node
            label_map = _create_temporary_volume_node(
                slicer.vtkMRMLLabelMapVolumeNode, clean_name, unique_name=False
            )
            _array_to_node(array, label_map, spacing)

            color_table = nc_labels_to_color_node(array.attrs["labels"], clean_name)
            label_map.GetDisplayNode().SetAndObserveColorNodeID(color_table.GetID())

            if is_labelmap:
                _make_temporary_node_permanent(label_map, show=True)
                node = label_map
                first_label_map = first_label_map or node
            else:
                # Convert to segmentation node
                node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode", clean_name
                )
                _update_segmentation_from_label_map(node, label_map, include_empty=True)

            # Set reference if specified
            if "reference" in array.attrs:
                master_name = array.attrs["reference"]
                try:
                    master = imported[master_name]
                    node.SetNodeReferenceID(role, master.GetID())
                    if not is_labelmap and hasattr(node, "SetReferenceImageGeometryParameterFromVolumeNode"):
                        node.SetReferenceImageGeometryParameterFromVolumeNode(master)
                except KeyError:
                    pass

            _remove_temporary_nodes()
        else:
            # Regular scalar or vector volume
            node_class = "vtkMRMLVectorVolumeNode" if "c" in array.dims else "vtkMRMLScalarVolumeNode"
            node = slicer.mrmlScene.AddNewNodeByClass(node_class, clean_name)
            node.CreateDefaultDisplayNodes()
            _array_to_node(array, node, spacing)
            first_scalar = first_scalar or node

        # Store non-special attributes as text nodes
        special_attrs = {"labels", "type", "reference", "transform"}
        attrs = {k: v for k, v in array.attrs.items() if k not in special_attrs}
        text_nodes = _create_text_nodes_for_attrs(attrs, clean_name) if attrs else {}

        imported[name] = node
        logger.info(f"Loaded '{clean_name}': shape={array.shape}, dtype={array.dtype}")
        yield (node, list(text_nodes.values()))

    if first_scalar:
        slicer.util.setSliceViewerLayers(
            background=first_scalar, label=first_label_map, fit=True
        )
        display = first_scalar.GetDisplayNode()
        if display:
            display.AutoWindowLevelOn()

    # Handle table data stored in dataset
    tables = defaultdict(list)
    for name, column in column_items:
        table_name, column_name = name.split("__")
        tables[table_name].append((column_name, column))

    for table_name, columns in tables.items():
        df = pd.DataFrame({col_name: col.data for col_name, col in columns})
        node = _dataframe_to_table_node(df)
        node.SetName(table_name)
        yield (node, [])


def _handle_dataset(dataset, path, callback=None, images="all"):
    """
    Import a dataset and organize nodes in the subject hierarchy.
    Handles HDF5 pixel sizes, attributes, progress callbacks, and PCR metadata.
    """
    if isinstance(path, str):
        path = Path(path)
    if callback is None:
        callback = lambda *args, **kwargs: None

    pixel_sizes = None
    if path.suffix in (".h5", ".hdf5"):
        pixel_sizes = extract_pixel_sizes_from_hdf5(path)

    dataset_name = path.with_suffix("").name if path.is_file() else path.name
    sh = slicer.mrmlScene.GetSubjectHierarchyNode()
    scene_id = sh.GetSceneItemID()

    existing = sh.GetItemByName(dataset_name)
    current_dir = existing if existing else sh.CreateFolderItem(scene_id, dataset_name)
    sh.SetItemAttribute(current_dir, "netcdf_path", path.as_posix())
    sh.SetItemAttribute(current_dir, "source_path", path.as_posix())

    nodes = []
    n_items = len(list(dataset.items()))
    progress_step = 90 / max(n_items, 1)

    for idx, (main_node, aux_nodes) in enumerate(import_dataset(dataset, images=images)):
        if pixel_sizes and hasattr(main_node, "SetSpacing"):
            main_node.SetSpacing(*pixel_sizes)
        callback("Loading...", 10 + idx * progress_step)

        if aux_nodes:
            image_folder = sh.CreateFolderItem(current_dir, main_node.GetName())
            sh.CreateItem(image_folder, main_node)
            attrs_folder = sh.CreateFolderItem(image_folder, f"{main_node.GetName()}_attrs")
            for aux_node in aux_nodes:
                sh.CreateItem(attrs_folder, aux_node)
        else:
            sh.CreateItem(current_dir, main_node)

        nodes.append(main_node)

    # Store dataset-level attributes
    special_dataset_attrs = {"geoslicer_version", "amora_version"}
    dataset_attrs_to_encode = {
        k: v for k, v in dataset.attrs.items() if k not in special_dataset_attrs
    }
    if dataset_attrs_to_encode:
        text_nodes = _create_text_nodes_for_attrs(dataset_attrs_to_encode, dataset_name)

        pcr_node = text_nodes.get("pcr") or text_nodes.get("PCR") or text_nodes.get("Pcr")
        if pcr_node is not None:
            for node in nodes:
                node.SetAttribute("PCR", pcr_node.GetID())

        attrs_folder = sh.CreateFolderItem(current_dir, f"{dataset_name}_attrs")
        for _, text_node in text_nodes.items():
            sh.CreateItem(attrs_folder, text_node)
            nodes.append(text_node)

    # Set up display for first scalar volume (percentile-based W/L)
    _setup_display(nodes)

    logger.info(f"Imported {len(nodes)} nodes from {dataset_name}")
    return nodes


# ===========================================================================
# NETCDF EXPORT (full-featured)
# ===========================================================================

def _segmentation_to_label_map(segmentation):
    """Convert a segmentation node to a temporary label map node."""
    label_map = _create_temporary_volume_node(
        slicer.vtkMRMLLabelMapVolumeNode, segmentation.GetName()
    )
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        segmentation, label_map, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
    )
    return label_map


def _node_to_data_array(node, dim_names, dtype=None):
    """
    Convert a Slicer volume or segmentation node to an xarray DataArray.
    Returns (node, DataArray) or an error string.
    """
    import xarray as xr

    attrs = {}
    if isinstance(node, slicer.vtkMRMLSegmentationNode):
        node_name = node.GetName()
        node = _segmentation_to_label_map(node)
        if node.GetImageData().GetPointData().GetScalars() is None:
            return f"Could not export {node_name}: segmentation is empty"
        attrs["type"] = "segmentation"
    elif isinstance(node, slicer.vtkMRMLLabelMapVolumeNode):
        attrs["type"] = "labelmap"
    elif not isinstance(node, slicer.vtkMRMLScalarVolumeNode):
        raise ValueError(f"Unsupported node type: {type(node)}")

    if isinstance(node, slicer.vtkMRMLLabelMapVolumeNode):
        attrs["labels"] = ["Name,Index,Color"] + _get_label_map_labels_csv(node, with_color=True)

    array = slicer.util.arrayFromVolume(node)
    if dtype:
        array = _safe_convert_array(array, dtype)
    dims = dim_names[:array.ndim]
    return node, xr.DataArray(array, dims=dims, attrs=attrs)


def export_netcdf(
    export_path,
    data_nodes,
    reference_item=None,
    single_coords=False,
    use_compression=False,
    callback=None,
    node_names=None,
    node_dtypes=None,
    save_in_place=False,
):
    """
    Export Slicer volume nodes to a NetCDF file.

    Supports scalar volumes, label maps, segmentations, vector volumes,
    table nodes, transforms, compression, and chunked storage.

    Parameters
    ----------
    export_path : str or Path
        Output .nc file path.
    data_nodes : list
        List of Slicer nodes to export.
    reference_item : node, optional
        Reference volume for single_coords mode.
    single_coords : bool
        If True, all volumes share the same coordinate system.
    use_compression : bool
        Enable zlib compression.
    callback : callable, optional
        Progress callback(message, percent).
    node_names : list, optional
        Custom names for each node.
    node_dtypes : list, optional
        Target dtypes for each node.
    save_in_place : bool
        If True, update existing NetCDF file.

    Returns
    -------
    list of str : warning messages
    """
    import xarray as xr
    from scipy import ndimage

    if not data_nodes:
        raise ValueError("No images selected.")

    if callback is None:
        callback = lambda *args, **kwargs: None

    if single_coords and not save_in_place:
        if not reference_item:
            raise ValueError("No reference node selected.")
        if reference_item not in data_nodes:
            raise ValueError("Reference image must be in the list of images to export.")

    warnings = []
    sh = slicer.mrmlScene.GetSubjectHierarchyNode()

    callback("Starting...", 0)

    arrays = {}
    table_arrays = {}
    coords = {}
    node_names = node_names or [node.GetName() for node in data_nodes]
    node_names = _sanitize_var_names(node_names)
    node_dtypes = node_dtypes or [None] * len(data_nodes)

    id_to_name_map = {node.GetID(): name for node, name in zip(data_nodes, node_names)}
    node_attrs = {}
    dataset_attrs = {}
    processed_dataset_folders = set()

    for node, name in zip(data_nodes, node_names):
        image_folder_id = sh.GetItemParent(sh.GetItemByDataNode(node))
        if not image_folder_id:
            continue

        attrs = _get_attrs_from_text_nodes(image_folder_id, sh)
        if attrs:
            node_attrs[name] = attrs

        dataset_folder_id = sh.GetItemParent(image_folder_id)
        if dataset_folder_id and dataset_folder_id not in processed_dataset_folders:
            if sh.GetItemAttribute(dataset_folder_id, "netcdf_path"):
                dataset_folder_attrs = _get_attrs_from_text_nodes(dataset_folder_id, sh)
                dataset_attrs.update(dataset_folder_attrs)
                processed_dataset_folders.add(dataset_folder_id)

    if save_in_place:
        existing_dataset = xr.load_dataset(export_path)
        existing_dims = _get_dataset_main_dims(existing_dataset)

    for node, dtype in zip(data_nodes, node_dtypes):
        name = id_to_name_map[node.GetID()]

        if isinstance(node, slicer.vtkMRMLTableNode):
            dim_name = f"table__{name}"
            if save_in_place and dim_name in existing_dataset.dims:
                continue

            df = _table_node_to_dataframe(node)
            names_list = list(df.columns)
            df.columns = _sanitize_var_names(names_list)

            ds = df.to_xarray()
            ds = ds.rename({"index": dim_name})
            for col_name, data_array in ds.items():
                table_arrays[f"{name}__{col_name}"] = data_array
            continue

        if save_in_place and name in existing_dataset:
            continue
        is_ref = node == reference_item

        if save_in_place:
            dims = existing_dims
        elif single_coords:
            dims = list("zyxc")
        else:
            dims = [f"{d}_{name}" for d in "zyx"] + ["c"]

        source_node = _get_source_volume(node)

        result = _node_to_data_array(node, dims, dtype)
        if isinstance(result, str):
            warnings.append(result)
            continue
        node, data_array = result

        if source_node:
            try:
                source_name = id_to_name_map[source_node.GetID()]
                data_array.attrs["reference"] = source_name
            except KeyError:
                pass

        if is_ref:
            reference_item = node
        if data_array.ndim == 4:
            coords["c"] = ["r", "g", "b"]
        arrays[name] = (data_array, _get_transform(node, data_array.shape), node)

    if single_coords:
        if save_in_place:
            ref_spacing = []
            ras_min = []
            output_shape = []
            for dim in existing_dims[:3]:
                dim_coords = existing_dataset.coords[dim]
                if len(dim_coords) > 1:
                    ref_spacing.append((dim_coords[1] - dim_coords[0]).item())
                else:
                    ref_spacing.append(1)
                ras_min.append(dim_coords[0].item())
                output_shape.append(len(dim_coords))
        else:
            ref_spacing = np.array(reference_item.GetSpacing())[::-1]
            ras_min = np.array([tr.ras_min for _, tr, _ in arrays.values()]).min(axis=0)
            ras_max = np.array([tr.ras_max for _, tr, _ in arrays.values()]).max(axis=0)
            output_shape = np.ceil((ras_max - ras_min) / ref_spacing).astype(int) + 1

        output_transform_no_color = np.array([
            [ref_spacing[0], 0, 0, ras_min[0]],
            [0, ref_spacing[1], 0, ras_min[1]],
            [0, 0, ref_spacing[2], ras_min[2]],
            [0, 0, 0, 1],
        ])
        output_transform_with_color = _add_color_dim(output_transform_no_color)

    if not arrays and not table_arrays:
        raise ValueError("No images to export.\n" + "\n".join(warnings))

    progress_range = np.arange(5, 90, 85 / len(arrays)) if arrays else []
    data_arrays = {}
    for (name, (data_array, transform, _)), progress in zip(arrays.items(), progress_range):
        callback(f'Processing "{name}"...', round(progress))

        array = data_array.data
        attrs = dict(data_array.attrs)

        if single_coords:
            input_transform = transform.transform
            output_transform = (
                output_transform_no_color if data_array.ndim == 3
                else output_transform_with_color
            )
            output_to_input = np.linalg.inv(input_transform) @ output_transform

            fill_value = 0 if "labels" in data_array.attrs else 255
            shape = output_shape.copy()
            if data_array.ndim == 4:
                shape = np.append(shape, 3)

            # Transform interpolation does not work on dimensions with size 1
            for i in range(3):
                if data_array.shape[i] == 1:
                    output_to_input[i, :3] = 0
                    output_to_input[:3, i] = 0
                    output_to_input[i, i] = 1

            identity = np.eye(output_to_input.shape[0])
            if np.allclose(output_to_input, identity):
                if data_array.shape != tuple(shape):
                    pads = []
                    for small, large in zip(data_array.shape, shape):
                        diff = large - small
                        pads.append((0, diff))
                    array = np.pad(array, pads, mode="constant", constant_values=fill_value)
            else:
                array = ndimage.affine_transform(
                    data_array.data, output_to_input, output_shape=shape,
                    order=0, cval=fill_value, mode="grid-constant"
                )
        else:
            attrs["transform"] = transform.ijk_to_ras.flatten().tolist()

        if "reference" in attrs and attrs["reference"] not in arrays:
            del attrs["reference"]

        new_data_array = xr.DataArray(array, dims=data_array.dims, attrs=attrs)
        data_arrays[name] = new_data_array

    _remove_temporary_nodes()
    callback("Exporting to NetCDF...", 90)

    if save_in_place:
        for dim in existing_dims[:3]:
            coords[dim] = existing_dataset.coords[dim]
    elif single_coords:
        for min_, spacing, size, dim in zip(ras_min, ref_spacing, output_shape, "zyx"):
            max_ = min_ + spacing * (size - 1)
            coords[dim] = np.linspace(min_, max_, size)
    else:
        for name, data_array in data_arrays.items():
            node = arrays[name][2]
            origin_zyx = list(node.GetOrigin()[::-1])
            origin_zyx[1] *= -1
            origin_zyx[2] *= -1
            spacing_zyx = node.GetSpacing()[::-1]
            for origin, spacing, size, dim in zip(
                origin_zyx, spacing_zyx, data_array.shape, "zyx"
            ):
                coord_name = f"{dim}_{name}"
                coords[coord_name] = np.linspace(origin, origin + spacing * (size - 1), size)

    dataset = xr.Dataset(data_arrays, coords=coords)
    table_dataset = xr.Dataset(table_arrays)
    dataset.update(table_dataset)

    for name in node_attrs:
        if name in dataset:
            dataset[name].attrs.update(node_attrs[name])

    if save_in_place:
        for var in dataset:
            existing_dataset[var] = dataset[var]
        dataset = existing_dataset

    encoding = {}
    for var in dataset:
        img = dataset[var]
        encoding[var] = {"zlib": use_compression, "chunksizes": _recommended_chunksizes(img)}

    dataset.attrs["amora_version"] = slicer.app.applicationVersion
    dataset.attrs.update(dataset_attrs)
    dataset.to_netcdf(str(export_path), encoding=encoding, format="NETCDF4")

    return warnings


# ===========================================================================
# DISPLAY SETUP
# ===========================================================================

def _subsample(arr, step=4):
    """Return a subsampled view of a 3D array for fast percentile estimation."""
    if arr.size < 1_000_000:
        return arr
    return arr[::step, ::step, ::step]


def _setup_display(nodes):
    """Configure slice viewers and window/level for loaded volumes."""
    first_scalar = None
    for n in nodes:
        if n is not None and hasattr(n, "IsA") and n.IsA("vtkMRMLScalarVolumeNode"):
            first_scalar = n
            break

    if first_scalar:
        slicer.util.setSliceViewerLayers(background=first_scalar, fit=True)

        # Compute window/level from subsampled data percentiles
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

        slicer.util.resetSliceViews()


# ===========================================================================
# FILE IMPORT
# ===========================================================================

def import_file(path, callback=None, images="all"):
    """
    Load a single .nc / .h5 / .hdf5 file into Slicer's MRML scene.
    Returns list of created volume nodes.
    """
    import xarray as xr

    path = Path(path)
    dataset = xr.open_dataset(str(path))
    return _handle_dataset(dataset, path, callback, images)


def import_directory(path, callback=None, images="all"):
    """
    Load multiple .nc files from a directory, concatenating variables.
    Returns list of created volume nodes.
    """
    path = Path(path)
    dataset_list = _open_dataset_from_directory(path)
    all_nodes = []
    for dataset in dataset_list:
        nodes = _handle_dataset(dataset, path, callback, images)
        all_nodes.extend(nodes)
    return all_nodes


# Keep backward-compat alias
import_nc_file = import_file


# ===========================================================================
# NPY LOADER
# ===========================================================================

def read_npy_data(path: Union[str, Path]) -> dict:
    """
    Read a .npy file into a dict ready for scene creation.
    Returns dict with 'array', 'name', 'window', 'level'.
    Tries memory-mapping first, falls back to regular load.
    """
    path = Path(path)

    try:
        arr = np.load(str(path), mmap_mode='r')
    except Exception:
        arr = np.load(str(path), allow_pickle=False)

    if arr.ndim < 3:
        raise ValueError(f"Expected 3D array, got {arr.ndim}D: shape={arr.shape}")

    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255

    if arr.dtype not in (np.float32, np.float64, np.uint8, np.uint16, np.int16):
        tensor = np.ascontiguousarray(arr, dtype=np.float32)
    else:
        tensor = np.ascontiguousarray(arr)

    sub = _subsample(tensor)
    p2 = float(np.percentile(sub, 2))
    p98 = float(np.percentile(sub, 98))
    window = max(p98 - p2, 1.0)
    level = (p98 + p2) / 2.0

    return {
        "array": tensor,
        "name": f"AMORA_{path.stem}",
        "window": window,
        "level": level,
    }


def import_npy_file(path: Union[str, Path]) -> slicer.vtkMRMLScalarVolumeNode:
    """Load a .npy volume file into Slicer."""
    data = read_npy_data(path)
    return _create_scalar_node(data)


def read_raw_data(
    path: Union[str, Path],
    shape: Tuple[int, int, int],
    dtype: str = "uint8",
    byte_order: str = "little",
    header_offset: int = 0,
) -> dict:
    """Read a .raw file using memory-mapping. Returns dict for scene creation."""
    path = Path(path)
    np_dtype = np.dtype(dtype)
    if byte_order == "big":
        np_dtype = np_dtype.newbyteorder(">")

    expected = shape[0] * shape[1] * shape[2]
    mmap = np.memmap(str(path), dtype=np_dtype, mode='r', offset=header_offset,
                     shape=(expected,))
    mmap = mmap.reshape(shape)

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
    """Load a headerless .raw volume file."""
    data = read_raw_data(path, shape, dtype, byte_order, header_offset)
    return _create_scalar_node(data)


# ===========================================================================
# CT PROJECTION LOADER  (folder of .raw projections + geometry.yaml)
# ===========================================================================

def _detect_projection_folder(path: Path):
    """
    Detect if a directory contains CT projection data.
    Returns (proj_dir, geom_path, config_path) or None.

    Supports two layouts:
      1. folder/ containing geometry.yaml + projections/*.raw
      2. folder/ directly containing *.raw + geometry.yaml (or ../geometry.yaml)
    """
    geom = path / "geometry.yaml"
    config = path / "config.yaml"
    proj_dir = path / "projections"

    if geom.exists() and proj_dir.is_dir():
        raw_files = sorted(proj_dir.glob("*.raw"))
        if raw_files:
            return proj_dir, geom, config if config.exists() else None

    raw_files = sorted(path.glob("*.raw"))
    if raw_files:
        parent_geom = path.parent / "geometry.yaml"
        parent_config = path.parent / "config.yaml"
        if parent_geom.exists():
            return path, parent_geom, parent_config if parent_config.exists() else None
        if geom.exists():
            return path, geom, config if config.exists() else None

    return None


def _parse_geometry_yaml(geom_path: Path) -> dict:
    """Parse geometry.yaml and return relevant CT acquisition parameters."""
    if yaml is None:
        info = {}
        with open(geom_path, 'r') as f:
            text = f.read()
        import re as _re
        m = _re.search(r'nDetector:\s*\[(\d+),\s*(\d+)\]', text)
        if m:
            info['nDetector'] = [int(m.group(1)), int(m.group(2))]
        m = _re.search(r'nVoxel:\s*\[(\d+),\s*(\d+),\s*(\d+)\]', text)
        if m:
            info['nVoxel'] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
        m = _re.search(r'dVoxel:\s*\[([\d.e+-]+),\s*([\d.e+-]+),\s*([\d.e+-]+)\]', text)
        if m:
            info['dVoxel'] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
        m = _re.search(r'dDetector:\s*\[([\d.e+-]+),\s*([\d.e+-]+)\]', text)
        if m:
            info['dDetector'] = [float(m.group(1)), float(m.group(2))]
        m = _re.search(r'DSO:\s*([\d.e+-]+)', text)
        if m:
            info['DSO'] = float(m.group(1))
        m = _re.search(r'DSD:\s*([\d.e+-]+)', text)
        if m:
            info['DSD'] = float(m.group(1))
        m = _re.search(r'mode:\s*(\w+)', text)
        if m:
            info['mode'] = m.group(1)
        return info
    else:
        with open(geom_path, 'r') as f:
            return yaml.safe_load(f)


def import_projections_folder(
    path: Path,
    max_projections: int = 0,
    progress_callback=None,
) -> slicer.vtkMRMLScalarVolumeNode:
    """
    Load a folder of .raw CT projections into Slicer as a 3D volume.
    Each .raw file is one 2D projection; they are stacked into (N, H, W).
    """
    detected = _detect_projection_folder(path)
    if detected is None:
        raise ValueError(f"Not a valid projection folder: {path}")

    proj_dir, geom_path, config_path = detected

    geom = _parse_geometry_yaml(geom_path)
    n_det = geom.get('nDetector', [0, 0])
    d_det = geom.get('dDetector', [1.0, 1.0])
    d_voxel = geom.get('dVoxel', [1.0, 1.0, 1.0])
    ct_mode = geom.get('mode', 'cone')

    det_h, det_w = n_det[0], n_det[1]
    expected_size = det_h * det_w * 4

    logger.info(f"CT geometry: mode={ct_mode}, detector={det_h}x{det_w}, "
                f"dDetector={d_det}, dVoxel={d_voxel}")

    raw_files = sorted(proj_dir.glob("*.raw"))
    if not raw_files:
        raise ValueError(f"No .raw files found in {proj_dir}")

    first_size = raw_files[0].stat().st_size
    if expected_size > 0 and first_size != expected_size:
        if first_size == det_h * det_w * 2:
            inferred_dtype = np.uint16
            logger.info("Inferred dtype: uint16")
        elif first_size == det_h * det_w * 8:
            inferred_dtype = np.float64
            logger.info("Inferred dtype: float64")
        elif first_size == det_h * det_w * 4:
            inferred_dtype = np.float32
        else:
            inferred_dtype = np.float32
            n_pixels = first_size // 4
            det_h_guess = int(np.sqrt(n_pixels * det_h / max(det_w, 1)))
            det_w_guess = n_pixels // max(det_h_guess, 1)
            if det_h_guess * det_w_guess == n_pixels:
                det_h, det_w = det_h_guess, det_w_guess
            logger.warning(f"File size {first_size} != expected {expected_size}. "
                          f"Using inferred shape ({det_h}, {det_w})")
    else:
        inferred_dtype = np.float32

    n_total = len(raw_files)
    if max_projections > 0:
        raw_files = raw_files[:max_projections]
    n_load = len(raw_files)

    logger.info(f"Loading {n_load}/{n_total} projections, "
                f"shape per proj: ({det_h}, {det_w}), dtype={inferred_dtype}")

    mem_gb = n_load * det_h * det_w * np.dtype(inferred_dtype).itemsize / (1024**3)
    logger.info(f"Estimated memory: {mem_gb:.2f} GB")

    volume = np.empty((n_load, det_h, det_w), dtype=np.float32)
    for i, raw_file in enumerate(raw_files):
        proj = np.fromfile(str(raw_file), dtype=inferred_dtype)
        proj = proj.reshape(det_h, det_w)
        volume[i] = proj.astype(np.float32)
        if progress_callback and i % 50 == 0:
            progress_callback(i, n_load)

    if progress_callback:
        progress_callback(n_load, n_load)

    name = f"Projections_{path.name}"
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
    node.CreateDefaultDisplayNodes()

    slicer.util.updateVolumeFromArray(node, volume)

    node.SetSpacing(d_det[1], d_det[0], 1.0)
    node.SetIJKToRASDirections(-1, 0, 0, 0, -1, 0, 0, 0, 1)

    node.SetAttribute("AMORA.ct_mode", ct_mode)
    node.SetAttribute("AMORA.nDetector", f"{det_h},{det_w}")
    node.SetAttribute("AMORA.nProjections", str(n_load))
    if 'DSO' in geom:
        node.SetAttribute("AMORA.DSO", str(geom['DSO']))
    if 'DSD' in geom:
        node.SetAttribute("AMORA.DSD", str(geom['DSD']))
    if 'nVoxel' in geom:
        nv = geom['nVoxel']
        node.SetAttribute("AMORA.nVoxel", f"{nv[0]},{nv[1]},{nv[2]}")
    if d_voxel:
        node.SetAttribute("AMORA.dVoxel", f"{d_voxel[0]},{d_voxel[1]},{d_voxel[2]}")

    slicer.util.setSliceViewerLayers(background=node, fit=True)
    display = node.GetDisplayNode()
    if display:
        sub = _subsample(volume)
        p2 = float(np.percentile(sub, 2))
        p98 = float(np.percentile(sub, 98))
        window = max(p98 - p2, 1.0)
        level = (p98 + p2) / 2.0
        display.AutoWindowLevelOff()
        display.SetWindow(window)
        display.SetLevel(level)
    slicer.util.resetSliceViews()

    logger.info(f"Loaded projections: {name}, shape={volume.shape}, "
                f"range=[{volume.min():.2f}, {volume.max():.2f}]")

    return node


# ===========================================================================
# UNIFIED LOADER
# ===========================================================================

def read_volume_data(path: Union[str, Path], **kwargs) -> list:
    """
    Thread-safe: read volume data from disk without touching Slicer scene.
    Returns list of dicts with 'array', 'name', 'window', 'level'.
    For .nc/.h5 files returns None (must use synchronous load_volume).
    """
    path = Path(path)

    if path.is_dir():
        return None

    ext = path.suffix.lower()

    if ext == ".npy":
        return [read_npy_data(path)]
    elif ext == ".raw":
        if "shape" not in kwargs:
            raise ValueError(".raw files require 'shape' parameter: (Z, Y, X)")
        return [read_raw_data(path, **kwargs)]
    elif ext in NETCDF_EXTENSIONS:
        return None
    else:
        return None


def load_volume(path: Union[str, Path], **kwargs) -> list:
    """
    Load any supported volume file or directory. Auto-detects format.
    Returns list of volume nodes.

    - .nc / .h5 / .hdf5 file  -> import_file()
    - Directory with CT projections -> import_projections_folder()
    - Directory with .nc files -> import_directory()
    - .npy file               -> import_npy_file()
    - .raw file               -> import_raw_file() (requires shape kwarg)
    - Everything else         -> Slicer's native loader as fallback
    """
    path = Path(path)

    if path.is_dir():
        # Check for CT projection folder first
        if _detect_projection_folder(path) is not None:
            max_proj = kwargs.get("max_projections", 0)
            node = import_projections_folder(
                path, max_projections=max_proj,
                progress_callback=kwargs.get("progress_callback")
            )
            return [node]

        nc_files = list(path.glob("*.nc"))
        if nc_files:
            return import_directory(path, callback=kwargs.get("callback"))

        raise ValueError(
            f"No supported data found in directory: {path}\n"
            f"Expected: .nc files, or .raw projections + geometry.yaml"
        )

    ext = path.suffix.lower()

    if ext == ".npy":
        return [import_npy_file(path)]
    elif ext in NETCDF_EXTENSIONS:
        return import_file(path, callback=kwargs.get("callback"))
    elif ext == ".raw":
        if "shape" not in kwargs:
            raise ValueError(".raw files require 'shape' parameter: (Z, Y, X)")
        return [import_raw_file(path, **kwargs)]
    else:
        node = slicer.util.loadVolume(str(path))
        if node:
            return [node]
        raise ValueError(f"Unsupported file format: {ext}")
