"""I/O for TreeSegPy: read and write LAS/LAZ point clouds with tree labels."""
from pathlib import Path
import numpy as np
import laspy


def read_laz(path):
    """
    Read a LAZ/LAS file and return the arrays TreeSegPy needs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAS/LAZ not found: {path}")

    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    if "treeID" in las.point_format.dimension_names:
        tree_id = np.asarray(las.treeID).astype(np.int32)
    else:
        tree_id = np.zeros(len(xyz), dtype=np.int32)

    if "intensity" in las.point_format.dimension_names:
        intensity = np.asarray(las.intensity).astype(np.float32)
    else:
        intensity = np.zeros(len(xyz), dtype=np.float32)

    return {
        "xyz": xyz,
        "tree_id": tree_id,
        "intensity": intensity,
        "header": las.header,
    }


def write_laz(path, xyz, tree_label=None, tree_prob=None,
              original=None, template_header=None):
    """Write a LAS/LAZ file, preserving all original fields if provided."""
    import laspy

    xyz = np.asarray(xyz, dtype=np.float64)

    if original is not None:
        header = laspy.LasHeader(
            point_format=original.header.point_format,
            version=original.header.version,
        )
        header.scales = original.header.scales
        header.offsets = original.header.offsets
    elif template_header is not None:
        header = laspy.LasHeader(
            point_format=template_header.point_format,
            version=template_header.version,
        )
        header.scales = template_header.scales
        header.offsets = template_header.offsets
    else:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.001, 0.001, 0.001]
        header.offsets = xyz.min(axis=0)

    if tree_label is not None and "tree_label" not in header.point_format.dimension_names:
        header.add_extra_dim(laspy.ExtraBytesParams(name="tree_label", type=np.uint8))
    if tree_prob is not None and "tree_prob" not in header.point_format.dimension_names:
        header.add_extra_dim(laspy.ExtraBytesParams(name="tree_prob", type=np.float32))

    las = laspy.LasData(header)

    if original is not None:
        for dim in original.point_format.dimension_names:
            try:
                setattr(las, dim, np.asarray(original[dim]))
            except Exception as e:
                print(f"  warning: could not copy field '{dim}': {e}")

    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    if tree_label is not None:
        las.tree_label = np.asarray(tree_label, dtype=np.uint8)
    if tree_prob is not None:
        las.tree_prob = np.asarray(tree_prob, dtype=np.float32)

    las.header.point_count = len(xyz)
    if original is not None:
        orig_counts = np.asarray(original.header.number_of_points_by_return)
        if orig_counts.sum() == len(xyz):
            las.header.number_of_points_by_return = orig_counts
        else:
            n = len(las.header.number_of_points_by_return)
            las.header.number_of_points_by_return = np.zeros(n, dtype=np.uint32)
            las.header.number_of_points_by_return[0] = len(xyz)

    las.write(path)
    return path


def list_plots(directory, pattern="*.laz"):
    return sorted(Path(directory).glob(pattern))