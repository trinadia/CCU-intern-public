"""
Standard sequential ICP for the D435_pcl_2026-10-07 dataset.

Input:
    D435_pcl_2026-10-07.zip
    Reads ONLY files ending in *_filtered_pcl.npy.
    Each file must contain an (N, 3) XYZ point cloud in meters.

Method:
    Standard Open3D point-to-point ICP.
    No robot poses, no multi-pair registration, no pose graph, no ML.

Ordering:
    Sorts clouds chronologically using the timestamp in each filename.
    The archive contains paired white_object_0000 / white_object_0001 clouds
    for the same capture times; both are retained and interleaved by time.
    If you want only one set, set OBJECT_INDEX below to "0000" or "0001".
"""

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import re

import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

ZIP_PATH = Path(r"2026-10-07\D435_pcl_2026-10-07.zip")
OUTPUT_FOLDER = Path("icp_output_8cm")

# None = use both white_object_0000 and white_object_0001.
# Set to "0000" or "0001" to use only that set of clouds.
OBJECT_INDEX = None

# Optional inclusive Z-depth filter, in meters. None disables each bound.
# The input files are already *_filtered_pcl.npy point clouds.
MIN_DEPTH = 0.05
MAX_DEPTH = 0.50

# Optional XYZ cropping. Use None to disable, otherwise [min, max].
X_RANGE = None
Y_RANGE = None
Z_RANGE = None

# Voxel downsampling before ICP; 0 or None disables it.
VOXEL_SIZE = 0.002  # 2 mm

# Standard point-to-point ICP settings.
MAX_CORRESPONDENCE_DISTANCE = 0.2  # meters
MAX_ITERATIONS = 200
RELATIVE_FITNESS = 1e-6
RELATIVE_RMSE = 1e-6

# Final merged cloud downsampling; 0 or None disables it.
FINAL_VOXEL_SIZE = 0.002

SAVE_FILTERED_CLOUDS = False
SAVE_REGISTERED_CLOUDS = False
VISUALIZE_RESULT = True


# ============================================================
# INPUT / PREPROCESSING
# ============================================================

def timestamp_key(filename):
    """Sort by YYYYMMDD_HHMMSS in the dataset filename."""
    match = re.search(r"_(\d{8})_(\d{6})_", filename)
    if match:
        return match.group(1) + match.group(2)
    return filename


def object_index_key(filename):
    match = re.search(r"white_object_(\d+)_", filename)
    return match.group(1) if match else ""


def load_clouds_from_zip(zip_path):
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Cannot find ZIP file: {zip_path.resolve()}\n"
            "Put the ZIP beside this script or update ZIP_PATH."
        )

    with ZipFile(zip_path, "r") as archive:
        names = [
            name for name in archive.namelist()
            if name.endswith("_filtered_pcl.npy")
        ]

        if OBJECT_INDEX is not None:
            names = [
                name for name in names
                if object_index_key(Path(name).name) == str(OBJECT_INDEX)
            ]

        names.sort(key=lambda name: (
            timestamp_key(Path(name).name),
            object_index_key(Path(name).name),
            Path(name).name,
        ))

        if not names:
            raise RuntimeError(
                "No *_filtered_pcl.npy files found in the ZIP "
                "with the current OBJECT_INDEX setting."
            )

        clouds = []
        stats = []
        for name in names:
            array = np.load(BytesIO(archive.read(name)), allow_pickle=False)
            array = np.asarray(array, dtype=np.float64)

            if array.ndim != 2 or array.shape[1] != 3:
                print(f"[SKIP] Unexpected array shape {array.shape}: {name}")
                continue

            # Remove NaN/Inf rows.
            valid = np.all(np.isfinite(array), axis=1)
            array = array[valid]
            original_count = len(array)

            # Optional depth and spatial filters.
            keep = np.ones(len(array), dtype=bool)
            if MIN_DEPTH is not None:
                keep &= array[:, 2] >= MIN_DEPTH
            if MAX_DEPTH is not None:
                keep &= array[:, 2] <= MAX_DEPTH
            if X_RANGE is not None:
                keep &= (array[:, 0] >= X_RANGE[0]) & (array[:, 0] <= X_RANGE[1])
            if Y_RANGE is not None:
                keep &= (array[:, 1] >= Y_RANGE[0]) & (array[:, 1] <= Y_RANGE[1])
            if Z_RANGE is not None:
                keep &= (array[:, 2] >= Z_RANGE[0]) & (array[:, 2] <= Z_RANGE[1])
            array = array[keep]

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(array)

            if VOXEL_SIZE is not None and VOXEL_SIZE > 0:
                cloud = cloud.voxel_down_sample(VOXEL_SIZE)

            if len(cloud.points) < 10:
                print(f"[SKIP] Too few points after filtering: {name}")
                continue

            clouds.append(cloud)
            stats.append({
                "name": Path(name).name,
                "original": original_count,
                "filtered": int(np.count_nonzero(keep)),
                "downsampled": len(cloud.points),
            })

        return clouds, stats


def save_cloud(cloud, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise IOError(f"Could not write point cloud: {path}")


# ============================================================
# STANDARD ICP
# ============================================================

def run_icp(source, target):
    """Estimate the source-to-target transform, starting from identity."""
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=RELATIVE_FITNESS,
        relative_rmse=RELATIVE_RMSE,
        max_iteration=MAX_ITERATIONS,
    )
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        MAX_CORRESPONDENCE_DISTANCE,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
        criteria,
    )
    return result


def main():
    output = OUTPUT_FOLDER
    output.mkdir(parents=True, exist_ok=True)
    filtered_dir = output / "preprocessed_clouds"
    registered_dir = output / "registered_clouds"

    print("=" * 72)
    print("STANDARD SEQUENTIAL ICP — D435 PCL 2026-10-07")
    print("=" * 72)
    print(f"Input ZIP: {ZIP_PATH.resolve()}")
    print(f"Object index filter: {OBJECT_INDEX}")
    print("ICP: point-to-point, identity initial transform")
    print("Registration uses point clouds only; robot pose is not used.")

    clouds, stats = load_clouds_from_zip(ZIP_PATH)
    if len(clouds) < 2:
        raise RuntimeError(
            f"Need at least 2 valid point clouds; found {len(clouds)}."
        )

    print(f"\nLoaded {len(clouds)} point clouds.")
    for i, stat in enumerate(stats):
        print(
            f"  PC_{i + 1:02d}: {stat['name']} | "
            f"points {stat['original']} -> {stat['filtered']} -> "
            f"{stat['downsampled']}"
        )
        if SAVE_FILTERED_CLOUDS:
            save_cloud(clouds[i], filtered_dir / f"PC_{i + 1:02d}.ply")

    # The first cloud defines the reference coordinate frame.
    global_transforms = [np.eye(4)]
    registered_clouds = [o3d.geometry.PointCloud(clouds[0])]
    pair_results = []

    if SAVE_REGISTERED_CLOUDS:
        save_cloud(registered_clouds[0], registered_dir / "PC_0001.ply")

    for i in range(1, len(clouds)):
        source = clouds[i]
        target = clouds[i - 1]

        print(f"\nRegistering PC_{i + 1:02d} -> PC_{i:02d} ...")
        result = run_icp(source, target)
        T_source_to_previous = result.transformation

        # T_global_i = T_global_(i-1) @ T_(i -> i-1)
        T_global_current = global_transforms[i - 1] @ T_source_to_previous
        global_transforms.append(T_global_current)

        aligned = o3d.geometry.PointCloud(source)
        aligned.transform(T_global_current)
        registered_clouds.append(aligned)

        pair_results.append({
            "source_index": i,
            "target_index": i - 1,
            "fitness": float(result.fitness),
            "rmse": float(result.inlier_rmse),
            "transform": T_source_to_previous.copy(),
        })

        print(
            f"  fitness={result.fitness:.6f}, "
            f"RMSE={result.inlier_rmse:.6f} m"
        )
        if result.fitness < 0.10:
            print("  WARNING: low fitness; this pair may not be well aligned.")

        if SAVE_REGISTERED_CLOUDS:
            save_cloud(
                aligned,
                registered_dir / f"PC_{i + 1:04d}.ply",
            )

    # Merge registered clouds in the first cloud's coordinate frame.
    final_cloud = o3d.geometry.PointCloud()
    for cloud in registered_clouds:
        final_cloud += cloud

    if FINAL_VOXEL_SIZE is not None and FINAL_VOXEL_SIZE > 0:
        final_cloud = final_cloud.voxel_down_sample(FINAL_VOXEL_SIZE)

    final_path = output / "final_registered.ply"
    save_cloud(final_cloud, final_path)

    # Save transformations and metrics.
    np.savez_compressed(
        output / "icp_transformations.npz",
        global_transforms=np.asarray(global_transforms),
        pairwise_transforms=np.asarray([p["transform"] for p in pair_results]),
        pairwise_fitness=np.asarray([p["fitness"] for p in pair_results]),
        pairwise_rmse=np.asarray([p["rmse"] for p in pair_results]),
        filenames=np.asarray([s["name"] for s in stats]),
    )

    report_path = output / "icp_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("STANDARD SEQUENTIAL ICP REPORT\n")
        f.write("=" * 72 + "\n")
        f.write(f"Input ZIP: {ZIP_PATH}\n")
        f.write(f"Object index filter: {OBJECT_INDEX}\n")
        f.write(f"Number of clouds: {len(clouds)}\n")
        f.write("ICP method: point-to-point\n")
        f.write(f"Max correspondence distance: {MAX_CORRESPONDENCE_DISTANCE} m\n")
        f.write(f"Max iterations: {MAX_ITERATIONS}\n")
        f.write(f"Voxel size before ICP: {VOXEL_SIZE}\n")
        f.write(f"Final voxel size: {FINAL_VOXEL_SIZE}\n\n")

        f.write("POINT COUNTS\n")
        for i, stat in enumerate(stats):
            f.write(
                f"PC_{i + 1:02d} | {stat['name']} | "
                f"valid={stat['original']} | filtered={stat['filtered']} | "
                f"downsampled={stat['downsampled']}\n"
            )

        f.write("\nPAIRWISE ICP RESULTS\n")
        for p in pair_results:
            f.write(
                f"\nPC_{p['source_index'] + 1:02d} -> "
                f"PC_{p['target_index'] + 1:02d}\n"
            )
            f.write(f"Fitness: {p['fitness']:.8f}\n")
            f.write(f"Inlier RMSE: {p['rmse']:.8f} m\n")
            f.write("Transformation (source -> target):\n")
            f.write(np.array2string(p["transform"], precision=8, suppress_small=True))
            f.write("\n")

        f.write("\nGLOBAL TRANSFORMS (PC_i -> PC_1)\n")
        for i, transform in enumerate(global_transforms):
            f.write(f"\nPC_{i + 1:02d} -> PC_01:\n")
            f.write(np.array2string(transform, precision=8, suppress_small=True))
            f.write("\n")

        f.write("\nNotes:\n")
        f.write(
            "This is standard sequential point-to-point ICP. It does not use "
            "robot poses, multi-pair registration, pose-graph optimization, "
            "or machine learning. ICP is a local optimizer and can fail when "
            "consecutive views have insufficient overlap or large initial misalignment.\n"
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Final point cloud: {final_path.resolve()}")
    print(f"ICP report:        {report_path.resolve()}")
    print(f"Transforms:        {(output / 'icp_transformations.npz').resolve()}")
    print(f"Registered clouds: {registered_dir.resolve()}")

    if VISUALIZE_RESULT:
        o3d.visualization.draw_geometries(
            [final_cloud],
            window_name="Standard Sequential ICP - D435 PCL",
            width=1200,
            height=800,
        )


if __name__ == "__main__":
    main()
