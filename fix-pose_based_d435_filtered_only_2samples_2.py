#!/usr/bin/env python3
"""
Pose-based multi-view point-cloud registration for D435 filtered PCL data.

IMPORTANT:
- Uses ONLY *_filtered_pcl.npy files.
- The two samples belonging to one robot pose are NOT assumed to be adjacent.
- Samples are paired by the acquisition timestamp in the filename.
  Example:
      white_object_0000_20261007_114517_..._filtered_pcl.npy
      white_object_0001_20261007_114518_..._filtered_pcl.npy
  are treated as one pose because their timestamps are within the same
  acquisition event.
- The pairing logic is based on the object index (0000/0001) and the
  timestamp. The files are grouped by their HHMMSS timestamp, with a
  configurable timestamp tolerance.

Frame convention assumed:
    T_base_tool0  : robot base -> tool0
    T_tool0_camera: tool0 -> camera

Therefore:
    T_base_camera = T_base_tool0 @ T_tool0_camera

and:
    p_base = T_base_camera @ p_camera

NO ICP is used in this script.
"""

import os
import re
import glob
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


# ============================================================
# USER SETTINGS
# ============================================================

NPY_FOLDER = r"2026-10-07\D435_pcl_2026-10-07"
POSE_FILE = r"2026-10-07\D435_pcl_pose_2026-10-07.txt"
OUTPUT_FOLDER = r"modified-pose_output_d435_filtered_07-10"

# Use all detected pose groups by default.
START_INDEX = 0
END_INDEX = None

# Expected number of filtered samples per robot pose.
EXPECTED_SAMPLES_PER_POSE = 2

# Timestamp tolerance in seconds.
# Your example has:
#   114517 vs 114518
#   120410 vs 120411
# so 2 seconds is sufficient.
TIMESTAMP_TOLERANCE_SEC = 2

# Final merged cloud voxel size in meters.
FINAL_VOXEL_SIZE = 0.001

# "base" = reconstruct directly in robot base frame.
# "first_pose" = express everything relative to the first camera pose.
REFERENCE_FRAME = "base"

SAVE_INDIVIDUAL_CLOUDS = True

# Show the final reconstructed cloud in an Open3D window.
SHOW_VISUALIZATION = True

# Coordinate range filters (meters), applied AFTER robot-pose transformation.
# Limits are evaluated in REFERENCE_FRAME (base or first_pose).
# Set a range to None to disable that axis filter.
X_RANGE = (-0.50, -0.44)
Y_RANGE = (-0.16, -0.06)
Z_RANGE = (0.005, 0.06)


# ============================================================
# FIXED TOOL0 -> CAMERA EXTRINSIC
# ============================================================

T_TOOL0_CAMERA = np.array([
    [-0.9992735273,  0.0253477268,  0.0284589249,  0.0392389837],
    [-0.0257328620, -0.9995810504, -0.0132492836, -0.0923932600],
    [ 0.0281111629, -0.0139719879,  0.9995071516,  0.0921349981],
    [ 0.0,           0.0,           0.0,           1.0]
], dtype=np.float64)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def timestamp_to_seconds(hhmmss):
    """Convert HHMMSS string to seconds since midnight."""
    h = int(hhmmss[0:2])
    m = int(hhmmss[2:4])
    s = int(hhmmss[4:6])
    return h * 3600 + m * 60 + s


def parse_filtered_filename(path):
    """
    Parse filenames of the form:

    white_object_0000_20261007_114517_points_021143_filtered_pcl.npy

    Returns:
        object_id, date, time, point_count
    """
    name = os.path.basename(path)

    pattern = (
        r"^white_object_(\d+)_(\d{8})_(\d{6})"
        r"_points_(\d+)_filtered_pcl\.npy$"
    )

    m = re.match(pattern, name)

    if not m:
        return None

    object_id = int(m.group(1))
    date = m.group(2)
    time = m.group(3)
    point_count = int(m.group(4))

    return {
        "path": path,
        "name": name,
        "object_id": object_id,
        "date": date,
        "time": time,
        "time_sec": timestamp_to_seconds(time),
        "point_count_from_name": point_count,
    }


def discover_filtered_clouds(folder):
    """Find ONLY filtered point-cloud NPY files."""
    pattern = os.path.join(folder, "*_filtered_pcl.npy")
    files = sorted(glob.glob(pattern))

    parsed = []

    for f in files:
        info = parse_filtered_filename(f)
        if info is None:
            print(f"[WARNING] Filename does not match expected format: {f}")
            continue

        parsed.append(info)

    return parsed


def group_samples_by_timestamp(cloud_infos):
    """
    Group the two samples belonging to the same acquisition event.

    The files are paired by:
      1. same date
      2. timestamp difference <= TIMESTAMP_TOLERANCE_SEC
      3. normally one object_0000 and one object_0001

    This means filesystem ordering is irrelevant.
    """

    # Sort by actual timestamp, then object ID.
    cloud_infos = sorted(
        cloud_infos,
        key=lambda x: (x["date"], x["time_sec"], x["object_id"])
    )

    groups = []

    for info in cloud_infos:
        if not groups:
            groups.append([info])
            continue

        current = groups[-1]

        same_date = current[0]["date"] == info["date"]
        time_difference = abs(
            current[0]["time_sec"] - info["time_sec"]
        )

        # Add to the current acquisition event if close enough.
        if same_date and time_difference <= TIMESTAMP_TOLERANCE_SEC:
            # Avoid accidentally putting duplicate object IDs into one group.
            object_ids = {x["object_id"] for x in current}

            if info["object_id"] not in object_ids:
                current.append(info)
            else:
                # Same object ID: start a new group rather than silently
                # combining two separate samples.
                groups.append([info])
        else:
            groups.append([info])

    return groups


def parse_pose_blocks(pose_file):
    """
    Parse the actual D435 pose TXT format.

    Each pose block has:
        At time <timestamp>
        - Translation: [x, y, z]
        - Rotation: in Quaternion (xyzw) [qx, qy, qz, qw]
        - Rotation: in RPY ...
        - Matrix: ...

    The quaternion is used to reconstruct T_base_tool0.
    """

    with open(pose_file, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(
        r"(?=^\s*At time\s+)",
        text,
        flags=re.MULTILINE
    )

    poses = []

    for block in blocks:
        if not re.search(r"^\s*At time\s+", block, flags=re.MULTILINE):
            continue

        m_time = re.search(
            r"^\s*At time\s+([0-9.]+)",
            block,
            flags=re.MULTILINE
        )
        m_translation = re.search(
            r"^\s*-\s*Translation:\s*\[([^\]]+)\]",
            block,
            flags=re.MULTILINE
        )
        m_quaternion = re.search(
            r"^\s*-\s*Rotation:\s*in Quaternion\s*\(xyzw\)\s*\[([^\]]+)\]",
            block,
            flags=re.MULTILINE
        )

        if m_time is None or m_translation is None or m_quaternion is None:
            continue

        timestamp = float(m_time.group(1))

        translation = np.fromstring(
            m_translation.group(1), sep=",", dtype=np.float64
        )
        quaternion = np.fromstring(
            m_quaternion.group(1), sep=",", dtype=np.float64
        )

        if translation.shape != (3,):
            raise ValueError(
                f"Invalid translation in pose at time {timestamp}: "
                f"{translation}"
            )

        if quaternion.shape != (4,):
            raise ValueError(
                f"Invalid quaternion in pose at time {timestamp}: "
                f"{quaternion}"
            )

        R = Rotation.from_quat(quaternion).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = translation

        poses.append({
            "pose_number": len(poses) + 1,
            "timestamp": timestamp,
            "translation": translation,
            "quaternion": quaternion,
            "T_base_tool0": T,
        })

    if not poses:
        raise RuntimeError(
            f"No valid robot poses were parsed from: {pose_file}"
        )

    return poses

def filter_pointcloud(points):
    """
    Apply X/Y/Z range filters to an Nx3 point cloud AFTER pose
    transformation, in the selected reference coordinate frame.
    """
    if points.size == 0:
        return points

    # Remove NaN/Inf coordinates as well.
    mask = np.isfinite(points).all(axis=1)

    if X_RANGE is not None:
        mask &= (
            (points[:, 0] >= X_RANGE[0]) &
            (points[:, 0] <= X_RANGE[1])
        )

    if Y_RANGE is not None:
        mask &= (
            (points[:, 1] >= Y_RANGE[0]) &
            (points[:, 1] <= Y_RANGE[1])
        )

    if Z_RANGE is not None:
        mask &= (
            (points[:, 2] >= Z_RANGE[0]) &
            (points[:, 2] <= Z_RANGE[1])
        )

    return points[mask]


def transform_points(points, T):
    """Apply 4x4 homogeneous transformation to Nx3 points."""
    if points.size == 0:
        return points

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points.astype(np.float64), ones])

    transformed = (T @ homogeneous.T).T

    return transformed[:, :3]


def save_ply(points, filename):
    """Save Nx3 points as PLY."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(filename, pcd)


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 70)
    print("POSE-BASED MULTI-VIEW POINT CLOUD REGISTRATION")
    print("=" * 70)
    print("ICP: DISABLED")
    print("Input: *_filtered_pcl.npy ONLY")
    print()

    # --------------------------------------------------------
    # 1. Discover point clouds
    # --------------------------------------------------------

    cloud_infos = discover_filtered_clouds(NPY_FOLDER)

    if len(cloud_infos) == 0:
        raise RuntimeError(
            f"No *_filtered_pcl.npy files found in: {NPY_FOLDER}"
        )

    print(f"Found {len(cloud_infos)} filtered point clouds.")

    # --------------------------------------------------------
    # 2. Group by acquisition timestamp
    # --------------------------------------------------------

    groups = group_samples_by_timestamp(cloud_infos)

    print(f"Detected {len(groups)} acquisition groups.")
    print()

    for i, group in enumerate(groups, start=1):
        print(f"Pose group {i}:")
        for item in group:
            print(
                f"  object_{item['object_id']:04d} | "
                f"{item['date']} {item['time']} | "
                f"{item['name']}"
            )

        if len(group) != EXPECTED_SAMPLES_PER_POSE:
            print(
                f"  [WARNING] Expected "
                f"{EXPECTED_SAMPLES_PER_POSE} samples, "
                f"but found {len(group)}."
            )

        print()

    # --------------------------------------------------------
    # 3. Apply index selection
    # --------------------------------------------------------

    end = END_INDEX if END_INDEX is not None else len(groups)

    groups = groups[START_INDEX:end]

    if len(groups) == 0:
        raise RuntimeError("No pose groups selected.")

    # --------------------------------------------------------
    # 4. Read robot poses
    # --------------------------------------------------------

    poses = parse_pose_blocks(POSE_FILE)

    print(f"Found {len(poses)} robot poses in pose file.")
    for p in poses:
        print(
            f"  Pose {p['pose_number']}: "
            f"timestamp={p['timestamp']:.9f}"
        )
    print(f"Selected {len(groups)} point-cloud pose groups.")
    print()

    if len(poses) != len(groups):
        raise RuntimeError(
            "\nMismatch between point-cloud groups and robot poses.\n"
            f"  Point-cloud groups: {len(groups)}\n"
            f"  Robot poses:        {len(poses)}\n\n"
            "The script cannot safely assign poses automatically.\n"
            "Make sure the pose TXT contains exactly one robot pose "
            "for each acquisition timestamp/group."
        )

    # --------------------------------------------------------
    # 5. Calculate base -> camera transforms
    # --------------------------------------------------------

    base_camera_matrices = []

    for pose in poses:
        T_base_camera = pose["T_base_tool0"] @ T_TOOL0_CAMERA
        base_camera_matrices.append(T_base_camera)

    # --------------------------------------------------------
    # 6. Determine reference transforms
    # --------------------------------------------------------

    if REFERENCE_FRAME.lower() == "base":
        reference_transforms = base_camera_matrices

    elif REFERENCE_FRAME.lower() == "first_pose":
        T_first_inv = np.linalg.inv(base_camera_matrices[0])

        reference_transforms = [
            T_first_inv @ T
            for T in base_camera_matrices
        ]

    else:
        raise ValueError(
            "REFERENCE_FRAME must be 'base' or 'first_pose'"
        )

    # --------------------------------------------------------
    # 7. Transform every cloud
    # --------------------------------------------------------

    transformed_clouds = []

    all_file_names = []
    all_pose_indices = []
    all_sample_indices = []
    all_transforms = []

    print("=" * 70)
    print("POSE ASSIGNMENT")
    print("=" * 70)

    for pose_idx, group in enumerate(groups):

        T = reference_transforms[pose_idx]

        print(
            f"\nPose {pose_idx + 1}: "
            f"{len(group)} samples"
        )

        for sample_idx, item in enumerate(group):

            print(
                f"  Sample {sample_idx + 1}: "
                f"{item['name']}"
            )

            points = np.load(item["path"])

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"Invalid point-cloud shape in {item['path']}: "
                    f"{points.shape}"
                )

            # Transform the original camera-frame points first.
            transformed = transform_points(points, T)
            transformed_count = len(transformed)

            # Apply X/Y/Z limits in the selected reference frame AFTER
            # the robot-pose transformation.
            filtered_transformed = filter_pointcloud(transformed)
            filtered_count = len(filtered_transformed)

            print(
                f"    Points: {transformed_count:,} -> {filtered_count:,} "
                f"after post-transform X/Y/Z filtering"
            )

            if filtered_count == 0:
                print(
                    "    [WARNING] No points remain after the post-transform "
                    f"range filter for {item['name']}. Check the configured "
                    "ranges against the transformed coordinate ranges."
                )

            transformed_clouds.append(filtered_transformed)

            all_file_names.append(item["name"])
            all_pose_indices.append(pose_idx)
            all_sample_indices.append(sample_idx)
            all_transforms.append(T)

            if SAVE_INDIVIDUAL_CLOUDS:
                output_name = (
                    f"pose_{pose_idx + 1:02d}_"
                    f"sample_{sample_idx + 1:02d}_"
                    f"transformed.ply"
                )

                output_path = os.path.join(
                    OUTPUT_FOLDER,
                    output_name
                )

                save_ply(transformed, output_path)

    # --------------------------------------------------------
    # 8. Merge all transformed clouds
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MERGING POINT CLOUDS")
    print("=" * 70)

    nonempty_clouds = [
        cloud for cloud in transformed_clouds if len(cloud) > 0
    ]

    if not nonempty_clouds:
        raise RuntimeError(
            "No points remain after the post-transform X/Y/Z filters. "
            "Check X_RANGE, Y_RANGE, and Z_RANGE against the transformed "
            "coordinates, or temporarily set the ranges to None."
        )

    merged_points = np.vstack(nonempty_clouds)

    print(f"Total points before voxel downsampling: {len(merged_points)}")

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(
        merged_points
    )

    if FINAL_VOXEL_SIZE > 0:
        merged_pcd = merged_pcd.voxel_down_sample(
            FINAL_VOXEL_SIZE
        )

    merged_path = os.path.join(
        OUTPUT_FOLDER,
        "final_registered_pose.ply"
    )

    o3d.io.write_point_cloud(
        merged_path,
        merged_pcd
    )

    print(
        f"Final points after voxel downsampling: "
        f"{len(merged_pcd.points)}"
    )

    print(f"Saved: {merged_path}")

    # --------------------------------------------------------
    # 8b. VISUALIZE FINAL RECONSTRUCTION
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("VISUALIZING FINAL POINT CLOUD")
    print("=" * 70)
    print("Close the Open3D visualization window to continue.")

    if SHOW_VISUALIZATION:
        o3d.visualization.draw_geometries(
            [merged_pcd],
            window_name="Pose-Based Multi-View Reconstruction",
            width=1280,
            height=800,
            point_show_normal=False
        )

    # --------------------------------------------------------
    # 9. Save transformation information
    # --------------------------------------------------------

    np.savez(
        os.path.join(
            OUTPUT_FOLDER,
            "pose_transformations.npz"
        ),
        point_cloud_files=np.array(
            all_file_names,
            dtype=object
        ),
        cloud_pose_indices=np.array(
            all_pose_indices,
            dtype=np.int32
        ),
        cloud_sample_indices=np.array(
            all_sample_indices,
            dtype=np.int32
        ),
        transforms=np.array(
            all_transforms,
            dtype=np.float64
        ),
        pose_matrices=np.array(
            [p["T_base_tool0"] for p in poses],
            dtype=np.float64
        ),
        pose_timestamps=np.array(
            [p["timestamp"] for p in poses],
            dtype=np.float64
        ),
        base_camera_matrices=np.array(
            base_camera_matrices,
            dtype=np.float64
        ),
        T_tool0_camera=T_TOOL0_CAMERA
    )

    # --------------------------------------------------------
    # 10. Write report
    # --------------------------------------------------------

    report_path = os.path.join(
        OUTPUT_FOLDER,
        "pose_registration_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("POSE-BASED MULTI-VIEW POINT CLOUD REGISTRATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("ICP: DISABLED\n")
        f.write("Input: *_filtered_pcl.npy ONLY\n")
        f.write(
            f"Timestamp tolerance: "
            f"{TIMESTAMP_TOLERANCE_SEC} seconds\n"
        )
        f.write(
            f"Reference frame: {REFERENCE_FRAME}\n"
        )
        f.write(
            f"Final voxel size: {FINAL_VOXEL_SIZE} m\n\n"
        )

        f.write("T_tool0_camera:\n")
        f.write(str(T_TOOL0_CAMERA))
        f.write("\n\n")

        for pose_idx, group in enumerate(groups):

            f.write("=" * 70 + "\n")
            f.write(
                f"POSE {pose_idx + 1}\n"
            )
            f.write("=" * 70 + "\n\n")

            f.write("Files assigned to this pose:\n")

            for sample_idx, item in enumerate(group):
                f.write(
                    f"  Sample {sample_idx + 1}: "
                    f"{item['name']}\n"
                )

            f.write("\n")

            f.write("T_base_tool0:\n")
            f.write(
                str(poses[pose_idx]["T_base_tool0"])
            )
            f.write("\n\n")

            f.write("T_base_camera:\n")
            f.write(
                str(base_camera_matrices[pose_idx])
            )
            f.write("\n\n")

            f.write("Transformation used:\n")
            f.write(
                str(reference_transforms[pose_idx])
            )
            f.write("\n\n")

    print(f"Saved: {report_path}")
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
