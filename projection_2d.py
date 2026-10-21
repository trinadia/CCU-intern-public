import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

# Accepts either .npz or .ply
INPUT_FILE = r"icp_output_8cm\final_registered.ply"

# If input is NPZ, should the generated PLY be saved?
SAVE_PLY = False

# Point size in the 2D projection
POINT_SIZE = 1

# Transparency
ALPHA = 1

# Keep the same scale on both axes
EQUAL_AXIS = True

# Optional distance filtering
USE_DISTANCE_FILTER = False

# Maximum distance from origin [meters] (def: 1.0 m)
MAX_DISTANCE = 0.6


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.isfile(INPUT_FILE):

    raise FileNotFoundError(
        f"File not found:\n{INPUT_FILE}"
    )


# ============================================================
# DETERMINE FILE TYPE
# ============================================================

extension = os.path.splitext(INPUT_FILE)[1].lower()

print("=" * 60)
print("POINT CLOUD LOADING")
print("=" * 60)

print(f"Input file: {INPUT_FILE}")
print(f"File type:  {extension}")


# ============================================================
# LOAD NPZ
# ============================================================

if extension == ".npz":

    print("\nLoading NPZ...")

    data = np.load(INPUT_FILE)

    print("Arrays inside NPZ:")
    print(data.files)

    # --------------------------------------------------------
    # Check for point_cloud
    # --------------------------------------------------------

    if "point_cloud" not in data:

        raise KeyError(
            "The NPZ file does not contain "
            "'point_cloud'."
        )

    # --------------------------------------------------------
    # Extract point cloud
    # --------------------------------------------------------

    points = np.asarray(
        data["point_cloud"]
    )

    print("\nPoint cloud extracted")
    print("--------------------")
    print("Shape:", points.shape)
    print("Data type:", points.dtype)

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if points.ndim != 2:

        raise ValueError(
            f"Expected point_cloud to be a 2D array, "
            f"but got shape {points.shape}"
        )

    if points.shape[1] < 3:

        raise ValueError(
            "Point cloud must contain at least "
            "X, Y, Z coordinates."
        )

    # --------------------------------------------------------
    # Extract XYZ
    # --------------------------------------------------------

    points = points[:, :3]

    # --------------------------------------------------------
    # Remove NaN / infinite points
    # --------------------------------------------------------

    valid = np.all(
        np.isfinite(points),
        axis=1
    )

    points = points[valid]

    print("Valid points:", len(points))

    # --------------------------------------------------------
    # Create Open3D point cloud
    # --------------------------------------------------------

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    # --------------------------------------------------------
    # Optionally save as PLY
    # --------------------------------------------------------

    if SAVE_PLY:

        directory = os.path.dirname(INPUT_FILE)
        filename = os.path.basename(INPUT_FILE)

        name = os.path.splitext(filename)[0]

        ply_file = os.path.join(
            directory,
            name + ".ply"
        )

        success = o3d.io.write_point_cloud(
            ply_file,
            pcd
        )

        if not success:

            raise IOError(
                f"Failed to save PLY:\n{ply_file}"
            )

        print("\nPLY created:")
        print(ply_file)


# ============================================================
# LOAD PLY
# ============================================================

elif extension == ".ply":

    print("\nLoading PLY...")

    pcd = o3d.io.read_point_cloud(
        INPUT_FILE
    )

    if pcd.is_empty():

        raise ValueError(
            "The PLY file contains no "
            "point-cloud data."
        )

    points = np.asarray(
        pcd.points
    )

    print("\nPoint cloud loaded")
    print("------------------")
    print("Number of points:", len(points))
    print("Shape:", points.shape)


# ============================================================
# INVALID FILE TYPE
# ============================================================

else:

    raise ValueError(
        f"Unsupported file type: {extension}\n"
        f"Please provide either .npz or .ply"
    )


# ============================================================
# REMOVE NaN / INFINITE POINTS
# ============================================================

valid = np.all(
    np.isfinite(points),
    axis=1
)

points = points[valid]

print("\nValid points:", len(points))


# ============================================================
# OPTIONAL DISTANCE FILTER
# ============================================================

if USE_DISTANCE_FILTER:

    distance = np.linalg.norm(
        points,
        axis=1
    )

    mask = distance <= MAX_DISTANCE

    points = points[mask]

    print(
        f"Points after distance filter "
        f"({MAX_DISTANCE} m): {len(points)}"
    )


# ============================================================
# CHECK WHETHER POINT CLOUD IS EMPTY
# ============================================================

if len(points) == 0:

    raise ValueError(
        "No points remain after filtering."
    )


# ============================================================
# SEPARATE X, Y, Z
# ============================================================

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


# ============================================================
# PRINT POINT CLOUD RANGE
# ============================================================

print("\nPoint cloud range:")
print(
    f"X: {x.min():.4f} → {x.max():.4f}"
)
print(
    f"Y: {y.min():.4f} → {y.max():.4f}"
)
print(
    f"Z: {z.min():.4f} → {z.max():.4f}"
)


# ============================================================
# CREATE 2D PROJECTIONS
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


# ============================================================
# X-Y PROJECTION
# ============================================================

axes[0].scatter(
    x,
    y,
    s=POINT_SIZE,
    alpha=ALPHA
)

axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_title("X-Y Projection")

axes[0].grid(True)

if EQUAL_AXIS:

    axes[0].set_aspect(
        "equal",
        adjustable="box"
    )


# ============================================================
# X-Z PROJECTION
# ============================================================

axes[1].scatter(
    x,
    z,
    s=POINT_SIZE,
    alpha=ALPHA
)

axes[1].set_xlabel("X")
axes[1].set_ylabel("Z")
axes[1].set_title("X-Z Projection")

axes[1].grid(True)

if EQUAL_AXIS:

    axes[1].set_aspect(
        "equal",
        adjustable="box"
    )


# ============================================================
# Y-Z PROJECTION
# ============================================================

axes[2].scatter(
    y,
    z,
    s=POINT_SIZE,
    alpha=ALPHA
)

axes[2].set_xlabel("Y")
axes[2].set_ylabel("Z")
axes[2].set_title("Y-Z Projection")

axes[2].grid(True)

if EQUAL_AXIS:

    axes[2].set_aspect(
        "equal",
        adjustable="box"
    )


# ============================================================
# DISPLAY
# ============================================================

plt.tight_layout()
plt.show()