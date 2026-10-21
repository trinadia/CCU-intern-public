"""
python evaluation/evaluate_reconstruction.py \
    --stl "evaluation/Yacht_Benchy_stls/obj_1_Yacht_Benchy.stl" \
    --reconstruction "pose_output_d435_07-10/final_pose_registered.ply"

or

python evaluate_reconstruction.py \
    --stl Yacht_Benchy_stls/obj_1_Yacht_Benchy.stl \
    --reconstruction merged_pointcloud.npz

or

python evaluation/evaluate_reconstruction_with_export.py --stl "evaluation/Yacht_Benchy_stls/obj_1_Yacht_Benchy.stl" --reconstruction "evaluation/aligned_reconstruction.ply" --gt-scale 0.001 --rec-scale 1.0 --thresholds 0.001 0.005 0.010 --save-results "evaluation/results.txt"
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree


def load_reconstruction(path, npz_key="point_cloud"):
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".npz":
        with np.load(path) as data:
            if npz_key not in data:
                raise KeyError(
                    f"Key '{npz_key}' not found. "
                    f"Available keys: {list(data.keys())}"
                )
            points = data[npz_key]

    elif ext == ".npy":
        points = np.load(path)

    elif ext == ".ply":
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)

    else:
        raise ValueError(f"Unsupported reconstruction format: {ext}")

    points = np.asarray(points).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) == 0:
        raise ValueError("Reconstruction contains no valid points.")

    return points


def load_ground_truth(path, samples=200_000):
    mesh = trimesh.load_mesh(path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not load STL as a triangle mesh.")

    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("Ground-truth mesh is empty.")

    # Uniformly sample points over the mesh surface.
    points, _ = trimesh.sample.sample_surface(mesh, samples)

    return np.asarray(points)


def evaluate(reconstruction, ground_truth, thresholds):
    """Compute metrics, print them, and return a dictionary for export."""
    tree_gt = cKDTree(ground_truth)
    tree_rec = cKDTree(reconstruction)
    d_rec_to_gt = tree_gt.query(reconstruction, k=1)[0]
    d_gt_to_rec = tree_rec.query(ground_truth, k=1)[0]
    cd = float(np.mean(d_rec_to_gt ** 2) + np.mean(d_gt_to_rec ** 2))

    results = {
        "reconstruction_points": int(len(reconstruction)),
        "ground_truth_samples": int(len(ground_truth)),
        "chamfer_distance_squared": cd,
        "sqrt_chamfer_distance": float(np.sqrt(cd)),
        "rec_to_gt_mean": float(np.mean(d_rec_to_gt)),
        "rec_to_gt_median": float(np.median(d_rec_to_gt)),
        "rec_to_gt_p95": float(np.percentile(d_rec_to_gt, 95)),
        "gt_to_rec_mean": float(np.mean(d_gt_to_rec)),
        "gt_to_rec_median": float(np.median(d_gt_to_rec)),
        "gt_to_rec_p95": float(np.percentile(d_gt_to_rec, 95)),
    }

    print("\n===== RECONSTRUCTION EVALUATION =====")
    print(f"Reconstruction points : {len(reconstruction):,}")
    print(f"Ground-truth samples  : {len(ground_truth):,}")
    print(f"Chamfer Distance (CD) : {cd:.8f}")
    print(f"sqrt(CD)              : {np.sqrt(cd):.8f}")
    print("\n--- Reconstruction -> Ground Truth ---")
    print(f"Mean distance         : {results['rec_to_gt_mean']:.8f}")
    print(f"Median distance       : {results['rec_to_gt_median']:.8f}")
    print(f"95th percentile       : {results['rec_to_gt_p95']:.8f}")
    print("\n--- Ground Truth -> Reconstruction ---")
    print(f"Mean distance         : {results['gt_to_rec_mean']:.8f}")
    print(f"Median distance       : {results['gt_to_rec_median']:.8f}")
    print(f"95th percentile       : {results['gt_to_rec_p95']:.8f}")
    print("\n--- Surface F-score ---")

    for tau in thresholds:
        precision = float(np.mean(d_rec_to_gt < tau))
        recall = float(np.mean(d_gt_to_rec < tau))
        f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        suffix = f"{tau:g}".replace("-", "m").replace(".", "p")
        results[f"precision_at_{suffix}"] = precision
        results[f"recall_at_{suffix}"] = recall
        results[f"fscore_at_{suffix}"] = float(f_score)
        print(f"Threshold = {tau:g} | Precision = {precision:.4f} | Recall = {recall:.4f} | F-score = {f_score:.4f}")

    return results


def save_results(results, output_path, metadata):
    """CSV appends each run; TXT writes a readable report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {**metadata, **results}

    if output_path.suffix.lower() == ".csv":
        exists = output_path.exists() and output_path.stat().st_size > 0
        fields = list(row.keys())
        if exists:
            with output_path.open("r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != fields:
                raise ValueError("Existing CSV schema differs; choose a new filename.")
        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    elif output_path.suffix.lower() == ".txt":
        with output_path.open("w", encoding="utf-8") as f:
            f.write("RECONSTRUCTION EVALUATION REPORT\n")
            f.write("=" * 40 + "\n")
            for key, value in row.items():
                f.write(f"{key}: {value}\n")
    else:
        raise ValueError("--save-results must end in .csv or .txt")

    print(f"\nEvaluation results saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a reconstructed point cloud against an STL."
    )
    parser.add_argument("--stl", required=True)
    parser.add_argument("--reconstruction", required=True)
    parser.add_argument("--npz-key", default="point_cloud")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.010],
        help="F-score thresholds in the same units as the input coordinates."
    )
    parser.add_argument(
        "--gt-scale",
        type=float,
        default=1.0,
        help="Multiply STL coordinates by this factor."
    )
    parser.add_argument(
        "--rec-scale",
        type=float,
        default=1.0,
        help="Multiply reconstruction coordinates by this factor."
    )

    parser.add_argument(
        "--save-results",
        default=None,
        help="Optional .csv or .txt output path. CSV appends each run; TXT overwrites the report."
    )
    args = parser.parse_args()

    gt = load_ground_truth(args.stl, args.samples)
    rec = load_reconstruction(args.reconstruction, args.npz_key)

    gt *= args.gt_scale
    rec *= args.rec_scale

    print("Ground-truth bounds:", gt.min(axis=0), gt.max(axis=0))
    print("Reconstruction bounds:", rec.min(axis=0), rec.max(axis=0))

    results = evaluate(rec, gt, args.thresholds)
    if args.save_results:
        metadata = {
            "stl_file": str(Path(args.stl)),
            "reconstruction_file": str(Path(args.reconstruction)),
            "gt_scale": float(args.gt_scale),
            "rec_scale": float(args.rec_scale),
            "samples": int(args.samples),
            "thresholds": ";".join(str(t) for t in args.thresholds),
        }
        save_results(results, args.save_results, metadata)


if __name__ == "__main__":
    main()
