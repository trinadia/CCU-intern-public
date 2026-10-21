#!/usr/bin/env python3
"""Align reconstructed PLY to ground-truth PLY using similarity ICP.

Estimates uniform scale, rotation, and translation, then optionally refines
rotation/translation with rigid ICP. Preserve the original source for evaluating
registration accuracy; this alignment is intended for shape comparison.

python evaluation/align_pointclouds_similarity_icp.py --target "evaluation/Yacht_Benchy_stls/obj_1_Yacht_Benchy_gt.ply" --source "2026-10-07_FIX/fix-pose_based_d435_filtered_only_2samples_2.py" --target-scale 0.001 --source-scale 1.0 --voxel 0.003 --max-distance 0.02 --visualize --output "evaluation/aligned_reconstruction_pose.ply"

python evaluation/align_pointclouds_similarity_icp.py --target "evaluation/Yacht_Benchy_stls/obj_1_Yacht_Benchy_gt.ply" --source "icp_output_8cm/final_registered.ply" --target-scale 0.001 --source-scale 1.0 --voxel 0.003 --max-distance 0.07 --visualize --output "evaluation/aligned_reconstruction_icp_2.ply"  
"""
import argparse
import copy
import os
import sys
import itertools

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_cloud(path):
    cloud = o3d.io.read_point_cloud(path)
    if cloud.is_empty():
        raise ValueError(f"Could not load points from {path}")
    p = np.asarray(cloud.points)
    p = p[np.isfinite(p).all(axis=1)]
    if not len(p):
        raise ValueError(f"No finite points in {path}")
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(p)
    return out


def scaled_copy(cloud, scale):
    out = copy.deepcopy(cloud)
    out.scale(float(scale), center=(0, 0, 0))
    return out


def downsample(cloud, voxel):
    return cloud.voxel_down_sample(voxel) if voxel > 0 else copy.deepcopy(cloud)


def pca_rotations(source, target):
    def basis(p):
        centered = p - p.mean(axis=0)
        cov = centered.T @ centered / max(len(p), 1)
        _, vec = np.linalg.eigh(cov)
        return vec[:, ::-1]

    a, b = basis(source), basis(target)
    rotations = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        r = b @ np.diag(signs) @ a.T
        if np.linalg.det(r) > 0.0:
            rotations.append(r)
    return rotations


def similarity_icp(src, tgt, initial_r, initial_scale, max_dist,
                   iterations, scale_min, scale_max, trim_fraction):
    """Trimmed nearest-neighbour similarity ICP using Umeyama updates."""
    tree = cKDTree(tgt)
    r = initial_r.copy()
    s = float(np.clip(initial_scale, scale_min, scale_max))
    t = tgt.mean(axis=0) - s * (r @ src.mean(axis=0))
    last_rmse = None

    for iteration in range(iterations):
        moved = s * (src @ r.T) + t
        d, idx = tree.query(moved, k=1, workers=-1)
        ids = np.flatnonzero(d < max_dist)
        if len(ids) < min(20, len(src)):
            ids = np.argsort(d)[:min(len(src), max(20, int(0.2 * len(src))))]
        keep = max(3, min(len(ids), int(len(ids) * trim_fraction)))
        ids = ids[np.argsort(d[ids])[:keep]]

        # Correspondence pairs use original source coordinates and current
        # nearest neighbours in the target.
        x = src[ids]
        y = tgt[idx[ids]]
        mx, my = x.mean(axis=0), y.mean(axis=0)
        x0, y0 = x - mx, y - my
        cov = (y0.T @ x0) / len(x)
        u, singular, vt = np.linalg.svd(cov)
        sign = np.ones(3)
        sign[-1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
        r_new = u @ np.diag(sign) @ vt
        var_x = np.mean(np.sum(x0 * x0, axis=1))
        if var_x <= 1e-15:
            break
        s_new = float(np.sum(singular * sign) / var_x)
        s_new = float(np.clip(s_new, scale_min, scale_max))
        t_new = my - s_new * (r_new @ mx)

        moved_new = s_new * (src @ r_new.T) + t_new
        d_new = tree.query(moved_new, k=1, workers=-1)[0]
        valid_new = d_new[d_new < max_dist]
        rmse = float(np.sqrt(np.mean(valid_new**2))) if len(valid_new) else float("inf")
        r, s, t = r_new, s_new, t_new
        if last_rmse is not None and abs(last_rmse - rmse) < 1e-7:
            break
        last_rmse = rmse

    return s, r, t, iteration + 1, last_rmse


def global_feature_init(source_cloud, target_cloud, voxel):
    """FPFH + RANSAC rigid initialization; can fail on partial/noisy overlap."""
    reg = o3d.pipelines.registration

    def prep(cloud):
        c = downsample(cloud, voxel)
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 2.5, max_nn=30))
        f = reg.compute_fpfh_feature(c, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 5.0, max_nn=100))
        return c, f

    src, sf = prep(source_cloud)
    tgt, tf = prep(target_cloud)
    result = reg.registration_ransac_based_on_feature_matching(
        src, tgt, sf, tf, True, voxel * 1.5,
        reg.TransformationEstimationPointToPoint(False), 4,
        [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        reg.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result.transformation, result.fitness, result.inlier_rmse


def main():
    parser = argparse.ArgumentParser(
        description="Similarity ICP alignment of reconstructed PLY to ground-truth PLY."
    )
    parser.add_argument("--target", required=True, help="Ground-truth PLY point cloud.")
    parser.add_argument("--source", required=True, help="Reconstructed PLY point cloud.")
    parser.add_argument("--output", default="evaluation/aligned_reconstruction.ply")
    parser.add_argument("--target-scale", type=float, default=1.0,
                        help="Known unit conversion for target coordinates.")
    parser.add_argument("--source-scale", type=float, default=1.0,
                        help="Known unit conversion for source coordinates.")
    parser.add_argument("--voxel", type=float, default=0.003,
                        help="Voxel size after unit conversion (default 0.003).")
    parser.add_argument("--max-distance", type=float, default=0.02,
                        help="Maximum ICP correspondence distance.")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--scale-min", type=float, default=0.25)
    parser.add_argument("--scale-max", type=float, default=10.0)
    parser.add_argument("--trim-fraction", type=float, default=0.85)
    parser.add_argument("--global-init", action="store_true",
                        help="Use FPFH/RANSAC for an initial rigid alignment.")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if args.scale_min <= 0 or args.scale_max < args.scale_min:
        parser.error("Require 0 < --scale-min <= --scale-max.")
    if not 0.1 <= args.trim_fraction <= 1.0:
        parser.error("--trim-fraction must be in [0.1, 1.0].")

    try:
        target_cloud = load_cloud(args.target)
        source_cloud = load_cloud(args.source)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    target_cloud = scaled_copy(target_cloud, args.target_scale)
    source_cloud = scaled_copy(source_cloud, args.source_scale)
    target_full = np.asarray(target_cloud.points)
    source_full = np.asarray(source_cloud.points)

    print("===== INPUTS (AFTER KNOWN UNIT CONVERSION) =====")
    print(f"Target points: {len(target_full):,}")
    print(f"Source points: {len(source_full):,}")
    print("Target bounds:\n", target_full.min(axis=0), "\n", target_full.max(axis=0))
    print("Source bounds:\n", source_full.min(axis=0), "\n", source_full.max(axis=0))

    target_ds = downsample(target_cloud, args.voxel)
    source_ds = downsample(source_cloud, args.voxel)
    tgt = np.asarray(target_ds.points)
    src = np.asarray(source_ds.points)
    if min(len(src), len(tgt)) < 10:
        print("ERROR: Too few points after downsampling; reduce --voxel.", file=sys.stderr)
        return 1

    # Initial scale guess based on bounding-box diagonal; can be biased by
    # incomplete reconstruction, so inspect the final scale and output.
    target_diag = np.linalg.norm(tgt.max(axis=0) - tgt.min(axis=0))
    source_diag = np.linalg.norm(src.max(axis=0) - src.min(axis=0))
    initial_scale = target_diag / source_diag
    initial_scale = float(np.clip(initial_scale, args.scale_min, args.scale_max))
    print(f"Initial bbox-diagonal scale estimate: {initial_scale:.6f}")

    rotations = pca_rotations(src, tgt)
    if args.global_init:
        try:
            transform, fitness, rmse = global_feature_init(
                source_ds, target_ds, args.voxel)
            r0 = transform[:3, :3]
            # Open3D's transform is rigid; remove numerical non-orthogonality.
            u, _, vt = np.linalg.svd(r0)
            r0 = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            rotations.insert(0, r0)
            print(f"FPFH/RANSAC initialization fitness={fitness:.5f}, RMSE={rmse:.6f}")
        except Exception as exc:
            print(f"WARNING: FPFH/RANSAC initialization failed: {exc}")

    # Cap candidate point count for runtime. Evenly spaced selection is
    # deterministic and adequate for initialization; ICP uses nearest neighbours.
    def subset(p, max_n=10000):
        if len(p) <= max_n:
            return p
        return p[np.linspace(0, len(p) - 1, max_n, dtype=int)]

    src_test, tgt_test = subset(src), subset(tgt)
    best = None
    for i, r0 in enumerate(rotations):
        s, r, t, n_iter, rmse = similarity_icp(
            src_test, tgt_test, r0, initial_scale, args.max_distance,
            min(args.iterations, 40), args.scale_min, args.scale_max,
            args.trim_fraction)
        moved = s * (src_test @ r.T) + t
        d_forward = cKDTree(tgt_test).query(moved, k=1, workers=-1)[0]
        # Symmetric score: inverse-transform target points into source frame.
        inv_target = ((tgt_test - t) @ r) / s
        d_reverse = cKDTree(src_test).query(inv_target, k=1, workers=-1)[0] * s
        score = 0.5 * (
            np.mean(np.minimum(d_forward, args.max_distance) ** 2)
            + np.mean(np.minimum(d_reverse, args.max_distance) ** 2)
        )
        if best is None or score < best[0]:
            best = (score, s, r, t, n_iter, rmse, i)

    _, scale, rotation, translation, n_iter, rmse, candidate = best
    print("\n===== SIMILARITY ICP RESULT =====")
    print(f"Selected rotation candidate: {candidate}")
    print(f"Estimated additional uniform scale: {scale:.8f}")
    print("Rotation matrix:\n", rotation)
    print("Translation vector:", translation)
    print(f"Iterations: {n_iter}; trimmed RMSE estimate: {rmse}")

    # Transform full-resolution source.
    aligned = copy.deepcopy(source_cloud)
    pts = np.asarray(aligned.points)
    pts = scale * (pts @ rotation.T) + translation
    aligned.points = o3d.utility.Vector3dVector(pts)

    # Rigid ICP refinement with scale fixed.
    reg = o3d.pipelines.registration
    aligned_ds = downsample(aligned, args.voxel)
    target_ref_ds = downsample(target_cloud, args.voxel)
    result = reg.registration_icp(
        aligned_ds, target_ref_ds, args.max_distance, np.eye(4),
        reg.TransformationEstimationPointToPoint(),
        reg.ICPConvergenceCriteria(max_iteration=args.iterations))
    correction = result.transformation
    aligned.transform(correction)
    final_rotation = correction[:3, :3] @ rotation
    final_translation = correction[:3, :3] @ translation + correction[:3, 3]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if not o3d.io.write_point_cloud(args.output, aligned):
        print(f"ERROR: Failed to save {args.output}", file=sys.stderr)
        return 1

    print("\n===== FINAL ALIGNMENT =====")
    print(f"Saved aligned point cloud: {os.path.abspath(args.output)}")
    print(f"Estimated additional scale: {scale:.8f}")
    print("Final rotation matrix:\n", final_rotation)
    print("Final translation vector:", final_translation)
    print(f"Rigid ICP fitness: {result.fitness:.6f}")
    print(f"Rigid ICP inlier RMSE: {result.inlier_rmse:.8f}")
    print("Fitness/RMSE only consider pairs within --max-distance.")

    if args.visualize:
        target_vis = copy.deepcopy(target_cloud)
        target_vis.paint_uniform_color([0.1, 0.8, 0.1])
        aligned.paint_uniform_color([0.9, 0.2, 0.2])
        o3d.visualization.draw_geometries(
            [target_vis, aligned],
            window_name="Ground truth (green) vs aligned reconstruction (red)",
            width=1280, height=800)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
