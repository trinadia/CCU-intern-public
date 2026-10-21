#!/usr/bin/env python3

"""
SAM3 + RealSense D435

重點：
1. RGB 給 SAM3 做 segmentation。
2. Depth 仍然由 RealSense D435 取得。
3. 使用 align = rs.align(rs.stream.color)，將 depth 對齊到 RGB。
4. 儲存 depth 時保留官方方式的 raw Z16 uint16：
   depth_raw = np.asanyarray(depth_frame.get_data())
   np.save(..., depth_raw)
   cv2.imwrite(..., depth_raw)  # 16-bit PNG
5. colormap / colorbar 只作為視覺化，不當成深度資料。
6. AutoSave 預設開啟，每 10 frames 存一次，不管 pcl 是否為 0 都會存。

source ~/sam3_venv/bin/activate
cd ~/sam3/scripts
python SAM3_D435.py \
  --prompts "white object,boat" \
  --width 848 \
  --height 480 \
  --fps 30 \
  --max_display_depth 0.5 \
  --max_depth 1.0
  
"""

import os
import time
import argparse
import re
from datetime import datetime

import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image as PILImage


# ============================================================
# Utility
# ============================================================

def sanitize_filename(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)

    if text == "":
        text = "prompt"

    return text


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompts",
        type=str,
        default="white object,boat",
        help="SAM3 prompts separated by comma"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./D435_sam3_aligned_output",
        help="Directory for saving RGB, raw depth, visualization, mask, overlay, projection, and npy"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="RealSense color/depth width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="RealSense color/depth height"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="RealSense FPS"
    )

    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=30,
        help="Number of warmup frames before processing"
    )

    parser.add_argument(
        "--frame_timeout_ms",
        type=int,
        default=10000,
        help="RealSense wait_for_frames timeout in ms"
    )

    parser.add_argument(
        "--run_sam_every_n_frames",
        type=int,
        default=30,
        help="Run SAM3 every N frames"
    )

    parser.add_argument(
        "--min_score",
        type=float,
        default=0.05,
        help="Minimum SAM3 score"
    )

    parser.add_argument(
        "--min_mask_pixels",
        type=int,
        default=20,
        help="Minimum mask pixels"
    )

    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.05,
        help="Minimum valid depth in meters for point cloud"
    )

    parser.add_argument(
        "--max_depth",
        type=float,
        default=2.0,
        help="Maximum valid depth in meters for point cloud"
    )

    parser.add_argument(
        "--min_display_depth",
        type=float,
        default=0.0,
        help="Minimum depth shown in depth colormap, unit meter"
    )

    parser.add_argument(
        "--max_display_depth",
        type=float,
        default=2.0,
        help="Maximum depth shown in depth colormap, unit meter"
    )

    parser.add_argument(
        "--point_read_stride",
        type=int,
        default=1,
        help="Use every N mask pixels for point cloud"
    )

    parser.add_argument(
        "--max_display_points",
        type=int,
        default=80000,
        help="Maximum projected points drawn on RGB image"
    )

    parser.add_argument(
        "--point_radius",
        type=int,
        default=1,
        help="Projected point radius"
    )

    parser.add_argument(
        "--display_scale",
        type=float,
        default=1.0,
        help="Display image scale"
    )

    parser.add_argument(
        "--auto_save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable auto save. Default is True. Use --no-auto_save to disable."
    )

    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=10,
        help="Auto save every N frames. Default is 10."
    )

    parser.add_argument(
        "--verify_depth_png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After saving raw depth PNG, read it back with IMREAD_UNCHANGED to verify uint16."
    )

    parser.add_argument(
        "--configure_depth_sensor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Configure D435 depth sensor. Disabled by default to match test.py raw mode."
    )

    parser.add_argument(
        "--depth_visual_preset",
        type=int,
        default=-1,
        help="D400 visual_preset value. 4 is commonly High Density. Use -1 to skip."
    )

    parser.add_argument(
        "--emitter_enabled",
        type=int,
        default=-1,
        help="Set emitter_enabled. Use -1 to skip."
    )

    parser.add_argument(
        "--laser_power",
        type=float,
        default=-1.0,
        help="Set laser_power. Negative means use the sensor maximum. Use --no-configure_depth_sensor to skip."
    )

    parser.add_argument(
        "--depth_auto_exposure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable depth auto exposure when supported."
    )

    parser.add_argument(
        "--use_depth_filters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated. Depth image post-processing is always disabled."
    )

    parser.add_argument(
        "--use_filtered_depth_for_pcl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated. PCL always uses raw aligned depth."
    )

    parser.add_argument(
        "--save_filtered_depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated. Filtered depth is never generated."
    )

    parser.add_argument(
        "--spatial_filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated. Spatial filtering is always disabled."
    )

    parser.add_argument(
        "--temporal_filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated. Temporal filtering is always disabled."
    )

    parser.add_argument(
        "--hole_filling_mode",
        type=int,
        default=-1,
        help="Deprecated. Hole filling is always disabled."
    )

    parser.add_argument(
        "--pcl_robust_depth_filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated. Robust PCL depth filtering is always disabled."
    )

    parser.add_argument(
        "--pcl_depth_mad_multiplier",
        type=float,
        default=3.0,
        help="MAD multiplier for robust PCL depth filtering."
    )

    parser.add_argument(
        "--pcl_depth_min_window",
        type=float,
        default=0.08,
        help="Minimum half-window in meters around median mask depth."
    )

    parser.add_argument(
        "--pcl_depth_max_window",
        type=float,
        default=0.35,
        help="Maximum half-window in meters around median mask depth."
    )

    return parser.parse_args()


# ============================================================
# Colorbar / Depth Visualization
# ============================================================

def make_depth_colorbar(
    height,
    width=70,
    text_width=130,
    min_depth=0.0,
    max_depth=2.0,
    colormap=cv2.COLORMAP_JET
):
    gradient = np.linspace(
        255,
        0,
        height,
        dtype=np.uint8
    ).reshape(height, 1)

    gradient = np.repeat(
        gradient,
        width,
        axis=1
    )

    colorbar = cv2.applyColorMap(
        gradient,
        colormap
    )

    canvas = np.ones(
        (height, width + text_width, 3),
        dtype=np.uint8
    ) * 255

    canvas[:, :width] = colorbar

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    mid_depth = (min_depth + max_depth) / 2.0

    cv2.putText(
        canvas,
        f"{max_depth:.2f} m",
        (width + 8, 32),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        f"{mid_depth:.2f} m",
        (width + 8, height // 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        f"{min_depth:.2f} m",
        (width + 8, height - 20),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "Depth",
        (width + 8, height // 2 + 45),
        font,
        0.65,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    return canvas


def add_colorbar_to_depth_image(
    depth_colormap,
    min_depth=0.0,
    max_depth=2.0
):
    h = depth_colormap.shape[0]

    colorbar = make_depth_colorbar(
        height=h,
        width=70,
        text_width=130,
        min_depth=min_depth,
        max_depth=max_depth,
        colormap=cv2.COLORMAP_JET
    )

    depth_with_bar = np.hstack(
        [
            depth_colormap,
            colorbar
        ]
    )

    return depth_with_bar


def make_depth_colormap_from_raw(
    depth_raw,
    depth_scale,
    min_display_depth,
    max_display_depth
):
    """
    只做顯示用。

    depth_raw 本身不會被修改。
    真正要保存深度資料請存 depth_raw uint16，不要存這裡產生的 colormap。
    """

    if max_display_depth <= min_display_depth:
        raise ValueError("max_display_depth must be larger than min_display_depth")

    depth_m = depth_raw.astype(np.float32) * depth_scale

    raw_valid_mask = np.isfinite(depth_m)
    raw_valid_mask &= depth_raw > 0

    display_range_mask = raw_valid_mask.copy()
    display_range_mask &= depth_m >= min_display_depth
    display_range_mask &= depth_m <= max_display_depth

    depth_norm = np.zeros_like(depth_m, dtype=np.float32)

    depth_norm[display_range_mask] = (
        (depth_m[display_range_mask] - min_display_depth)
        / (max_display_depth - min_display_depth)
    )

    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(
        depth_uint8,
        cv2.COLORMAP_JET
    )

    depth_colormap[~display_range_mask] = (0, 0, 0)

    depth_colormap_with_bar = add_colorbar_to_depth_image(
        depth_colormap=depth_colormap,
        min_depth=min_display_depth,
        max_depth=max_display_depth
    )

    return depth_colormap, depth_colormap_with_bar, depth_m, display_range_mask


# ============================================================
# SAM3 Image Segmenter
# ============================================================

class SAM3ImageSegmenter:
    def __init__(self, prompts):
        self.prompts = prompts

        print("[SAM3] Loading SAM3 image model...")

        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = build_sam3_image_model()
        self.model = self.model.to(device=self.device)
        self.model.eval()

        self.processor = Sam3Processor(self.model)

        print("[SAM3] device:", self.device)
        print("[SAM3] prompts:", self.prompts)
        print("[SAM3] model loaded.")

    def _to_numpy_float32(self, x):
        if hasattr(x, "detach"):
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def predict_one_prompt(self, image_pil, prompt):
        with self.torch.inference_mode():
            if self.device == "cuda":
                with self.torch.autocast(
                    device_type="cuda",
                    dtype=self.torch.bfloat16
                ):
                    state = self.processor.set_image(image_pil)
                    output = self.processor.set_text_prompt(
                        state=state,
                        prompt=prompt
                    )
            else:
                state = self.processor.set_image(image_pil)
                output = self.processor.set_text_prompt(
                    state=state,
                    prompt=prompt
                )

        masks = output["masks"]
        scores = output["scores"]

        if len(masks) == 0:
            h = image_pil.height
            w = image_pil.width
            return np.zeros((h, w), dtype=bool), 0.0

        scores_np = self._to_numpy_float32(scores)
        best_idx = int(np.argmax(scores_np))
        score = float(scores_np[best_idx])

        mask_np = self._to_numpy_float32(masks[best_idx])
        mask_np = np.squeeze(mask_np)

        mask_bool = mask_np > 0.5

        return mask_bool, score

    def predict_best_mask(self, rgb_bgr):
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(rgb)

        best_mask = None
        best_score = -1.0
        best_prompt = ""

        for prompt in self.prompts:
            mask, score = self.predict_one_prompt(image_pil, prompt)

            print(
                f"[SAM3 debug] prompt='{prompt}' "
                f"score={score:.4f} "
                f"mask_pixels={int(mask.sum())}"
            )

            if score > best_score:
                best_score = score
                best_mask = mask
                best_prompt = prompt

        if best_mask is None:
            h, w = rgb_bgr.shape[:2]
            best_mask = np.zeros((h, w), dtype=bool)
            best_score = 0.0
            best_prompt = ""

        print(
            f"[SAM3 debug] best_prompt='{best_prompt}' "
            f"best_score={best_score:.4f} "
            f"best_mask_pixels={int(best_mask.sum())}"
        )

        return best_mask, best_score, best_prompt


# ============================================================
# Display
# ============================================================

def resize_for_display(img, scale):
    if scale == 1.0:
        return img

    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def draw_text_box(img, lines, x=20, y=45):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    line_height = 36
    padding = 10

    if not isinstance(lines, (list, tuple)):
        lines = [str(lines)]

    max_w = 0
    for line in lines:
        (text_w, _), _ = cv2.getTextSize(
            str(line),
            font,
            font_scale,
            thickness
        )
        max_w = max(max_w, text_w)

    box_w = max_w + padding * 2
    box_h = line_height * len(lines) + padding * 2

    x0 = max(0, x - padding)
    y0 = max(0, y - 30)
    x1 = min(img.shape[1] - 1, x0 + box_w)
    y1 = min(img.shape[0] - 1, y0 + box_h)

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x1, y1),
        (0, 0, 0),
        -1
    )

    alpha = 0.65
    img[:] = cv2.addWeighted(
        overlay,
        alpha,
        img,
        1.0 - alpha,
        0
    )

    for i, line in enumerate(lines):
        cv2.putText(
            img,
            str(line),
            (x, y + i * line_height),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA
        )

    return img


def make_overlay(rgb_bgr, mask):
    overlay = rgb_bgr.copy()

    if mask is None:
        return overlay

    if mask.shape[:2] != rgb_bgr.shape[:2]:
        return overlay

    mask = mask.astype(bool)

    if int(mask.sum()) == 0:
        return overlay

    green_layer = np.zeros_like(overlay)
    green_layer[:, :, 1] = 255

    alpha = 0.45

    overlay[mask] = cv2.addWeighted(
        overlay[mask],
        1.0 - alpha,
        green_layer[mask],
        alpha,
        0
    )

    return overlay


# ============================================================
# Depth + Mask to Point Cloud
# ============================================================

def mask_depth_to_camera_points(
    mask,
    depth_m,
    intrinsics,
    min_depth,
    max_depth,
    point_read_stride,
    robust_depth_filter=True,
    depth_mad_multiplier=3.0,
    depth_min_window=0.08,
    depth_max_window=0.35
):
    if mask is None or depth_m is None:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    h, w = depth_m.shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    mask = mask.astype(bool)

    v_all, u_all = np.where(mask)

    if len(u_all) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    if point_read_stride > 1:
        u_all = u_all[::point_read_stride]
        v_all = v_all[::point_read_stride]

    Z = depth_m[v_all, u_all].astype(np.float32)

    valid = (
        np.isfinite(Z) &
        (Z > min_depth) &
        (Z < max_depth)
    )

    u = u_all[valid].astype(np.int32)
    v = v_all[valid].astype(np.int32)
    Z = Z[valid]

    if len(Z) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    if robust_depth_filter and len(Z) >= 20:
        median_z = float(np.median(Z))
        mad_z = float(np.median(np.abs(Z - median_z)))

        half_window = max(
            float(depth_min_window),
            mad_z * float(depth_mad_multiplier)
        )
        half_window = min(
            float(depth_max_window),
            half_window
        )

        robust_valid = np.abs(Z - median_z) <= half_window

        if int(np.sum(robust_valid)) >= 10:
            u = u[robust_valid]
            v = v[robust_valid]
            Z = Z[robust_valid]

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.ppx)
    cy = float(intrinsics.ppy)

    X = (u.astype(np.float32) - cx) * Z / fx
    Y = (v.astype(np.float32) - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=1).astype(np.float32)

    return points, u, v


def draw_projected_points(rgb_bgr, u, v, max_display_points, point_radius):
    projection_img = rgb_bgr.copy()

    n = len(u)

    if n == 0:
        return projection_img, 0

    if n > max_display_points:
        sample_idx = np.linspace(
            0,
            n - 1,
            max_display_points
        ).astype(np.int32)

        u_draw = u[sample_idx]
        v_draw = v[sample_idx]
    else:
        u_draw = u
        v_draw = v

    for uu, vv in zip(u_draw, v_draw):
        cv2.circle(
            projection_img,
            (int(uu), int(vv)),
            point_radius,
            (0, 255, 0),
            -1
        )

    return projection_img, len(u_draw)


# ============================================================
# RealSense helpers
# ============================================================

def print_depth_sensor_options(depth_sensor):
    print("========== Depth Sensor Options ==========")

    sensor_options_to_check = [
        ("visual_preset", rs.option.visual_preset),
        ("emitter_enabled", rs.option.emitter_enabled),
        ("laser_power", rs.option.laser_power),
        ("exposure", rs.option.exposure),
        ("gain", rs.option.gain),
        ("enable_auto_exposure", rs.option.enable_auto_exposure),
    ]

    for name, option in sensor_options_to_check:
        try:
            value = depth_sensor.get_option(option)
            print(f"[Sensor] {name}: {value}")
        except Exception as e:
            print(f"[Sensor] {name}: not available ({e})")


def set_sensor_option_if_supported(sensor, option, value, name):
    try:
        if not sensor.supports(option):
            print(f"[Sensor config] {name}: not supported")
            return False

        sensor.set_option(option, float(value))
        actual = sensor.get_option(option)
        print(f"[Sensor config] {name}: set to {actual}")
        return True
    except Exception as e:
        print(f"[Sensor config] {name}: failed ({e})")
        return False


def get_sensor_option_max(sensor, option):
    option_range = sensor.get_option_range(option)
    return float(option_range.max)


def configure_depth_sensor_for_density(depth_sensor, args):
    if not args.configure_depth_sensor:
        print("[RealSense] sensor options are NOT modified")
        return

    print("[RealSense] configuring depth sensor for denser real-camera depth")

    if args.depth_visual_preset >= 0:
        set_sensor_option_if_supported(
            depth_sensor,
            rs.option.visual_preset,
            args.depth_visual_preset,
            "visual_preset"
        )

    if args.depth_auto_exposure:
        set_sensor_option_if_supported(
            depth_sensor,
            rs.option.enable_auto_exposure,
            1,
            "enable_auto_exposure"
        )

    if args.emitter_enabled >= 0:
        set_sensor_option_if_supported(
            depth_sensor,
            rs.option.emitter_enabled,
            args.emitter_enabled,
            "emitter_enabled"
        )

    if args.laser_power < 0:
        try:
            laser_power = get_sensor_option_max(
                depth_sensor,
                rs.option.laser_power
            )
        except Exception as e:
            print(f"[Sensor config] laser_power: cannot read max ({e})")
            laser_power = None
    else:
        laser_power = args.laser_power

    if laser_power is not None:
        set_sensor_option_if_supported(
            depth_sensor,
            rs.option.laser_power,
            laser_power,
            "laser_power"
        )


def set_filter_option_if_supported(depth_filter, option, value, name):
    try:
        if hasattr(depth_filter, "supports") and not depth_filter.supports(option):
            return False

        depth_filter.set_option(option, float(value))
        return True
    except Exception:
        return False


def create_depth_filters(args):
    print("[Depth filters] disabled: raw aligned depth only")
    return []


def apply_depth_filters(depth_frame, filters):
    filtered_frame = depth_frame

    for name, depth_filter in filters:
        try:
            filtered_frame = depth_filter.process(filtered_frame)
        except Exception as e:
            print(f"[Depth filters] {name} failed: {e}")
            return depth_frame

    return filtered_frame


def depth_frame_to_raw_uint16(depth_frame):
    depth_raw = np.asanyarray(
        depth_frame.get_data()
    )

    if depth_raw.dtype != np.uint16:
        raise TypeError(
            f"RealSense Z16 depth must be uint16, got {depth_raw.dtype}"
        )

    return depth_raw


def imwrite_or_raise(path, image):
    ok = cv2.imwrite(path, image)

    if not ok:
        raise IOError(f"Failed to write image: {path}")


# ============================================================
# Save
# ============================================================

def save_current_result(
    save_dir,
    save_index,
    prompt_name,
    rgb_bgr,
    depth_raw_aligned,
    depth_filtered_aligned,
    depth_scale,
    depth_colormap,
    depth_colormap_with_bar,
    mask_img,
    overlay_img,
    pcl_projection_img,
    points,
    depth_for_pcl_source,
    verify_depth_png=True,
):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prefix = (
        f"{prompt_name}_"
        f"{save_index:04d}_"
        f"{timestamp}_"
        f"points_{len(points):06d}"
    )

    rgb_path = os.path.join(save_dir, prefix + "_rgb.png")

    # 真正深度資料：raw Z16 uint16
    depth_raw_npy_path = os.path.join(save_dir, prefix + "_depth_raw_aligned_uint16.npy")
    depth_raw_png_path = os.path.join(save_dir, prefix + "_depth_raw_aligned_uint16.png")

    # 輔助資料：meter 版 float32，方便之後直接讀公尺
    depth_meter_npy_path = os.path.join(save_dir, prefix + "_depth_meter_aligned_float32.npy")
    depth_filtered_npy_path = os.path.join(save_dir, prefix + "_depth_filtered_aligned_uint16.npy")
    depth_filtered_png_path = os.path.join(save_dir, prefix + "_depth_filtered_aligned_uint16.png")
    depth_filtered_meter_npy_path = os.path.join(save_dir, prefix + "_depth_filtered_meter_aligned_float32.npy")

    # 顯示用圖片：不能當成深度資料
    depth_vis_path = os.path.join(save_dir, prefix + "_depth_visual_colormap.png")
    depth_vis_bar_path = os.path.join(save_dir, prefix + "_depth_visual_colormap_colorbar.png")

    mask_path = os.path.join(save_dir, prefix + "_mask.png")
    overlay_path = os.path.join(save_dir, prefix + "_overlay.png")
    projection_path = os.path.join(save_dir, prefix + "_depth_pcl_projection.png")
    pcl_npy_path = os.path.join(save_dir, prefix + "_filtered_pcl.npy")

    imwrite_or_raise(rgb_path, rgb_bgr)

    # ========================================================
    # Save raw depth exactly like librealsense common practice
    # ========================================================

    depth_raw_to_save = depth_raw_aligned.astype(np.uint16, copy=False)

    np.save(
        depth_raw_npy_path,
        depth_raw_to_save
    )

    imwrite_or_raise(
        depth_raw_png_path,
        depth_raw_to_save
    )

    # meter float32 version for easier later use
    depth_meter = depth_raw_to_save.astype(np.float32) * float(depth_scale)

    np.save(
        depth_meter_npy_path,
        depth_meter.astype(np.float32)
    )

    saved_filtered_depth = depth_filtered_aligned is not None

    if saved_filtered_depth:
        depth_filtered_to_save = depth_filtered_aligned.astype(np.uint16, copy=False)

        np.save(
            depth_filtered_npy_path,
            depth_filtered_to_save
        )

        imwrite_or_raise(
            depth_filtered_png_path,
            depth_filtered_to_save
        )

        depth_filtered_meter = (
            depth_filtered_to_save.astype(np.float32)
            * float(depth_scale)
        )

        np.save(
            depth_filtered_meter_npy_path,
            depth_filtered_meter.astype(np.float32)
        )

    # ========================================================
    # Save visualizations only for checking
    # ========================================================

    imwrite_or_raise(depth_vis_path, depth_colormap)
    imwrite_or_raise(depth_vis_bar_path, depth_colormap_with_bar)

    imwrite_or_raise(mask_path, mask_img)
    imwrite_or_raise(overlay_path, overlay_img)
    imwrite_or_raise(projection_path, pcl_projection_img)

    np.save(
        pcl_npy_path,
        points.astype(np.float32)
    )

    print("========== Saved ==========")
    print("[Save] RGB:", rgb_path)
    print("[Save] RAW aligned depth NPY uint16:", depth_raw_npy_path)
    print("[Save] RAW aligned depth PNG uint16:", depth_raw_png_path)
    print("[Save] aligned depth meter NPY float32:", depth_meter_npy_path)

    if saved_filtered_depth:
        print("[Save] FILTERED aligned depth NPY uint16:", depth_filtered_npy_path)
        print("[Save] FILTERED aligned depth PNG uint16:", depth_filtered_png_path)
        print("[Save] FILTERED depth meter NPY float32:", depth_filtered_meter_npy_path)

    print("[Save] depth visualization colormap:", depth_vis_path)
    print("[Save] depth visualization colormap with colorbar:", depth_vis_bar_path)
    print("[Save] Mask:", mask_path)
    print("[Save] Overlay:", overlay_path)
    print("[Save] Depth PCL Projection:", projection_path)
    print("[Save] Filtered PCL npy:", pcl_npy_path)
    print("[Save] Filtered PCL shape:", points.shape)
    print("[Save] PCL columns: X Y Z")
    print("[Save] PCL depth source:", depth_for_pcl_source)
    print("[Save] depth_raw dtype:", depth_raw_to_save.dtype)
    print("[Save] depth_raw shape:", depth_raw_to_save.shape)
    print("[Save] depth_raw min:", int(depth_raw_to_save.min()))
    print("[Save] depth_raw max:", int(depth_raw_to_save.max()))
    print("[Save] depth_scale:", float(depth_scale))
    print("[Save] depth_meter = depth_raw * depth_scale")

    if verify_depth_png:
        check = cv2.imread(
            depth_raw_png_path,
            cv2.IMREAD_UNCHANGED
        )

        if check is None:
            print("[Check PNG] failed to read saved png")
        elif check.dtype != np.uint16:
            raise TypeError(
                f"Saved raw depth PNG must be uint16, got {check.dtype}"
            )
        else:
            print(
                "[Check PNG]",
                "raw",
                "dtype=", check.dtype,
                "shape=", check.shape,
                "min=", int(check.min()),
                "max=", int(check.max())
            )

        if saved_filtered_depth:
            filtered_check = cv2.imread(
                depth_filtered_png_path,
                cv2.IMREAD_UNCHANGED
            )

            if filtered_check is None:
                print("[Check PNG] failed to read saved filtered png")
            elif filtered_check.dtype != np.uint16:
                raise TypeError(
                    f"Saved filtered depth PNG must be uint16, got {filtered_check.dtype}"
                )
            else:
                print(
                    "[Check PNG]",
                    "filtered",
                    "dtype=", filtered_check.dtype,
                    "shape=", filtered_check.shape,
                    "min=", int(filtered_check.min()),
                    "max=", int(filtered_check.max())
                )

    return save_index + 1


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    prompts = [
        p.strip()
        for p in args.prompts.split(",")
        if p.strip()
    ]

    prompt_name = sanitize_filename(
        "_".join(prompts)
    )

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    print("========== Settings ==========")
    print("source: RealSense D435")
    print("depth source: RealSense aligned raw Z16 uint16")
    print("width:", args.width)
    print("height:", args.height)
    print("fps:", args.fps)
    print("prompts:", prompts)
    print("prompt_name:", prompt_name)
    print("save_dir:", args.save_dir)
    print("warmup_frames:", args.warmup_frames)
    print("frame_timeout_ms:", args.frame_timeout_ms)
    print("run_sam_every_n_frames:", args.run_sam_every_n_frames)
    print("min_score:", args.min_score)
    print("min_mask_pixels:", args.min_mask_pixels)
    print("min_depth:", args.min_depth)
    print("max_depth:", args.max_depth)
    print("min_display_depth:", args.min_display_depth)
    print("max_display_depth:", args.max_display_depth)
    print("point_read_stride:", args.point_read_stride)
    print("auto_save:", args.auto_save)
    print("save_every_n_frames:", args.save_every_n_frames)
    print("verify_depth_png:", args.verify_depth_png)
    print("configure_depth_sensor:", args.configure_depth_sensor)
    print("depth_visual_preset:", args.depth_visual_preset)
    print("emitter_enabled:", args.emitter_enabled)
    print("laser_power:", args.laser_power)
    print("depth_auto_exposure:", args.depth_auto_exposure)
    print("depth_post_processing: disabled")
    print("depth_for_pcl: raw aligned depth")
    print("save_filtered_depth: disabled")
    print("spatial_filter: disabled")
    print("temporal_filter: disabled")
    print("hole_filling: disabled")
    print("pcl_robust_depth_filter: disabled")
    print("pcl_depth_mad_multiplier:", args.pcl_depth_mad_multiplier)
    print("pcl_depth_min_window:", args.pcl_depth_min_window)
    print("pcl_depth_max_window:", args.pcl_depth_max_window)

    # ========================================================
    # Load SAM3
    # ========================================================

    segmenter = SAM3ImageSegmenter(prompts)

    # ========================================================
    # RealSense pipeline
    # ========================================================

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps
    )

    config.enable_stream(
        rs.stream.depth,
        args.width,
        args.height,
        rs.format.z16,
        args.fps
    )

    print("[RealSense] Starting pipeline...")
    profile = pipeline.start(config)

    # ========================================================
    # Align depth to color
    # ========================================================

    align = rs.align(rs.stream.color)

    print("[RealSense] align enabled: depth -> color")

    # ========================================================
    # Depth sensor
    # ========================================================

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print("[RealSense] depth scale:", depth_scale)
    print("[RealSense] depth_raw is uint16 Z16")
    print("[RealSense] depth_meter = depth_raw * depth_scale")
    print("[RealSense] saved depth is aligned to RGB frame")

    print("[RealSense] sensor options before config:")
    print_depth_sensor_options(depth_sensor)
    configure_depth_sensor_for_density(depth_sensor, args)
    print("[RealSense] sensor options after config:")
    print_depth_sensor_options(depth_sensor)

    # ========================================================
    # Color intrinsics
    # ========================================================

    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    print("[RealSense] color intrinsics:")
    print("  width:", color_intrinsics.width)
    print("  height:", color_intrinsics.height)
    print("  fx:", color_intrinsics.fx)
    print("  fy:", color_intrinsics.fy)
    print("  cx:", color_intrinsics.ppx)
    print("  cy:", color_intrinsics.ppy)

    depth_filters = create_depth_filters(args)

    # ========================================================
    # Windows
    # ========================================================

    cv2.namedWindow("D435 RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("D435 Aligned Depth Visualization", cv2.WINDOW_NORMAL)
    cv2.namedWindow("SAM3 Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("SAM3 Overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("SAM3 Depth PCL Projection", cv2.WINDOW_NORMAL)

    frame_count = 0
    save_index = 0

    last_mask = None
    last_score = 0.0
    last_prompt = ""

    print("========== Warmup ==========")

    for i in range(args.warmup_frames):
        try:
            frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if depth_frame:
                _ = depth_frame_to_raw_uint16(depth_frame)

                if depth_filters:
                    filtered_depth_frame = apply_depth_filters(
                        depth_frame,
                        depth_filters
                    )
                    _ = depth_frame_to_raw_uint16(filtered_depth_frame)

            if not depth_frame or not color_frame:
                print("[Warmup warning] No color or depth frame")

            print(f"[Warmup] {i + 1}/{args.warmup_frames}")
        except Exception as e:
            print("[Warmup warning]", e)

    print("[RealSense] Warmup finished")

    print("========== Controls ==========")
    print("AutoSave is ON by default.")
    print("It saves once every 10 frames by default.")
    print("Saved raw depth is uint16 NPY and uint16 PNG.")
    print("Depth colormap is visualization only.")
    print("Press S to save current frame manually.")
    print("Press Q or ESC to quit.")

    try:
        while True:
            t0 = time.perf_counter()
            frame_count += 1

            # ====================================================
            # Get aligned frames
            # ====================================================

            try:
                frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            except Exception as e:
                print("[Warning] wait_for_frames failed:", e)
                continue

            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("[Warning] No color or depth frame")
                continue

            rgb_bgr = np.asanyarray(
                color_frame.get_data()
            )

            # ====================================================
            # Official-style raw depth extraction
            # ====================================================

            try:
                depth_raw_aligned = depth_frame_to_raw_uint16(depth_frame)
            except Exception as e:
                print("[Warning] invalid depth frame:", e)
                continue

            h, w = rgb_bgr.shape[:2]

            if depth_raw_aligned.shape[:2] != (h, w):
                print(
                    "[Warning] aligned depth shape does not match RGB:",
                    depth_raw_aligned.shape,
                    rgb_bgr.shape
                )
                continue

            depth_filtered_aligned = None

            if depth_filters:
                filtered_depth_frame = apply_depth_filters(
                    depth_frame,
                    depth_filters
                )

                try:
                    depth_filtered_aligned = depth_frame_to_raw_uint16(
                        filtered_depth_frame
                    )
                except Exception as e:
                    print("[Warning] invalid filtered depth frame:", e)
                    depth_filtered_aligned = None

                if (
                    depth_filtered_aligned is not None and
                    depth_filtered_aligned.shape[:2] != (h, w)
                ):
                    print(
                        "[Warning] filtered depth shape does not match RGB:",
                        depth_filtered_aligned.shape,
                        rgb_bgr.shape
                    )
                    depth_filtered_aligned = None

            use_filtered_depth = (
                args.use_filtered_depth_for_pcl and
                depth_filtered_aligned is not None
            )

            if use_filtered_depth:
                depth_for_work_raw = depth_filtered_aligned
                depth_for_work_source = "filtered"
            else:
                depth_for_work_raw = depth_raw_aligned
                depth_for_work_source = "raw"

            # ====================================================
            # Depth visualization + aligned depth in meters
            # ====================================================

            (
                depth_colormap,
                depth_colormap_with_bar,
                depth_m,
                valid_display_mask
            ) = make_depth_colormap_from_raw(
                depth_raw=depth_for_work_raw,
                depth_scale=depth_scale,
                min_display_depth=args.min_display_depth,
                max_display_depth=args.max_display_depth
            )

            if depth_m.shape[:2] != (h, w):
                depth_m = cv2.resize(
                    depth_m,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            # ====================================================
            # SAM3 inference
            # ====================================================

            run_sam = (
                last_mask is None or
                frame_count % args.run_sam_every_n_frames == 0
            )

            t_sam0 = time.perf_counter()

            if run_sam:
                try:
                    mask, score, used_prompt = segmenter.predict_best_mask(rgb_bgr)
                except Exception as e:
                    print("[ERROR] SAM3 inference failed:", e)
                    continue

                last_mask = mask
                last_score = score
                last_prompt = used_prompt

            else:
                mask = last_mask
                score = last_score
                used_prompt = last_prompt

            t_sam1 = time.perf_counter()

            if mask is None or not isinstance(mask, np.ndarray):
                mask = np.zeros((h, w), dtype=bool)

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            mask = mask.astype(bool)
            mask_pixels = int(mask.sum())

            if score < args.min_score:
                print(
                    f"[WARN] SAM3 score too low: score={score:.3f}. Use empty mask."
                )

                mask = np.zeros((h, w), dtype=bool)
                mask_pixels = 0

            if mask_pixels < args.min_mask_pixels:
                print(
                    f"[WARN] Mask too small: mask_pixels={mask_pixels}."
                )

            # ====================================================
            # SAM3 mask / overlay
            # ====================================================

            mask_img = (mask.astype(np.uint8) * 255)

            overlay_img = make_overlay(
                rgb_bgr,
                mask
            )

            # ====================================================
            # Mask + aligned RealSense depth to PCL
            # ====================================================

            t_pcl0 = time.perf_counter()

            points, cube_u, cube_v = mask_depth_to_camera_points(
                mask=mask,
                depth_m=depth_m,
                intrinsics=color_intrinsics,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                point_read_stride=max(1, args.point_read_stride),
                robust_depth_filter=False,
                depth_mad_multiplier=args.pcl_depth_mad_multiplier,
                depth_min_window=args.pcl_depth_min_window,
                depth_max_window=args.pcl_depth_max_window
            )

            t_pcl1 = time.perf_counter()

            cube_point_count = len(points)

            pcl_projection_img, displayed_points = draw_projected_points(
                rgb_bgr=rgb_bgr,
                u=cube_u,
                v=cube_v,
                max_display_points=args.max_display_points,
                point_radius=max(1, args.point_radius)
            )

            # ====================================================
            # Info text
            # ====================================================

            if cube_point_count > 0:
                center = points.mean(axis=0)
                bbox_size = points.max(axis=0) - points.min(axis=0)

                pcl_lines = [
                    f"pcl: {cube_point_count} | display: {displayed_points}",
                    f"depth: {depth_for_work_source}",
                    f"center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]",
                    f"bbox: [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]",
                ]
            else:
                center = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                bbox_size = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

                pcl_lines = [
                    "pcl: 0",
                    f"depth: {depth_for_work_source}",
                    "No valid SAM3 mask or depth inside mask",
                ]

            overlay_lines = [
                f"prompt: {used_prompt} | score: {score:.3f} | run_sam: {run_sam}",
                f"mask: {mask_pixels} | pcl: {cube_point_count} | depth: {depth_for_work_source}",
            ]

            overlay_img = draw_text_box(
                overlay_img,
                overlay_lines,
                x=20,
                y=45
            )

            pcl_projection_img = draw_text_box(
                pcl_projection_img,
                pcl_lines,
                x=20,
                y=45
            )

            # ====================================================
            # Depth statistics
            # ====================================================

            raw_depth_m = depth_raw_aligned.astype(np.float32) * depth_scale

            raw_valid_mask = (
                np.isfinite(raw_depth_m) &
                (depth_raw_aligned > 0) &
                (raw_depth_m >= args.min_depth) &
                (raw_depth_m <= args.max_depth)
            )

            work_valid_mask = (
                np.isfinite(depth_m) &
                (depth_for_work_raw > 0) &
                (depth_m >= args.min_depth) &
                (depth_m <= args.max_depth)
            )

            total_count = depth_raw_aligned.size

            raw_valid_count = int(np.sum(raw_valid_mask))
            raw_zero_count = int(np.sum(depth_raw_aligned == 0))
            work_valid_count = int(np.sum(work_valid_mask))
            work_zero_count = int(np.sum(depth_for_work_raw == 0))
            raw_invalid_count = total_count - raw_valid_count
            work_invalid_count = total_count - work_valid_count

            in_display_count = int(np.sum(valid_display_mask))

            if work_valid_count > 0:
                valid_depth_m = depth_m[work_valid_mask]
                d_min = float(np.min(valid_depth_m))
                d_max = float(np.max(valid_depth_m))
                d_mean = float(np.mean(valid_depth_m))
            else:
                d_min = 0.0
                d_max = 0.0
                d_mean = 0.0

            # ====================================================
            # Show
            # ====================================================

            cv2.imshow(
                "D435 RGB",
                resize_for_display(rgb_bgr, args.display_scale)
            )

            cv2.imshow(
                "D435 Aligned Depth Visualization",
                resize_for_display(depth_colormap_with_bar, args.display_scale)
            )

            cv2.imshow(
                "SAM3 Mask",
                resize_for_display(mask_img, args.display_scale)
            )

            cv2.imshow(
                "SAM3 Overlay",
                resize_for_display(overlay_img, args.display_scale)
            )

            cv2.imshow(
                "SAM3 Depth PCL Projection",
                resize_for_display(pcl_projection_img, args.display_scale)
            )

            # ====================================================
            # Save
            # ====================================================

            saved = False

            should_auto_save = (
                args.auto_save and
                frame_count % args.save_every_n_frames == 0
            )

            if should_auto_save:
                save_index = save_current_result(
                    save_dir=args.save_dir,
                    save_index=save_index,
                    prompt_name=prompt_name,
                    rgb_bgr=rgb_bgr,
                    depth_raw_aligned=depth_raw_aligned,
                    depth_filtered_aligned=(
                        depth_filtered_aligned
                        if args.save_filtered_depth
                        else None
                    ),
                    depth_scale=depth_scale,
                    depth_colormap=depth_colormap,
                    depth_colormap_with_bar=depth_colormap_with_bar,
                    mask_img=mask_img,
                    overlay_img=overlay_img,
                    pcl_projection_img=pcl_projection_img,
                    points=points,
                    depth_for_pcl_source=depth_for_work_source,
                    verify_depth_png=args.verify_depth_png,
                )

                saved = True

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") or key == ord("S"):
                save_index = save_current_result(
                    save_dir=args.save_dir,
                    save_index=save_index,
                    prompt_name=prompt_name,
                    rgb_bgr=rgb_bgr,
                    depth_raw_aligned=depth_raw_aligned,
                    depth_filtered_aligned=(
                        depth_filtered_aligned
                        if args.save_filtered_depth
                        else None
                    ),
                    depth_scale=depth_scale,
                    depth_colormap=depth_colormap,
                    depth_colormap_with_bar=depth_colormap_with_bar,
                    mask_img=mask_img,
                    overlay_img=overlay_img,
                    pcl_projection_img=pcl_projection_img,
                    points=points,
                    depth_for_pcl_source=depth_for_work_source,
                    verify_depth_png=args.verify_depth_png,
                )

                saved = True

            if key == ord("q") or key == 27:
                print("[Exit] quit.")
                break

            # ====================================================
            # Timing print
            # ====================================================

            sam_ms = (t_sam1 - t_sam0) * 1000.0
            pcl_ms = (t_pcl1 - t_pcl0) * 1000.0
            total_ms = (time.perf_counter() - t0) * 1000.0

            print(
                f"frame={frame_count} "
                f"rgb_shape={rgb_bgr.shape} "
                f"depth_shape={depth_raw_aligned.shape} "
                f"depth_dtype={depth_raw_aligned.dtype} "
                f"depth_source={depth_for_work_source} "
                f"raw_valid_depth={raw_valid_count}/{total_count} "
                f"raw_zero_depth={raw_zero_count}/{total_count} "
                f"raw_invalid_depth={raw_invalid_count}/{total_count} "
                f"work_valid_depth={work_valid_count}/{total_count} "
                f"work_zero_depth={work_zero_count}/{total_count} "
                f"work_invalid_depth={work_invalid_count}/{total_count} "
                f"in_display={in_display_count}/{total_count} "
                f"depth_m_min={d_min:.4f} "
                f"depth_m_max={d_max:.4f} "
                f"depth_m_mean={d_mean:.4f} "
                f"run_sam={run_sam} "
                f"prompt='{used_prompt}' "
                f"score={score:.3f} "
                f"mask_pixels={mask_pixels} "
                f"pcl={cube_point_count} "
                f"displayed={displayed_points} "
                f"center=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] "
                f"bbox_size=[{bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f}] "
                f"sam_ms={sam_ms:.2f} "
                f"pcl_ms={pcl_ms:.2f} "
                f"total_ms={total_ms:.2f} "
                f"saved={saved}"
            )

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[RealSense] pipeline stopped.")


if __name__ == "__main__":
    main()
