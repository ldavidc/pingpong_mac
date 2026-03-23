"""
Shared preprocessing and ball detection for ping pong trackers.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Default: white / light ball (works without calibration)
DEFAULT_LOWER_HSV = np.array([0, 0, 160], dtype=np.uint8)
DEFAULT_UPPER_HSV = np.array([179, 70, 255], dtype=np.uint8)


@dataclass(frozen=True)
class BallKinematics:
    """Instantaneous motion in image space (pixels / seconds)."""

    # Step from previous frame (px)
    dx_px: float
    dy_px: float
    step_px: float
    # Velocity (px/s); zero on first frame after a detection gap
    vx_px_s: float
    vy_px_s: float
    speed_px_s: float
    # Displacement from origin set at first detection (or after reset)
    disp_x_px: float
    disp_y_px: float
    dist_from_origin_px: float


class BallKinematicsTracker:
    """
    Tracks frame-to-frame displacement and velocity, and cumulative displacement
    from an origin. Call :meth:`on_lost` when the ball is not detected so the next
    hit does not produce a huge velocity spike.
    """

    def __init__(self) -> None:
        self._origin: Optional[tuple[float, float]] = None
        self._last: Optional[tuple[float, float]] = None

    def reset_origin(self) -> None:
        """Next detection becomes (0,0) displacement reference."""
        self._origin = None
        self._last = None

    def on_lost(self) -> None:
        """Forget previous position (e.g. ball left FOV)."""
        self._last = None

    def update(self, x: float, y: float, dt_s: float) -> BallKinematics:
        if self._origin is None:
            self._origin = (x, y)

        ox, oy = self._origin
        disp_x = x - ox
        disp_y = y - oy
        dist_o = math.hypot(disp_x, disp_y)

        if self._last is None or dt_s <= 0:
            self._last = (x, y)
            return BallKinematics(
                dx_px=0.0,
                dy_px=0.0,
                step_px=0.0,
                vx_px_s=0.0,
                vy_px_s=0.0,
                speed_px_s=0.0,
                disp_x_px=disp_x,
                disp_y_px=disp_y,
                dist_from_origin_px=dist_o,
            )

        lx, ly = self._last
        dx = x - lx
        dy = y - ly
        self._last = (x, y)
        vx = dx / dt_s
        vy = dy / dt_s
        return BallKinematics(
            dx_px=dx,
            dy_px=dy,
            step_px=math.hypot(dx, dy),
            vx_px_s=vx,
            vy_px_s=vy,
            speed_px_s=math.hypot(vx, vy),
            disp_x_px=disp_x,
            disp_y_px=disp_y,
            dist_from_origin_px=dist_o,
        )


def implied_speed_px_s(
    x: float,
    y: float,
    last_pos: Optional[tuple[float, float]],
    last_t: Optional[float],
    now_t: float,
) -> Optional[float]:
    """
    Speed (pixels per second) implied by moving from ``last_pos`` at ``last_t``
    to ``(x, y)`` at ``now_t``. ``None`` if there is no prior accepted point.
    """
    if last_pos is None or last_t is None:
        return None
    dt = max(now_t - last_t, 1e-6)
    dist = math.hypot(x - last_pos[0], y - last_pos[1])
    return dist / dt


def velocity_within_limit(
    x: float,
    y: float,
    last_pos: Optional[tuple[float, float]],
    last_t: Optional[float],
    now_t: float,
    max_speed_px_s: Optional[float],
) -> bool:
    """
    If ``max_speed_px_s`` is set, reject detections whose implied speed from the
    last **accepted** position exceeds it (filters many false positives).
    First detection after a gap is always allowed.
    """
    if max_speed_px_s is None:
        return True
    spd = implied_speed_px_s(x, y, last_pos, last_t, now_t)
    if spd is None:
        return True
    return spd <= max_speed_px_s


def add_velocity_gate_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group(
        "False positive filtering",
        "Reject blob jumps that would require impossible ball speeds.",
    )
    g.add_argument(
        "--max-speed-px-s",
        type=float,
        default=None,
        metavar="PX_S",
        help=(
            "Max plausible speed in pixels per second vs last accepted position. "
            "If set, detections above this are ignored (e.g. 6000–12000 for HD). "
            "Default: disabled."
        ),
    )


@dataclass(frozen=True)
class BallKinematicsWorld:
    """Same kinematics in meters (see :func:`world_from_px`)."""

    dx_m: float
    dy_m: float
    step_m: float
    vx_m_s: float
    vy_m_s: float
    speed_m_s: float
    disp_x_m: float
    disp_y_m: float
    dist_from_origin_m: float


def world_from_px(kin: BallKinematics, meters_per_pixel: float) -> BallKinematicsWorld:
    """Convert pixel kinematics to meters using a uniform scale (meters per pixel)."""
    s = meters_per_pixel
    return BallKinematicsWorld(
        dx_m=kin.dx_px * s,
        dy_m=kin.dy_px * s,
        step_m=kin.step_px * s,
        vx_m_s=kin.vx_px_s * s,
        vy_m_s=kin.vy_px_s * s,
        speed_m_s=kin.speed_px_s * s,
        disp_x_m=kin.disp_x_px * s,
        disp_y_m=kin.disp_y_px * s,
        dist_from_origin_m=kin.dist_from_origin_px * s,
    )


def equivalent_radius_from_area(area_px: float) -> float:
    """
    Radius (px) of a circle with the same area as the contour blob.

    Used with a known physical ball radius to estimate meters-per-pixel near the ball
    (larger area ⇒ closer / larger apparent size ⇒ fewer meters per pixel).
    """
    return math.sqrt(max(float(area_px), 1.0) / math.pi)


def world_from_ball_area(
    kin: BallKinematics,
    area_px: float,
    area_px_prev: Optional[float],
    ball_radius_m: float,
) -> BallKinematicsWorld:
    """
    Convert pixel kinematics to meters using apparent ball size.

    Assumes the blob's area matches a disk of the real ball (diameter ~40 mm), so
    ``r_eq = sqrt(area/π)`` in pixels and ``m/px ≈ ball_radius_m / r_eq``.

    - **Step / velocity:** uses the average of current and previous equivalent
      radii (when previous exists) so motion toward/away from the camera blends
      scales between frames.
    - **Displacement from origin:** uses the **current** frame's area only.
    """
    r_curr = equivalent_radius_from_area(area_px)
    mpp_disp = ball_radius_m / max(r_curr, 1e-6)

    if area_px_prev is None:
        mpp_step = mpp_disp
    else:
        r_prev = equivalent_radius_from_area(area_px_prev)
        r_avg = 0.5 * (r_curr + r_prev)
        mpp_step = ball_radius_m / max(r_avg, 1e-6)

    return BallKinematicsWorld(
        dx_m=kin.dx_px * mpp_step,
        dy_m=kin.dy_px * mpp_step,
        step_m=kin.step_px * mpp_step,
        vx_m_s=kin.vx_px_s * mpp_step,
        vy_m_s=kin.vy_px_s * mpp_step,
        speed_m_s=kin.speed_px_s * mpp_step,
        disp_x_m=kin.disp_x_px * mpp_disp,
        disp_y_m=kin.disp_y_px * mpp_disp,
        dist_from_origin_m=kin.dist_from_origin_px * mpp_disp,
    )


def add_real_world_scale_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --m-per-pixel / reference pair / overlay options (optional group)."""
    g = parser.add_argument_group(
        "Real-world scale (meters)",
        "Set meters per pixel, or derive it from a known length in the image "
        "(e.g. table length 2.74 m and its span in pixels).",
    )
    g.add_argument(
        "--m-per-pixel",
        type=float,
        default=None,
        metavar="M",
        help="Meters per image pixel (uniform scale). E.g. 0.002 means 1 px = 2 mm.",
    )
    g.add_argument(
        "--reference-meters",
        type=float,
        default=None,
        metavar="M",
        help="Real length of a reference segment (meters), used with --reference-pixels.",
    )
    g.add_argument(
        "--reference-pixels",
        type=float,
        default=None,
        metavar="PX",
        help="Length of that same segment in the processed image (pixels).",
    )
    g.add_argument(
        "--overlay-pixel-units",
        action="store_true",
        help="When using real-world scale, also show pixel lines under the meter readout.",
    )
    g.add_argument(
        "--scale-from-ball-area",
        action="store_true",
        help=(
            "Estimate m/px from contour area and physical ball size (see --ball-diameter-m). "
            "Conflicts with --m-per-pixel and --reference-*."
        ),
    )
    g.add_argument(
        "--ball-diameter-m",
        type=float,
        default=0.04,
        metavar="M",
        help="Physical ball diameter in meters (regulation ~0.04). Used with --scale-from-ball-area.",
    )


def resolve_meters_per_pixel_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Optional[float]:
    """
    Returns meters per pixel, or None if no scale was given.
    Calls parser.error() on invalid combinations.
    """
    mpp = getattr(args, "m_per_pixel", None)
    rm = getattr(args, "reference_meters", None)
    rp = getattr(args, "reference_pixels", None)

    has_ref = rm is not None or rp is not None
    if mpp is not None and has_ref:
        parser.error("Use either --m-per-pixel or --reference-meters/--reference-pixels, not both.")
    if mpp is not None:
        if mpp <= 0:
            parser.error("--m-per-pixel must be positive.")
        return mpp
    if rm is not None or rp is not None:
        if rm is None or rp is None:
            parser.error("Provide both --reference-meters and --reference-pixels.")
        if rm <= 0:
            parser.error("--reference-meters must be positive.")
        if rp <= 0:
            parser.error("--reference-pixels must be positive.")
        return rm / rp
    return None


def resolve_scale_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Optional[float], bool, float]:
    """
    Returns ``(uniform_meters_per_pixel | None, use_ball_area_scale, ball_radius_m)``.

    ``ball_radius_m`` is half of ``--ball-diameter-m`` (for API consistency).
    """
    use_ball = bool(getattr(args, "scale_from_ball_area", False))
    diameter_m = float(getattr(args, "ball_diameter_m", 0.04))
    ball_radius_m = diameter_m / 2.0

    has_uniform = (
        getattr(args, "m_per_pixel", None) is not None
        or getattr(args, "reference_meters", None) is not None
        or getattr(args, "reference_pixels", None) is not None
    )
    if use_ball and has_uniform:
        parser.error(
            "Do not combine --scale-from-ball-area with --m-per-pixel or "
            "--reference-meters/--reference-pixels."
        )
    if use_ball:
        if diameter_m <= 0:
            parser.error("--ball-diameter-m must be positive.")
        return None, True, ball_radius_m

    mpp = resolve_meters_per_pixel_from_args(args, parser)
    return mpp, False, ball_radius_m


def draw_kinematics_overlay(
    frame: np.ndarray,
    kin: BallKinematics,
    x: int = 10,
    y0: int = 55,
    line_gap: int = 22,
    font_scale: float = 0.55,
    color: tuple[int, int, int] = (200, 230, 255),
    meters_per_pixel: Optional[float] = None,
    overlay_pixel_units: bool = False,
    ball_area_scale: bool = False,
    area_px: Optional[float] = None,
    area_px_prev: Optional[float] = None,
    ball_radius_m: float = 0.02,
) -> None:
    """
    Draw velocity / displacement: meters if scaled, else pixels.

    Set ``ball_area_scale`` + ``area_px`` to use :func:`world_from_ball_area`;
    otherwise ``meters_per_pixel`` for uniform :func:`world_from_px`.

    Image coords: +x right, +y down.
    """
    lines: list[tuple[str, float, tuple[int, int, int]]] = []

    if ball_area_scale and area_px is not None:
        w = world_from_ball_area(kin, area_px, area_px_prev, ball_radius_m)
        r_eq = equivalent_radius_from_area(area_px)
        lines.extend(
            [
                (
                    f"v=({w.vx_m_s:+.3f},{w.vy_m_s:+.3f}) m/s  |v|={w.speed_m_s:.3f}",
                    font_scale,
                    (160, 255, 220),
                ),
                (
                    f"step=({w.dx_m:+.4f},{w.dy_m:+.4f}) m  step={w.step_m:.4f}",
                    font_scale,
                    (160, 255, 220),
                ),
                (
                    f"disp=({w.disp_x_m:+.4f},{w.disp_y_m:+.4f}) m  |disp|={w.dist_from_origin_m:.4f}",
                    font_scale,
                    (160, 255, 220),
                ),
                (
                    f"r_eq={r_eq:.1f}px from area={area_px:.0f}px^2  (d_ball={2*ball_radius_m*1000:.0f}mm)",
                    font_scale * 0.82,
                    (140, 200, 200),
                ),
            ]
        )
        if overlay_pixel_units:
            lines.extend(
                [
                    (
                        f"(px) v=({kin.vx_px_s:+.1f},{kin.vy_px_s:+.1f}) px/s  |v|={kin.speed_px_s:.1f}",
                        font_scale * 0.85,
                        color,
                    ),
                    (
                        f"(px) step=({kin.dx_px:+.1f},{kin.dy_px:+.1f})  step={kin.step_px:.1f}",
                        font_scale * 0.85,
                        color,
                    ),
                    (
                        f"(px) disp=({kin.disp_x_px:+.1f},{kin.disp_y_px:+.1f})  |disp|={kin.dist_from_origin_px:.1f}",
                        font_scale * 0.85,
                        color,
                    ),
                ]
            )
    elif meters_per_pixel is not None:
        w = world_from_px(kin, meters_per_pixel)
        lines.extend(
            [
                (
                    f"v=({w.vx_m_s:+.3f},{w.vy_m_s:+.3f}) m/s  |v|={w.speed_m_s:.3f}",
                    font_scale,
                    (180, 255, 200),
                ),
                (
                    f"step=({w.dx_m:+.4f},{w.dy_m:+.4f}) m  step={w.step_m:.4f}",
                    font_scale,
                    (180, 255, 200),
                ),
                (
                    f"disp=({w.disp_x_m:+.4f},{w.disp_y_m:+.4f}) m  |disp|={w.dist_from_origin_m:.4f}",
                    font_scale,
                    (180, 255, 200),
                ),
            ]
        )
        if overlay_pixel_units:
            lines.extend(
                [
                    (
                        f"(px) v=({kin.vx_px_s:+.1f},{kin.vy_px_s:+.1f}) px/s  |v|={kin.speed_px_s:.1f}",
                        font_scale * 0.85,
                        color,
                    ),
                    (
                        f"(px) step=({kin.dx_px:+.1f},{kin.dy_px:+.1f})  step={kin.step_px:.1f}",
                        font_scale * 0.85,
                        color,
                    ),
                    (
                        f"(px) disp=({kin.disp_x_px:+.1f},{kin.disp_y_px:+.1f})  |disp|={kin.dist_from_origin_px:.1f}",
                        font_scale * 0.85,
                        color,
                    ),
                ]
            )
    else:
        lines.extend(
            [
                (
                    f"v=({kin.vx_px_s:+.1f},{kin.vy_px_s:+.1f}) px/s  |v|={kin.speed_px_s:.1f}",
                    font_scale,
                    color,
                ),
                (
                    f"step=({kin.dx_px:+.1f},{kin.dy_px:+.1f}) px  step={kin.step_px:.1f}",
                    font_scale,
                    color,
                ),
                (
                    f"disp=({kin.disp_x_px:+.1f},{kin.disp_y_px:+.1f}) px  |disp|={kin.dist_from_origin_px:.1f}",
                    font_scale,
                    color,
                ),
            ]
        )

    for i, (text, fs, col) in enumerate(lines):
        y = y0 + i * line_gap
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            col,
            1,
            cv2.LINE_AA,
        )


def preprocess_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        frame = cv2.resize(
            frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return frame


def maybe_mirror_frame(frame: np.ndarray, mirror: bool) -> np.ndarray:
    """
    Horizontal flip (mirror) so left/right match a typical selfie-style webcam preview.
    """
    if not mirror:
        return frame
    return cv2.flip(frame, 1)


def detect_ball(
    frame: np.ndarray,
    min_radius: int,
    max_radius: int,
    lower_hsv: np.ndarray,
    upper_hsv: np.ndarray,
) -> tuple[Optional[tuple], np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        if radius < min_radius or radius > max_radius:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        score = area * max(circularity, 0.0)
        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(radius), area, circularity)

    return best, mask


def hsv_bounds_from_roi(
    hsv: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    hue_pad: float = 6.0,
    sat_pad: float = 14.0,
    val_pad: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute lower/upper HSV from pixels inside the calibration ROI.

    Uses **percentiles** (not mean±std with large floors), so white/light balls
    don't get huge S/V ranges that turn the whole frame "white-washed".

    For **low-saturation** samples (typical white ball), pads are tightened further
    and saturation/value spans are capped so the mask stays discriminative.
    OpenCV H: 0–179.
    """
    roi = hsv[y : y + h, x : x + w]
    if roi.size == 0:
        raise ValueError("Empty ROI")

    pixels = roi.reshape(-1, 3).astype(np.float64)
    # Inner percentiles: robust to a few edge pixels in the drag box
    p_lo = np.percentile(pixels, 8, axis=0)
    p_hi = np.percentile(pixels, 92, axis=0)
    mean_s = float(np.mean(pixels[:, 1]))

    # White / very light ball: old mean/std + big floors made S and V span ~full range
    if mean_s < 52:
        pad = np.array([4.0, 6.0, 10.0])
    else:
        pad = np.array([hue_pad, sat_pad, val_pad])

    lower = p_lo - pad
    upper = p_hi + pad

    # Cap saturation span for low-S ROIs (keeps table/sky from matching "any white")
    if mean_s < 52:
        s_span = float(upper[1] - lower[1])
        max_s_span = 42.0
        if s_span > max_s_span:
            sm = (upper[1] + lower[1]) / 2.0
            lower[1] = sm - max_s_span / 2.0
            upper[1] = sm + max_s_span / 2.0
        # Cap value span similarly (bright table + ball both high V)
        v_span = float(upper[2] - lower[2])
        max_v_span = 48.0
        if v_span > max_v_span:
            vm = (upper[2] + lower[2]) / 2.0
            lower[2] = vm - max_v_span / 2.0
            upper[2] = vm + max_v_span / 2.0

    lower[0] = np.clip(lower[0], 0, 179)
    upper[0] = np.clip(upper[0], 0, 179)
    lower[1] = np.clip(lower[1], 0, 255)
    upper[1] = np.clip(upper[1], 0, 255)
    lower[2] = np.clip(lower[2], 0, 255)
    upper[2] = np.clip(upper[2], 0, 255)

    return lower.astype(np.uint8), upper.astype(np.uint8)


def save_calibration(path: str | Path, lower: np.ndarray, upper: np.ndarray) -> None:
    path = Path(path)
    data = {
        "lower_hsv": lower.astype(int).tolist(),
        "upper_hsv": upper.astype(int).tolist(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    lower = np.array(data["lower_hsv"], dtype=np.uint8)
    upper = np.array(data["upper_hsv"], dtype=np.uint8)
    return lower, upper
