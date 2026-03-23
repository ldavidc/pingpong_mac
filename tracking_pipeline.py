"""
Shared per-frame ping pong tracking (used by CLI and web UI).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from tracker_core import (
    BallKinematicsTracker,
    detect_ball,
    draw_kinematics_overlay,
    equivalent_radius_from_area,
    maybe_mirror_frame,
    preprocess_frame,
    velocity_within_limit,
)


@dataclass
class PipelineConfig:
    min_radius: int = 5
    max_radius: int = 80
    trail: int = 32
    max_width: int = 1280
    mirror: bool = True
    max_speed_px_s: Optional[float] = None
    meters_per_pixel: Optional[float] = None
    use_ball_area: bool = False
    ball_radius_m: float = 0.02
    overlay_pixel_units: bool = False
    footer_reset_hint: str = "O=reset disp. origin"


class TrackingPipeline:
    """
    One frame in: BGR from camera. Out: annotated BGR + binary mask.
    """

    def __init__(
        self,
        lower_hsv: np.ndarray,
        upper_hsv: np.ndarray,
        config: PipelineConfig,
    ) -> None:
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.conf = config
        self.points: deque = deque(maxlen=max(config.trail, 0))
        self.prev_frame_t = time.time()
        self.fps_smoothed = 0.0
        self.kin_tracker = BallKinematicsTracker()
        self.last_area_px: Optional[float] = None
        self.last_accept_pos: Optional[tuple[float, float]] = None
        self.last_accept_t: Optional[float] = None

    def reset_origin(self) -> None:
        self.kin_tracker.reset_origin()
        self.last_area_px = None
        self.last_accept_pos = None
        self.last_accept_t = None

    def step(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Returns (annotated_frame, mask, status) where status is
        'ok' | 'none' | 'rejected'.
        """
        cfg = self.conf
        m_per_px = cfg.meters_per_pixel
        use_ball_area = cfg.use_ball_area
        ball_radius_m = cfg.ball_radius_m

        frame_t = time.time()
        dt_frame = max(frame_t - self.prev_frame_t, 1e-6)
        self.prev_frame_t = frame_t

        frame = preprocess_frame(frame_bgr, cfg.max_width)
        frame = maybe_mirror_frame(frame, cfg.mirror)
        hit, mask = detect_ball(
            frame, cfg.min_radius, cfg.max_radius, self.lower_hsv, self.upper_hsv
        )
        raw_hit = hit

        accepted = False
        if hit is not None:
            x, y, radius, area, circularity = hit
            if velocity_within_limit(
                float(x),
                float(y),
                self.last_accept_pos,
                self.last_accept_t,
                frame_t,
                cfg.max_speed_px_s,
            ):
                accepted = True

        status = "none"
        if accepted:
            assert raw_hit is not None
            x, y, radius, area, circularity = raw_hit
            center = (x, y)
            self.points.appendleft(center)
            area_prev = self.last_area_px
            dt_kin = (
                max(frame_t - self.last_accept_t, dt_frame)
                if self.last_accept_t is not None
                else dt_frame
            )
            kin = self.kin_tracker.update(float(x), float(y), dt_kin)
            self.last_accept_pos = (float(x), float(y))
            self.last_accept_t = frame_t
            self.last_area_px = float(area)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"r={radius}px area={int(area)} circ={circularity:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (60, 220, 60),
                2,
                cv2.LINE_AA,
            )
            draw_kinematics_overlay(
                frame,
                kin,
                y0=55,
                meters_per_pixel=m_per_px,
                overlay_pixel_units=cfg.overlay_pixel_units,
                ball_area_scale=use_ball_area,
                area_px=float(area),
                area_px_prev=area_prev,
                ball_radius_m=ball_radius_m,
            )
            status = "ok"
        else:
            self.points.appendleft(None)
            if raw_hit is None:
                self.kin_tracker.on_lost()
                self.last_area_px = None
                self.last_accept_pos = None
                self.last_accept_t = None
                cv2.putText(
                    frame,
                    "No ball detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "Rejected: velocity too high (false positive?)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
                status = "rejected"

        for i in range(1, len(self.points)):
            if self.points[i - 1] is None or self.points[i] is None:
                continue
            thickness = (
                int(np.sqrt(cfg.trail / float(i + 1)) * 2.5) if cfg.trail > 0 else 1
            )
            cv2.line(frame, self.points[i - 1], self.points[i], (255, 0, 0), thickness)

        fps = 1.0 / dt_frame
        self.fps_smoothed = (
            fps if self.fps_smoothed == 0.0 else (self.fps_smoothed * 0.9 + fps * 0.1)
        )
        scale_hint = ""
        if use_ball_area and self.last_area_px is not None:
            req = equivalent_radius_from_area(self.last_area_px)
            mpp = ball_radius_m / max(req, 1e-6)
            scale_hint = f"  1px={mpp * 1000:.3f}mm (from ball area)"
        elif m_per_px is not None:
            scale_hint = f"  1px={m_per_px * 1000:.3f}mm"
        cv2.putText(
            frame,
            f"FPS: {self.fps_smoothed:.1f}   {cfg.footer_reset_hint}{scale_hint}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame, mask, status
