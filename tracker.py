"""
tracker.py — ByteTrack + Kalman Filter multi-object tracker

Architecture:
  - KalmanBoxFilter: constant-velocity 6D state [cx, cy, w, h, vx, vy]
    * Predicts next position using estimated velocity → smooth even during fast camera rotation
    * Handles 1-2 missed frames by continuing to extrapolate, then dies quickly (MAX_AGE=3)
  - Track lifecycle:  TENTATIVE ──(hits≥MIN_HITS)──► CONFIRMED ──(miss>MAX_AGE)──► deleted
    * TENTATIVE: just appeared, not shown yet → prevents single-frame false detections
    * CONFIRMED: shown on overlay with Kalman-smoothed coordinates
  - ByteTrack 2-stage matching:
    * Stage 1: high-confidence detections (≥HIGH_CONF) → match ALL tracks via IoU
    * Stage 2: low-confidence detections → match remaining CONFIRMED tracks
    * Creates new TENTATIVE track only from unmatched high-confidence detections

Result: no flickering (TENTATIVE phase), no residual boxes (MAX_AGE=3 vs old TTL=10),
        smooth tracking even at high speed (Kalman prediction).
"""

import numpy as np
from detector import Detection
from config import (
    CONFIDENCE_THRESHOLD, NEW_TRACK_CONF,
    TRACKER_IOU_THRESH, TRACKER_HIGH_CONF,
    TRACKER_MAX_AGE, TRACKER_MIN_HITS,
)


# ── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanBoxFilter:
    """
    Constant-velocity Kalman filter for a single bounding box.

    State  (6D): [cx, cy,  w,  h, vx, vy]
    Measurement(4D): [cx, cy,  w,  h]

    F (state transition): position += velocity each frame, size is static.
    Q (process noise):    velocity can change suddenly (character acceleration).
    R (measurement noise):YOLO bbox coordinates fluctuate ±5–10px per frame.
    """

    def __init__(self, cx: float, cy: float, w: float, h: float):
        self.x = np.array([cx, cy, w, h, 0., 0.], dtype=np.float64)

        # Transition: cx(t+1)=cx(t)+vx,  cy(t+1)=cy(t)+vy,  w/h/v constant
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement maps state → [cx, cy, w, h]
        self.H = np.eye(4, 6, dtype=np.float64)

        # Initial covariance: position well-known from first detection, velocity unknown
        self.P = np.diag([10., 10., 20., 20., 1000., 1000.]).astype(np.float64)

        # Process noise: allow large velocity changes (fast acceleration in-game)
        self.Q = np.diag([1., 1., 2., 2., 20., 20.]).astype(np.float64)

        # Measurement noise: YOLO boxes vary ~5px in position, ~10px in size
        self.R = np.diag([4., 4., 9., 9.]).astype(np.float64)

    def predict(self) -> tuple:
        """Advance state by one frame using constant-velocity model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Keep dimensions positive
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self._to_xyxy()

    def update(self, cx: float, cy: float, w: float, h: float) -> tuple:
        """Correct state with a new detection measurement."""
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R         # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self._to_xyxy()

    def _to_xyxy(self) -> tuple:
        cx, cy, w, h = self.x[:4]
        return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


# ── Track ─────────────────────────────────────────────────────────────────────

_TENTATIVE = 0   # just created, not shown yet
_CONFIRMED = 1   # confirmed, shown on overlay


class _Track:
    _id_counter = 1

    def __init__(self, det: Detection):
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        w  = float(det.x2 - det.x1)
        h  = float(det.y2 - det.y1)

        self.kf = KalmanBoxFilter(cx, cy, w, h)
        self.id = _Track._id_counter
        _Track._id_counter += 1

        self.state       = _TENTATIVE
        self.conf        = det.confidence
        self.hits        = 1   # consecutive successful matches
        self.miss_streak = 0   # consecutive missed frames
        self._box        = (float(det.x1), float(det.y1),
                            float(det.x2), float(det.y2))

    def predict(self):
        self._box = self.kf.predict()
        self.miss_streak += 1

    def update(self, det: Detection):
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        w  = float(det.x2 - det.x1)
        h  = float(det.y2 - det.y1)
        self._box = self.kf.update(cx, cy, w, h)
        # Smooth confidence to avoid label jitter
        self.conf        = 0.65 * det.confidence + 0.35 * self.conf
        self.hits       += 1
        self.miss_streak = 0
        if self.state == _TENTATIVE and self.hits >= TRACKER_MIN_HITS:
            self.state = _CONFIRMED

    @property
    def box_ints(self) -> tuple:
        x1, y1, x2, y2 = self._box
        return (int(x1), int(y1), int(x2), int(y2))

    def to_detection(self, screen_cx: int, screen_cy: int) -> Detection:
        x1, y1, x2, y2 = self.box_ints
        return Detection(
            x1, y1, x2, y2, self.conf,
            track_id=self.id,
            _screen_cx=screen_cx,
            _screen_cy=screen_cy,
        )


# ── IoU Matrix + Greedy Assignment ───────────────────────────────────────────

def _iou_matrix(tracks: list, dets: list) -> np.ndarray:
    """Returns [N_tracks, N_dets] IoU matrix."""
    n, m = len(tracks), len(dets)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    tb = np.array([t._box for t in tracks], dtype=np.float32)            # [N, 4]
    db = np.array([(d.x1, d.y1, d.x2, d.y2) for d in dets], dtype=np.float32)  # [M, 4]

    ix1 = np.maximum(tb[:, 0:1], db[:, 0])
    iy1 = np.maximum(tb[:, 1:2], db[:, 1])
    ix2 = np.minimum(tb[:, 2:3], db[:, 2])
    iy2 = np.minimum(tb[:, 3:4], db[:, 3])

    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    area_t = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
    area_d = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
    union  = area_t[:, None] + area_d[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _greedy_match(iou: np.ndarray, thresh: float):
    """
    Greedy max-IoU matching (near-optimal for N < 20).
    Returns: ([(track_idx, det_idx)], unmatched_track_idxs, unmatched_det_idxs)
    """
    n_t, n_d = iou.shape
    used_t: set[int] = set()
    used_d: set[int] = set()
    pairs:  list[tuple[int, int]] = []

    flat = np.argsort(iou.flatten())[::-1]  # descending IoU
    for idx in flat:
        ti, di = divmod(int(idx), n_d)
        if iou[ti, di] < thresh:
            break
        if ti not in used_t and di not in used_d:
            pairs.append((ti, di))
            used_t.add(ti)
            used_d.add(di)

    unmatched_t = [i for i in range(n_t) if i not in used_t]
    unmatched_d = [i for i in range(n_d) if i not in used_d]
    return pairs, unmatched_t, unmatched_d


# ── ByteTracker ───────────────────────────────────────────────────────────────

class ByteTracker:
    """
    ByteTrack-style tracker with Kalman filter.

    Lifecycle:
      New det (conf≥NEW_TRACK_CONF) → TENTATIVE
      TENTATIVE + hits≥TRACKER_MIN_HITS → CONFIRMED  (shown on overlay)
      CONFIRMED + miss_streak > TRACKER_MAX_AGE → deleted

    The short MAX_AGE (3 frames ≈ 67ms @ 45 FPS) ensures residual boxes
    disappear quickly when you rotate the view.
    """

    def __init__(self):
        self._tracks: list[_Track] = []

    def reset(self):
        self._tracks.clear()
        _Track._id_counter = 1

    def update(
        self,
        detections: list[Detection],
        screen_w: int,
        screen_h: int,
    ) -> list[Detection]:

        # ── 1. Predict all tracks ──────────────────────────────────────────────
        for t in self._tracks:
            t.predict()

        # ── 2. Split detections by confidence ─────────────────────────────────
        high_dets = [d for d in detections if d.confidence >= TRACKER_HIGH_CONF]
        low_dets  = [d for d in detections
                     if CONFIDENCE_THRESHOLD <= d.confidence < TRACKER_HIGH_CONF]

        # ── 3. Stage 1: match high-conf dets → ALL tracks ─────────────────────
        if self._tracks and high_dets:
            iou1 = _iou_matrix(self._tracks, high_dets)
            pairs1, unmatched_t1, unmatched_hd = _greedy_match(iou1, TRACKER_IOU_THRESH)
            for ti, di in pairs1:
                self._tracks[ti].update(high_dets[di])
        else:
            unmatched_t1 = list(range(len(self._tracks)))
            unmatched_hd = list(range(len(high_dets)))

        # ── 4. Stage 2: match low-conf dets → remaining CONFIRMED tracks ──────
        still_unmatched_t = set(unmatched_t1)
        confirmed_remaining = [(ti, self._tracks[ti])
                               for ti in unmatched_t1
                               if self._tracks[ti].state == _CONFIRMED]
        if confirmed_remaining and low_dets:
            rc_tracks = [t for _, t in confirmed_remaining]
            rc_idx    = [i for i, _ in confirmed_remaining]
            iou2 = _iou_matrix(rc_tracks, low_dets)
            low_thresh = max(TRACKER_IOU_THRESH * 0.6, 0.10)
            pairs2, _, _ = _greedy_match(iou2, low_thresh)
            for ti2, di2 in pairs2:
                orig_ti = rc_idx[ti2]
                self._tracks[orig_ti].update(low_dets[di2])
                still_unmatched_t.discard(orig_ti)

        # ── 5. Create new TENTATIVE tracks from unmatched high-conf dets ──────
        for di in unmatched_hd:
            if high_dets[di].confidence >= NEW_TRACK_CONF:
                self._tracks.append(_Track(high_dets[di]))

        # ── 6. Prune dead tracks ───────────────────────────────────────────────
        self._tracks = [
            t for t in self._tracks
            if not (
                (t.state == _TENTATIVE and t.miss_streak > 1)
                or (t.state == _CONFIRMED and t.miss_streak >= TRACKER_MAX_AGE)
            )
        ]

        # ── 7. Return CONFIRMED tracks sorted by proximity to crosshair ───────
        cx, cy = screen_w // 2, screen_h // 2
        result = [t.to_detection(cx, cy)
                  for t in self._tracks if t.state == _CONFIRMED]
        result.sort(key=lambda d: d.distance_to_center)
        return result
