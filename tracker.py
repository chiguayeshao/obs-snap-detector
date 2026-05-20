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
    TRACKER_CENTER_DIST_FALLBACK, TRACKER_DEDUP_DIST, TRACKER_GATE_DIST,
    TRACKER_REID_DIST, TRACKER_REID_TTL, TRACKER_GHOST_MISS_LIMIT,
    HEAD_ZONE_RATIO,
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

        # Measurement noise: increased to smooth YOLO box jitter
        # Higher R → Kalman trusts prediction more → less per-frame shimmer
        # σ≈8px position, σ≈10px size — good balance for fast-moving game targets
        self.R = np.diag([64., 64., 100., 100.]).astype(np.float64)

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

    def __init__(self, det: Detection, reuse_id: int | None = None):
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        w  = float(det.x2 - det.x1)
        h  = float(det.y2 - det.y1)

        self.kf = KalmanBoxFilter(cx, cy, w, h)
        if reuse_id is not None:
            self.id = reuse_id
        else:
            self.id = _Track._id_counter
            _Track._id_counter += 1

        self.state       = _TENTATIVE
        self.conf        = det.confidence
        self.hits        = 1   # consecutive successful matches
        self.miss_streak = 0   # consecutive missed frames
        self._box        = (float(det.x1), float(det.y1),
                            float(det.x2), float(det.y2))
        # EMA snap_point: running smoothed HEAD POSITION across box-size changes.
        # Using snap = top-center of HEAD_ZONE_RATIO of detection bbox.
        # This stays stable even when YOLO alternates between full-body and head-only
        # detections — it's the anchor used for Stage 1b matching and DEDUP.
        head_h0 = max(h * HEAD_ZONE_RATIO, 10.0)
        self._snap_ema: tuple = (cx, float(det.y1) + head_h0 * 0.5)
        # With MIN_HITS=1 the very first detection immediately confirms the track
        if self.hits >= TRACKER_MIN_HITS:
            self.state = _CONFIRMED

    def predict(self):
        self._box = self.kf.predict()
        self.miss_streak += 1
        # Freeze velocity after first miss: prevents Kalman ghost box drift during
        # camera rotation. The box stays at its last known position instead of
        # continuing to extrapolate in the last-known motion direction.
        if self.miss_streak >= 1:
            self.kf.x[4] = 0.0  # vx = 0
            self.kf.x[5] = 0.0  # vy = 0

    def update(self, det: Detection):
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        w  = float(det.x2 - det.x1)
        h  = float(det.y2 - det.y1)
        self._box = self.kf.update(cx, cy, w, h)
        # Slow EMA for confidence display stability:
        # alpha=0.12 means ~8-frame time constant → confidence changes smoothly
        # (prevents rapid flicker between YOLO's head-only and body-only detections)
        alpha = 0.12
        self.conf        = alpha * det.confidence + (1.0 - alpha) * self.conf
        self.hits       += 1
        self.miss_streak = 0
        if self.state == _TENTATIVE and self.hits >= TRACKER_MIN_HITS:
            self.state = _CONFIRMED
        # Update EMA snap_point from the NEW detection's bbox (not Kalman predicted box).
        # Using alpha=0.4: fast enough to track real head movement, smooth enough to
        # ignore single-frame bbox jitter.
        dh = max(h * HEAD_ZONE_RATIO, 10.0)
        dsnx = cx
        dsny = float(det.y1) + dh * 0.5
        alpha_s = 0.40
        self._snap_ema = (
            alpha_s * dsnx + (1.0 - alpha_s) * self._snap_ema[0],
            alpha_s * dsny + (1.0 - alpha_s) * self._snap_ema[1],
        )

    def update_weak(self, det: Detection):
        """Stage 2 (low-confidence) match: update Kalman + snap_ema position,
        but do NOT reset miss_streak. This prevents background noise detections
        (4-6% confidence environment objects) from keeping ghost tracks alive
        when the camera has moved away from real targets."""
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        w  = float(det.x2 - det.x1)
        h  = float(det.y2 - det.y1)
        self._box = self.kf.update(cx, cy, w, h)
        # miss_streak intentionally NOT reset — track still counts as "missed" this frame.
        # snap_ema uses same alpha=0.40 as full update() to keep DEDUP accurate:
        # low alpha here would let snap_ema drift from actual target position, causing
        # DEDUP failures (>170px gap) and spurious duplicate tracks on the same target.
        dh = max(h * HEAD_ZONE_RATIO, 10.0)
        alpha_s = 0.40
        self._snap_ema = (
            alpha_s * cx + (1.0 - alpha_s) * self._snap_ema[0],
            alpha_s * (float(det.y1) + dh * 0.5) + (1.0 - alpha_s) * self._snap_ema[1],
        )

    @property
    def box_ints(self) -> tuple:
        x1, y1, x2, y2 = self._box
        return (int(x1), int(y1), int(x2), int(y2))

    def to_detection(self, screen_cx: int, screen_cy: int) -> Detection:
        x1, y1, x2, y2 = self.box_ints
        # Use snap_ema (smoothed head position EMA) for distance_to_center.
        # This prevents Kalman box jitter (±120px) from causing primary-target cycling.
        sx, sy = self._snap_ema
        snap_dist = ((sx - screen_cx) ** 2 + (sy - screen_cy) ** 2) ** 0.5
        return Detection(
            x1, y1, x2, y2, self.conf,
            track_id=self.id,
            _screen_cx=screen_cx,
            _screen_cy=screen_cy,
            _snap_dist=snap_dist,
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


def _gated_iou_matrix(tracks: list, dets: list, max_snap_dist: float) -> np.ndarray:
    """
    IoU matrix with a snap-point distance gate.

    After computing the standard IoU, zeroes out any cell where the Euclidean
    distance between track._snap_ema and the detection's head snap_point exceeds
    max_snap_dist pixels.

    Why this fixes near-target drift:
      A near-target track (T6) has a large Kalman body box (~300x600px) that can
      overlap a medium-distance detection with IoU≈0.26.  The greedy matcher then
      assigns T6 to the wrong detection, causing it to drift.  But the snap_ema of
      T6 (head position of near target) is ~120px away from the medium detection's
      snap_point — gating kills that false match, so T6 stays on the near target.

    Safe zone: at 39fps the near-target moves <5px/frame; gate=120px allows >3x
    that in any direction, so true fast-moving game targets are never blocked.
    """
    iou = _iou_matrix(tracks, dets)
    if iou.size == 0:
        return iou

    # Use track snap_ema vs det head snap for the gate
    def _det_snap_g(d):
        head_h = max((d.y2 - d.y1) * HEAD_ZONE_RATIO, 10.0)
        return (d.x1 + d.x2) * 0.5, d.y1 + head_h * 0.5

    ts = np.array([t._snap_ema for t in tracks], dtype=np.float32)   # [N, 2]
    ds = np.array([_det_snap_g(d) for d in dets], dtype=np.float32)  # [M, 2]
    diff = ts[:, None, :] - ds[None, :, :]
    snap_dist = np.sqrt((diff ** 2).sum(axis=2))   # [N, M]
    iou[snap_dist > max_snap_dist] = 0.0
    return iou


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


def _center_dist_matrix(tracks: list, dets: list) -> np.ndarray:
    """Returns [N_tracks, N_dets] center-to-center Euclidean distance matrix (pixels)."""
    n, m = len(tracks), len(dets)
    if n == 0 or m == 0:
        return np.full((n, m), np.inf, dtype=np.float32)
    # Use _box (x1,y1,x2,y2) center — most current estimate (Kalman predicted or updated)
    tc = np.array([[0.5*(t._box[0]+t._box[2]), 0.5*(t._box[1]+t._box[3])]
                   for t in tracks], dtype=np.float32)  # [N, 2]
    dc = np.array([[0.5*(d.x1+d.x2), 0.5*(d.y1+d.y2)]
                   for d in dets], dtype=np.float32)    # [M, 2]
    diff = tc[:, None, :] - dc[None, :, :]  # [N, M, 2]
    return np.sqrt((diff**2).sum(axis=2))   # [N, M]


def _snap_dist_matrix(tracks: list, dets: list) -> np.ndarray:
    """
    Returns [N_tracks, N_dets] snap-point Euclidean distance matrix (pixels).

    Uses track._snap_ema — a smoothed running estimate of each track's head position
    that is updated ONLY from real detections (not Kalman predictions).
    This is stable even as YOLO alternates between full-body and head-only boxes:
    the EMA converges to the true head position and stays there.

    For detections, computes snap_point on-the-fly from the YOLO bbox.
    """
    n, m = len(tracks), len(dets)
    if n == 0 or m == 0:
        return np.full((n, m), np.inf, dtype=np.float32)

    # Use stored EMA for tracks — immune to Kalman box-size drift
    ts = np.array([t._snap_ema for t in tracks], dtype=np.float32)

    # Compute snap_point from raw detection bbox
    def _det_snap(d):
        head_h = max((d.y2 - d.y1) * HEAD_ZONE_RATIO, 10.0)
        return (d.x1 + d.x2) * 0.5, d.y1 + head_h * 0.5
    ds = np.array([_det_snap(d) for d in dets], dtype=np.float32)

    diff = ts[:, None, :] - ds[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _greedy_match_by_dist(dist: np.ndarray, max_dist: float):
    """
    Greedy min-distance matching.
    Returns: ([(track_idx, det_idx)], unmatched_track_idxs, unmatched_det_idxs)
    """
    n_t, n_d = dist.shape
    used_t: set[int] = set()
    used_d: set[int] = set()
    pairs:  list[tuple[int, int]] = []

    flat = np.argsort(dist.flatten())  # ascending distance
    for idx in flat:
        ti, di = divmod(int(idx), n_d)
        if dist[ti, di] > max_dist:
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
        # Re-identification: remember snap positions of dead tracks.
        # Entry: (snap_x, snap_y, track_id, expiry_frame)
        self._dead_snaps: list[tuple] = []
        self._frame_count: int = 0

    def reset(self):
        self._tracks.clear()
        self._dead_snaps.clear()
        self._frame_count = 0
        _Track._id_counter = 1

    def update(
        self,
        detections: list[Detection],
        screen_w: int,
        screen_h: int,
    ) -> list[Detection]:

        # ── 1. Predict all tracks ──────────────────────────────────────────────
        self._frame_count += 1
        for t in self._tracks:
            t.predict()

        # ── 2. Split detections by confidence ─────────────────────────────────
        high_dets = [d for d in detections if d.confidence >= TRACKER_HIGH_CONF]
        low_dets  = [d for d in detections
                     if CONFIDENCE_THRESHOLD <= d.confidence < TRACKER_HIGH_CONF]

        # ── 3. Stage 1: IoU match high-conf dets → ALL tracks ─────────────────
        # TRACKER_IOU_THRESH=0.20: high enough to block cross-target IoU≈0.14,
        # still catches torso-inside-body IoU≈0.43 and frame-to-frame same-size boxes.
        # Head-inside-body (IoU≈0.07-0.12) is NOT matched here; Stage 1b handles it.
        if self._tracks and high_dets:
            iou1 = _iou_matrix(self._tracks, high_dets)
            pairs1, unmatched_t1, unmatched_hd = _greedy_match(iou1, TRACKER_IOU_THRESH)
            for ti, di in pairs1:
                self._tracks[ti].update(high_dets[di])
        else:
            unmatched_t1 = list(range(len(self._tracks)))
            unmatched_hd = list(range(len(high_dets)))

        # ── 3b. Stage 1b: snap-point distance fallback ───────────────────────
        # Catches head-inside-body mismatches that Stage 1 IoU can't handle:
        # head snap_y ≈ 330px, full-body snap_y ≈ 380px → delta ≈ 50px << 160px.
        # Cross-target snap gap ≈ 168px > 160px → blocked.
        if unmatched_t1 and unmatched_hd:
            conf_unm_ti = [ti for ti in unmatched_t1
                           if self._tracks[ti].state == _CONFIRMED]
            if conf_unm_ti:
                sub_tracks = [self._tracks[ti] for ti in conf_unm_ti]
                sub_dets   = [high_dets[di]    for di in unmatched_hd]
                sdist = _snap_dist_matrix(sub_tracks, sub_dets)
                pairs1b, _, _ = _greedy_match_by_dist(sdist, TRACKER_CENTER_DIST_FALLBACK)
                matched_hd_1b: set[int] = set()
                for subi, subdi in pairs1b:
                    orig_ti = conf_unm_ti[subi]
                    orig_di = unmatched_hd[subdi]
                    self._tracks[orig_ti].update(high_dets[orig_di])
                    matched_hd_1b.add(orig_di)
                    unmatched_t1.remove(orig_ti)
                if matched_hd_1b:
                    unmatched_hd = [di for di in unmatched_hd if di not in matched_hd_1b]

        # ── 4. Stage 2: match low-conf dets → remaining CONFIRMED tracks ──────
        # Use snap-dist (not IoU) so that low-conf partial-body dets can still
        # match large-box tracks (head vs full-body IoU is too low for IoU threshold).
        still_unmatched_t = set(unmatched_t1)
        confirmed_remaining = [(ti, self._tracks[ti])
                               for ti in unmatched_t1
                               if self._tracks[ti].state == _CONFIRMED]
        if confirmed_remaining and low_dets:
            rc_tracks = [t for _, t in confirmed_remaining]
            rc_idx    = [i for i, _ in confirmed_remaining]
            sdist2 = _snap_dist_matrix(rc_tracks, low_dets)
            pairs2, _, _ = _greedy_match_by_dist(sdist2, TRACKER_CENTER_DIST_FALLBACK)
            for ti2, di2 in pairs2:
                orig_ti = rc_idx[ti2]
                self._tracks[orig_ti].update_weak(low_dets[di2])
                still_unmatched_t.discard(orig_ti)

        # ── 5. Create new TENTATIVE tracks from unmatched high-conf dets ──────
        # DEDUP using EMA snap_point: compare the SMOOTHED HEAD POSITION stored in
        # each track against the new detection's snap_point.
        # t._snap_ema is a running EMA updated only from real detections — it doesn't
        # drift with Kalman predictions. This ensures:
        # - Same-target duplicate boxes (full-body + head-only) are blocked even when
        #   both arrive in the same frame (active_snaps is updated after each new track).
        # - Distinct targets with different head positions are NOT blocked.
        def _det_snap(d):
            head_h = max((d.y2 - d.y1) * HEAD_ZONE_RATIO, 10.0)
            return (d.x1 + d.x2) * 0.5, d.y1 + head_h * 0.5

        # Seed snap list from EMA of ALL existing tracks
        active_snaps: list = [t._snap_ema for t in self._tracks]
        dedup_sq = TRACKER_DEDUP_DIST ** 2
        reid_sq  = TRACKER_REID_DIST ** 2

        for di in unmatched_hd:
            if high_dets[di].confidence < NEW_TRACK_CONF:
                continue
            det = high_dets[di]
            dsnx, dsny = _det_snap(det)
            too_close = any(
                (sx - dsnx) ** 2 + (sy - dsny) ** 2 < dedup_sq
                for sx, sy in active_snaps
            )
            if too_close:
                continue  # existing track covers this head position — skip

            # Re-identification: if a recently-dead track was at this position,
            # reuse its ID so the target keeps a consistent ID across brief disappearances.
            # Compare detection bbox CENTER against dead track Kalman CENTER (both stable
            # regardless of full-body vs partial-body detection type).
            det_cx = (det.x1 + det.x2) * 0.5
            det_cy = (det.y1 + det.y2) * 0.5
            reuse_id = None
            for idx, (sx, sy, old_id, _exp) in enumerate(self._dead_snaps):
                if (sx - det_cx) ** 2 + (sy - det_cy) ** 2 < reid_sq:
                    reuse_id = old_id
                    self._dead_snaps.pop(idx)
                    break

            new_track = _Track(det, reuse_id=reuse_id)
            self._tracks.append(new_track)
            # Register snap so subsequent dets in THIS SAME FRAME see it
            active_snaps.append((dsnx, dsny))

        # ── 6. Prune dead tracks — record snap positions for re-id first ───────
        # Adaptive max-age strategy:
        # - TENTATIVE tracks: die after MAX_AGE consecutive misses (fast, no ghost boxes)
        # - CONFIRMED tracks: adaptive survival based on scene context + per-track limit
        #   * "camera still on scene" (any Stage-1 match this frame) AND this track's own
        #     consecutive miss_streak < GHOST_MISS_LIMIT (15 frames):
        #     → extend survival to MAX_AGE*10 (≈340ms) so rare YOLO misses don't kill T3/T4
        #   * "camera moved away" OR this track has been missing 15+ consecutive frames:
        #     → short survival MAX_AGE*3 (≈200ms) so ghost boxes disappear quickly
        #
        # GHOST_MISS_LIMIT=15 is the key: a track that LEFT the screen will have its
        # miss_streak increase monotonically (15,16,17...→ max_age=9 → immediate death).
        # A track still IN frame at 30% detection rate hits 15 consecutive misses only
        # 0.47% of the time → REID cleanly resurrects it with same ID.
        any_recently_detected = any(
            t.state == _CONFIRMED and t.miss_streak == 0
            for t in self._tracks
        )

        def _effective_max_age(t) -> int:
            if t.state != _CONFIRMED:
                return TRACKER_MAX_AGE
            if any_recently_detected and t.miss_streak < TRACKER_GHOST_MISS_LIMIT:
                return TRACKER_MAX_AGE * 10  # camera on scene + track still relevant
            return TRACKER_MAX_AGE         # camera away OR track too stale → instant death (~68ms)

        dead: list[_Track] = [
            t for t in self._tracks
            if (t.state == _TENTATIVE and t.miss_streak >= TRACKER_MAX_AGE)
            or (t.state == _CONFIRMED and t.miss_streak >= _effective_max_age(t))
        ]
        for dt in dead:
            # Only remember CONFIRMED tracks for re-id (tentative = unconfirmed false det)
            # Use Kalman CENTER (cx,cy) instead of snap_ema for stable REID matching:
            # snap_ema drifts when YOLO alternates full-body vs partial-body detections,
            # causing >100px position jumps. Kalman center is consistent regardless of bbox type.
            if dt.state == _CONFIRMED:
                self._dead_snaps.append(
                    (float(dt.kf.x[0]), float(dt.kf.x[1]), dt.id,
                     self._frame_count + TRACKER_REID_TTL)
                )
        # Expire old dead snaps
        self._dead_snaps = [
            s for s in self._dead_snaps if s[3] > self._frame_count
        ]

        self._tracks = [t for t in self._tracks if t not in dead]

        # ── 7. Return CONFIRMED tracks sorted by proximity to crosshair ───────
        cx, cy = screen_w // 2, screen_h // 2
        result = [t.to_detection(cx, cy)
                  for t in self._tracks if t.state == _CONFIRMED]
        result.sort(key=lambda d: d.distance_to_center)
        return result
