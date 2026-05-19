"""
tracker.py  IoU 跟踪 + 自适应 EMA + 距离回退匹配 + 置信度迟滞

改进：
  1. 自适应 EMA alpha：目标大幅跳动时快速跟上（alpha → 0.85），稳定时平滑（alpha → 0.25）
  2. 距离回退匹配：IoU=0 但中心距离够近时仍匹配（处理高速移动目标）
  3. 置信度迟滞：已有轨迹低置信可维持，新轨迹需 new_track_conf（防误检闪烁）
"""

from dataclasses import dataclass
from config import JUMP_SCALE
from detector import Detection


def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_dist(t: "_Track", det: "Detection") -> float:
    tx = (t.x1 + t.x2) * 0.5; ty = (t.y1 + t.y2) * 0.5
    dx = (det.x1 + det.x2) * 0.5; dy = (det.y1 + det.y2) * 0.5
    return ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5


def _adaptive_alpha(base: float, t: "_Track", det: "Detection") -> float:
    """根据跳变距离动态调整 EMA 平滑系数：越远越快响应。"""
    dist = _center_dist(t, det)
    # 线性插值：dist=0 → base，dist=JUMP_SCALE → base+0.5，上限 0.85
    alpha = min(base + dist / JUMP_SCALE * 0.5, 0.85)
    return alpha


@dataclass
class _Track:
    x1: float; y1: float; x2: float; y2: float
    conf: float
    ttl: int
    age: int = 0
    cx: int  = 960
    cy: int  = 540

    def to_detection(self) -> Detection:
        return Detection(
            int(self.x1), int(self.y1), int(self.x2), int(self.y2),
            self.conf, _screen_cx=self.cx, _screen_cy=self.cy,
        )

    def ema_update(self, det: Detection, alpha: float):
        b = 1.0 - alpha
        self.x1   = alpha * det.x1   + b * self.x1
        self.y1   = alpha * det.y1   + b * self.y1
        self.x2   = alpha * det.x2   + b * self.x2
        self.y2   = alpha * det.y2   + b * self.y2
        self.conf = alpha * det.confidence + b * self.conf


class Tracker:
    """
    多策略目标追踪器。

    iou_thresh     IoU 匹配阈值
    ema_alpha      坐标平滑基础系数（越小越平滑）；实际 alpha 由自适应逻辑调整
    ttl            未匹配后保持的推理帧数
    min_age        新目标显示所需连续帧数
    new_track_conf 创建新轨迹的最低置信度（> CONFIDENCE_THRESHOLD，实现迟滞）
    """

    def __init__(
        self,
        iou_thresh:     float = 0.25,
        ema_alpha:      float = 0.25,
        ttl:            int   = 10,
        min_age:        int   = 2,
        new_track_conf: float = 0.20,
    ):
        self.iou_thresh     = iou_thresh
        self.ema_alpha      = ema_alpha
        self.ttl            = ttl
        self.min_age        = min_age
        self.new_track_conf = new_track_conf
        self._tracks: list[_Track] = []

    def reset(self):
        self._tracks.clear()

    def update(
        self,
        detections: list[Detection],
        screen_w: int,
        screen_h: int,
    ) -> list[Detection]:
        cx, cy = screen_w // 2, screen_h // 2
        matched_t: set[int] = set()
        matched_d: set[int] = set()

        # ── 1. IoU 匹配（贪心） ──────────────────────────────────────────────
        for di, det in enumerate(detections):
            best_score, best_ti = self.iou_thresh, -1
            for ti, t in enumerate(self._tracks):
                if ti in matched_t:
                    continue
                score = _iou(t.x1, t.y1, t.x2, t.y2,
                             det.x1, det.y1, det.x2, det.y2)
                if score > best_score:
                    best_score, best_ti = score, ti
            if best_ti >= 0:
                alpha = _adaptive_alpha(self.ema_alpha, self._tracks[best_ti], det)
                self._tracks[best_ti].ema_update(det, alpha)
                self._tracks[best_ti].ttl  = self.ttl
                self._tracks[best_ti].age += 1
                self._tracks[best_ti].cx   = cx
                self._tracks[best_ti].cy   = cy
                matched_t.add(best_ti)
                matched_d.add(di)

        # ── 2. 距离回退匹配（IoU 失败时用中心距离兜底） ─────────────────────
        for di, det in enumerate(detections):
            if di in matched_d:
                continue
            best_dist, best_ti = float('inf'), -1
            for ti, t in enumerate(self._tracks):
                if ti in matched_t:
                    continue
                dist  = _center_dist(t, det)
                # 阈值 = 轨迹框最长边 * 2（允许目标移动了自身尺寸的 2 倍）
                thresh = max(t.x2 - t.x1, t.y2 - t.y1) * 2.0
                if dist < thresh and dist < best_dist:
                    best_dist, best_ti = dist, ti
            if best_ti >= 0:
                alpha = _adaptive_alpha(self.ema_alpha, self._tracks[best_ti], det)
                self._tracks[best_ti].ema_update(det, alpha)
                self._tracks[best_ti].ttl  = self.ttl
                self._tracks[best_ti].age += 1
                self._tracks[best_ti].cx   = cx
                self._tracks[best_ti].cy   = cy
                matched_t.add(best_ti)
                matched_d.add(di)

        # ── 3. 未匹配轨迹：TTL 倒计时 ───────────────────────────────────────
        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                t.ttl -= 1

        # ── 4. 未匹配新检测 → 创建新轨迹（迟滞）──────────────────────────────
        for di, det in enumerate(detections):
            if di not in matched_d and det.confidence >= self.new_track_conf:
                self._tracks.append(_Track(
                    x1=float(det.x1), y1=float(det.y1),
                    x2=float(det.x2), y2=float(det.y2),
                    conf=det.confidence,
                    ttl=self.ttl, age=0, cx=cx, cy=cy,
                ))

        # ── 5. 清理死轨迹 ────────────────────────────────────────────────────
        self._tracks = [t for t in self._tracks if t.ttl > 0]

        # 返回稳定目标（age >= min_age），按距准星距离排序
        stable = [t for t in self._tracks if t.age >= self.min_age]
        result = [t.to_detection() for t in stable]
        result.sort(key=lambda d: d.distance_to_center)
        return result
