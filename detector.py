"""
detector.py — ONNX + DirectML GPU 加速检测器

推理性能（RTX 4080 Super，游戏运行中）：
  DirectML GPU : ~45 FPS / 22ms  (imgsz=1280)
  CPU PyTorch  : ~7 FPS  / 138ms（后备）

YOLO11 ONNX 输出格式：
  输入 [1, 3, imgsz, imgsz]  → 输出 [1, 84, N]
  前4行: xc, yc, w, h（letterbox 坐标，单位像素）
  后80行: 80个 COCO 类的置信度（class 0 = person）
"""

from dataclasses import dataclass, field
import numpy as np
import cv2

from config import (
    CONFIDENCE_THRESHOLD, NEW_TRACK_CONF,
    MODEL_ONNX_PATH, USE_DIRECTML, NMS_IOU_THRESH,
    MODEL_NAME, DETECT_CLASSES, HEAD_ZONE_RATIO, INFERENCE_IMGSZ, CAPTURE_CROP,
    BOTTOM_STRIP_RATIO, HANDS_CENTER_Y_RATIO, HANDS_BOX_HEIGHT_RATIO,
    HANDS_TOP_EDGE_RATIO,
)


@dataclass
class Detection:
    """单个检测结果，含瞄准点元数据。"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int = -1                              # 由 ByteTracker 设置，用于覆盖层持久化
    _screen_cx: int = field(default=960, repr=False)
    _screen_cy: int = field(default=540, repr=False)

    _snap_dist: float = field(default=-1.0, repr=False)  # smoothed dist from tracker snap_ema; -1 = not set

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def head_box(self) -> tuple[int, int, int, int]:
        head_h = max(int(self.height * HEAD_ZONE_RATIO), 10)
        cx     = (self.x1 + self.x2) // 2
        half_w = max(self.width // 4, 10)
        return (cx - half_w, self.y1, cx + half_w, self.y1 + head_h)

    @property
    def snap_point(self) -> tuple[int, int]:
        hx1, hy1, hx2, hy2 = self.head_box
        return ((hx1 + hx2) // 2, (hy1 + hy2) // 2)

    @property
    def distance_to_center(self) -> float:
        # Prefer tracker's smoothed snap_ema distance (set by to_detection).
        # Falls back to Kalman box snap_point for raw YOLO detections.
        if self._snap_dist >= 0.0:
            return self._snap_dist
        sx, sy = self.snap_point
        return ((sx - self._screen_cx) ** 2 + (sy - self._screen_cy) ** 2) ** 0.5


# ── 纯 NumPy NMS ─────────────────────────────────────────────────────────────
def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """boxes: [N,4] xyxy；返回保留的索引列表。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def _vertical_merge(boxes: np.ndarray, scores: np.ndarray,
                    cx_thresh: float = 50.0,
                    gap_thresh: float = 50.0) -> tuple:
    """
    Merge YOLO body-part boxes for the same person into one hull bbox.

    YOLO often detects head and body as separate boxes where:
      - head:  y1=310, y2=430  (above the body)
      - body:  y1=420, y2=1020 (shoulders to feet)
    Containment NMS fails here because the head sticks ABOVE the body box
    (containment ratio = 8%, not 50%).

    This function groups boxes that share similar horizontal center (< cx_thresh)
    and are vertically adjacent or overlapping (vertical gap < gap_thresh),
    then merges each group into its bounding hull with max confidence.
    Different targets are kept separate because their horizontal centers differ
    by more than cx_thresh pixels.
    """
    n = len(boxes)
    if n <= 1:
        return boxes, scores

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cxs = (boxes[:, 0] + boxes[:, 2]) * 0.5

    for i in range(n):
        for j in range(i + 1, n):
            if abs(cxs[i] - cxs[j]) > cx_thresh:
                continue
            vert_gap = max(
                0.0,
                float(boxes[j, 1]) - float(boxes[i, 3]),
                float(boxes[i, 1]) - float(boxes[j, 3]),
            )
            if vert_gap <= gap_thresh:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out_boxes: list = []
    out_scores: list = []
    for indices in groups.values():
        gb = boxes[indices]
        out_boxes.append([gb[:, 0].min(), gb[:, 1].min(),
                          gb[:, 2].max(), gb[:, 3].max()])
        out_scores.append(float(scores[indices].max()))

    return (np.array(out_boxes, dtype=np.float32),
            np.array(out_scores, dtype=np.float32))


# ── 检测器 ───────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, screen_size: tuple[int, int] = (2560, 1440)):
        self._screen_w, self._screen_h = screen_size
        self._mode = 'pytorch'

        if USE_DIRECTML:
            if self._try_init_directml():
                return
        self._init_pytorch()

    # ── DirectML 初始化 ───────────────────────────────────────────────────────
    def _try_init_directml(self) -> bool:
        try:
            import onnxruntime as ort
            # onnxruntime-directml 1.24.x: try get_available_providers, fallback to direct session
            try:
                avail = [p.lower() for p in ort.get_available_providers()]
                if 'dmlexecutionprovider' not in avail:
                    print("[Detector] WARN: DirectML not available, install onnxruntime-directml")
                    return False
            except AttributeError:
                pass  # older directml version: skip check, try creating session directly
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            self._sess       = ort.InferenceSession(MODEL_ONNX_PATH, providers=providers)
            inp              = self._sess.get_inputs()[0]
            self._input_name = inp.name
            self._imgsz      = inp.shape[2]   # [1, 3, H, W]
            # 预热（DirectML 首次推理有编译延迟）
            dummy = np.zeros((1, 3, self._imgsz, self._imgsz), dtype=np.float32)
            self._sess.run(None, {self._input_name: dummy})
            self._mode = 'directml'
            print(f"[Detector] OK DirectML GPU | input {self._imgsz}x{self._imgsz} "
                  f"| screen {self._screen_w}x{self._screen_h}")
            return True
        except FileNotFoundError:
            print(f"[Detector] WARN: {MODEL_ONNX_PATH} not found, fallback CPU")
            return False
        except Exception as e:
            print(f"[Detector] WARN: DirectML init failed: {e}")
            return False

    # ── PyTorch 后备 ─────────────────────────────────────────────────────────
    def _init_pytorch(self):
        import torch
        from ultralytics import YOLO
        print(f"[Detector] 加载 PyTorch 模型 {MODEL_NAME}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = YOLO(MODEL_NAME)
        self._model.to(device)
        self._mode  = 'pytorch'
        gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
        crop_info = f"crop={CAPTURE_CROP}px " if CAPTURE_CROP else "fullscreen "
        print(f"[Detector] PyTorch {gpu_name} | {crop_info}imgsz={INFERENCE_IMGSZ} FP16 | 屏幕 {self._screen_w}x{self._screen_h}")

    # ── Letterbox 预处理 ─────────────────────────────────────────────────────
    def _letterbox(self, frame: np.ndarray):
        """等比缩放 + 灰色填充至 imgsz×imgsz，返回 (inp, scale, pad_x, pad_y)。"""
        h, w  = frame.shape[:2]
        scale = self._imgsz / max(h, w)
        nh    = int(round(h * scale))
        nw    = int(round(w * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((self._imgsz, self._imgsz, 3), 114, dtype=np.uint8)
        pad_y   = (self._imgsz - nh) // 2
        pad_x   = (self._imgsz - nw) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        inp = canvas.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]   # HWC → NCHW (1,3,H,W)
        return inp, scale, pad_x, pad_y

    # ── ONNX 输出解码 ────────────────────────────────────────────────────────
    def _postprocess(
        self,
        raw: np.ndarray,          # [1, 84, N]
        scale: float,
        pad_x: int, pad_y: int,
        frame_h: int, frame_w: int,
    ) -> list[tuple[np.ndarray, float]]:
        preds = raw[0]            # [84, N]
        person_conf = preds[4, :]  # class 0 (person) 在行索引 4
        mask = person_conf > CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return []
        coords = preds[:4, mask]  # [4, K]  — xc,yc,w,h in letterbox pixels
        scores = person_conf[mask]

        # xywh → xyxy（letterbox 坐标）
        x1 = coords[0] - coords[2] * 0.5
        y1 = coords[1] - coords[3] * 0.5
        x2 = coords[0] + coords[2] * 0.5
        y2 = coords[1] + coords[3] * 0.5
        boxes = np.stack([x1, y1, x2, y2], axis=1)   # [K, 4]

        keep  = _nms(boxes, scores, NMS_IOU_THRESH)
        boxes  = boxes[keep]
        scores = scores[keep]

        # Skip vertical_merge: it was merging separate targets arranged near-to-far
        # (all share similar cx on screen) into a single detection, causing raw:1.
        # Instead, DEDUP_DIST=170px in the tracker blocks head+body duplicate tracks.

        # Undo letterbox → 原始帧坐标
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / scale, 0, frame_w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / scale, 0, frame_h)
        return [(boxes[i].astype(int), float(scores[i])) for i in range(len(boxes))]

    # ── 中心裁剪 ─────────────────────────────────────────────────────────────
    def _crop_center(self, frame: np.ndarray) -> "tuple[np.ndarray, int, int]":
        """若 CAPTURE_CROP > 0，从帧中心裁剪正方形区域，返回 (crop, off_x, off_y)。"""
        if not CAPTURE_CROP or CAPTURE_CROP <= 0:
            return frame, 0, 0
        h, w = frame.shape[:2]
        crop = min(CAPTURE_CROP, h, w)
        x0 = (w - crop) // 2
        y0 = (h - crop) // 2
        return frame[y0:y0 + crop, x0:x0 + crop], x0, y0

    # ── 手部/武器过滤 ────────────────────────────────────────────────────────
    def _is_hand(self, x1: int, y1: int, x2: int, y2: int, frame_h: int) -> bool:
        cy_r = (y1 + y2) / 2 / frame_h
        bh_r = (y2 - y1) / frame_h
        y1_r = y1 / frame_h
        # Rule 1: center is in absolute bottom strip
        if cy_r > BOTTOM_STRIP_RATIO:
            return True
        # Rule 2: box height + center low (combined filter)
        if bh_r > HANDS_BOX_HEIGHT_RATIO and cy_r > HANDS_CENTER_Y_RATIO:
            return True
        # Rule 3: top of bbox is in lower screen portion → ENTIRE box is in lower area
        # Real targets' heads (y1) are always in the upper portion of screen.
        # Player's hands/weapon: y1 (top of arm/gun) starts below 60% screen height.
        # This is safer than y2-based filtering (which wrongly filters tall nearby targets).
        if y1_r > HANDS_TOP_EDGE_RATIO:
            return True
        return False

    # ── 主接口 ───────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> "list[Detection]":
        # 中心裁剪：减少推理区域（CAPTURE_CROP=0则全帧）
        frame_inp, off_x, off_y = self._crop_center(frame)
        h_inp, w_inp = frame_inp.shape[:2]

        if self._mode == 'directml':
            inp, scale, pad_x, pad_y = self._letterbox(frame_inp)
            raw = self._sess.run(None, {self._input_name: inp})[0]
            boxes_scores = self._postprocess(raw, scale, pad_x, pad_y, h_inp, w_inp)
        else:
            import torch
            with torch.inference_mode():
                results = self._model(
                    frame_inp, conf=CONFIDENCE_THRESHOLD,
                    classes=DETECT_CLASSES, imgsz=INFERENCE_IMGSZ,
                    verbose=False, half=True,
                )
            boxes_scores = []
            for result in results:
                for box in result.boxes:
                    bxy  = list(map(int, box.xyxy[0].tolist()))
                    conf = float(box.conf[0])
                    boxes_scores.append((np.array(bxy), conf))

        cx = self._screen_w // 2
        cy = self._screen_h // 2

        detections: list[Detection] = []
        for box, conf in boxes_scores:
            # 先加偏移量转回屏幕坐标，再用屏幕尺寸过滤手部
            x1 = int(box[0]) + off_x
            y1 = int(box[1]) + off_y
            x2 = int(box[2]) + off_x
            y2 = int(box[3]) + off_y
            if self._is_hand(x1, y1, x2, y2, self._screen_h):
                continue
            detections.append(Detection(x1, y1, x2, y2, conf, _screen_cx=cx, _screen_cy=cy))

        detections.sort(key=lambda d: d.distance_to_center)
        return detections

