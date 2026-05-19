"""
detector.py — YOLOv8 人体检测模块

加载 YOLOv8n 预训练模型（COCO 数据集），
只保留 class=0（person）的检测结果。

检测流水线：
  RGB frame (numpy) → YOLO inference → filter person class → List[Detection]
"""

from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO
from config import CONFIDENCE_THRESHOLD, MODEL_NAME


@dataclass
class Detection:
    """单个检测结果"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class Detector:
    PERSON_CLASS_ID = 0  # COCO 数据集中 person 的 class id

    def __init__(self):
        print(f"[Detector] 加载模型 {MODEL_NAME}（首次运行会自动下载）...")
        self._model = YOLO(MODEL_NAME)
        print("[Detector] 模型加载完成 ✓")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        对一帧图像进行检测，只返回 person 类结果。

        Args:
            frame: RGB numpy array, shape (H, W, 3)

        Returns:
            检测到的人体列表，按置信度降序排列
        """
        results = self._model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            classes=[self.PERSON_CLASS_ID],  # 只检测 person，跳过其他 79 类
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append(Detection(x1, y1, x2, y2, conf))

        # 按置信度降序排列
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
