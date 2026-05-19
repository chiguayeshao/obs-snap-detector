"""
capture.py — 屏幕截帧模块

使用 dxcam（封装 DXGI Desktop Duplication API），
这与 OBS Studio 的 "Display Capture" 后端使用的是同一套 Windows API。
原理：通过 IDXGIOutputDuplication 在 GPU 层面拷贝帧缓冲，
无需经过 GDI，延迟极低（<1ms）。
"""

import numpy as np
import dxcam
from config import CAPTURE_REGION, TARGET_FPS


class ScreenCapturer:
    def __init__(self):
        # dxcam 默认使用 GPU 0，output 0（主显示器）
        self._camera = dxcam.create(output_color="RGB")
        self._region = CAPTURE_REGION  # None 或 (left, top, right, bottom)
        self._started = False

    def start(self):
        """启动连续截帧模式（比单次 grab 延迟更低）"""
        if not self._started:
            self._camera.start(
                region=self._region,
                target_fps=TARGET_FPS,
                video_mode=True,   # 始终有帧输出，不等待屏幕变化
            )
            self._started = True

    def grab_frame(self) -> np.ndarray | None:
        """
        获取最新一帧，返回 RGB numpy array (H, W, 3)。
        如果当前无新帧则返回 None。
        """
        frame = self._camera.get_latest_frame()
        return frame  # 已经是 RGB ndarray，无需转换

    def stop(self):
        if self._started:
            self._camera.stop()
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
