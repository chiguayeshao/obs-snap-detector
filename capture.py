"""
capture.py — 屏幕截帧模块

优先使用 dxcam（DXGI Desktop Duplication，与 OBS 同 API）；
如果 dxcam 初始化失败，自动回退到 mss（GDI，兼容性更好）。
"""

import numpy as np
from config import CAPTURE_REGION, TARGET_FPS, CAPTURE_MONITOR


class ScreenCapturer:
    def __init__(self):
        self._region  = CAPTURE_REGION
        self._started = False
        self._backend = None
        self._camera  = None
        self._mss_sct = None
        self._init_camera()

    # ── 初始化 ────────────────────────────────────────
    def _init_camera(self):
        """尝试 dxcam，失败则回退到 mss。"""
        try:
            import dxcam
            self._camera  = dxcam.create(device_idx=CAPTURE_MONITOR, output_color="RGB")
            self._backend = "dxcam"
            print("[Capture] backend: dxcam (DXGI)")
        except Exception as e:
            print(f"[Capture] dxcam 初始化失败: {e}")
            print("[Capture] 回退到 mss (GDI) ...")
            try:
                import mss
                self._mss_sct = mss.mss()
                self._backend = "mss"
                print("[Capture] backend: mss (GDI)")
            except Exception as e2:
                raise RuntimeError(
                    f"所有截帧后端均失败:\n  dxcam: {e}\n  mss: {e2}"
                )

    # ── 生命周期 ──────────────────────────────────────
    def start(self):
        if not self._started:
            if self._backend == "dxcam":
                self._camera.start(
                    region=self._region,
                    target_fps=TARGET_FPS,
                    video_mode=True,
                )
            self._started = True

    def stop(self):
        if self._started:
            if self._backend == "dxcam":
                self._camera.stop()
            elif self._backend == "mss" and self._mss_sct:
                self._mss_sct.close()
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── 截帧 ──────────────────────────────────────────
    def grab_frame(self) -> "np.ndarray | None":
        """返回最新一帧 RGB numpy array (H, W, 3)，无新帧则返回 None。"""
        if self._backend == "dxcam":
            return self._camera.get_latest_frame()
        elif self._backend == "mss":
            return self._grab_mss()
        return None

    def _grab_mss(self) -> "np.ndarray | None":
        from PIL import Image
        monitors = self._mss_sct.monitors
        mon_idx  = CAPTURE_MONITOR + 1
        monitor  = monitors[mon_idx] if mon_idx < len(monitors) else monitors[1]
        if self._region:
            l, t, r, b = self._region
            monitor = {"left": l, "top": t, "width": r - l, "height": b - t}
        sct = self._mss_sct.grab(monitor)
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        return np.array(img)

    @property
    def backend(self) -> str:
        return self._backend or "none"

