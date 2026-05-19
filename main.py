"""
main.py — 主循环

流水线：
  ScreenCapturer.grab_frame()
      → Detector.detect(frame)
          → Overlay.update(detections, fps)

快捷键：
  F9  — 开/关 覆盖层
  ESC — 退出程序
"""

import time
import threading
import keyboard
from capture import ScreenCapturer
from detector import Detector
from overlay import Overlay
from config import TOGGLE_KEY, EXIT_KEY, TARGET_FPS

# ── 全局状态 ──────────────────────────────────────────
_enabled = True   # 覆盖层开关
_running = True   # 主循环开关


def _on_toggle():
    global _enabled
    _enabled = not _enabled
    state = "开启" if _enabled else "关闭"
    print(f"[Main] 覆盖层已{state}")


def _on_exit():
    global _running
    print("[Main] 退出...")
    _running = False


def _register_hotkeys():
    keyboard.add_hotkey(TOGGLE_KEY, _on_toggle)
    keyboard.add_hotkey(EXIT_KEY, _on_exit)
    print(f"[Main] 快捷键已注册 — {TOGGLE_KEY.upper()} 开关 / {EXIT_KEY.upper()} 退出")


def main():
    print("=" * 50)
    print("  OBS Snap Detector  |  学习用途")
    print("=" * 50)

    capturer = ScreenCapturer()
    detector = Detector()
    overlay = Overlay()

    _register_hotkeys()

    # 键盘监听在后台线程运行
    hotkey_thread = threading.Thread(
        target=keyboard.wait,
        daemon=True,
    )
    hotkey_thread.start()

    frame_interval = 1.0 / TARGET_FPS
    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.perf_counter()

    print("[Main] 启动截帧... 按 F9 切换覆盖层，ESC 退出")

    with capturer:
        while _running:
            t0 = time.perf_counter()

            frame = capturer.grab_frame()

            if frame is None:
                # dxcam 暂无新帧，稍等
                time.sleep(0.005)
                continue

            detections = detector.detect(frame) if _enabled else []
            overlay.update(detections, fps_display, _enabled)

            # FPS 计算（每秒更新一次显示值）
            fps_counter += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.perf_counter()
                print(
                    f"[Main] FPS: {fps_display:.1f}  |  "
                    f"检测到 {len(detections)} 人  |  "
                    f"{'🟢 开启' if _enabled else '🔴 关闭'}"
                )

            # 限速，避免空转占满 CPU
            spent = time.perf_counter() - t0
            sleep_time = frame_interval - spent
            if sleep_time > 0:
                time.sleep(sleep_time)

    overlay.destroy()
    print("[Main] 已退出")


if __name__ == "__main__":
    main()
