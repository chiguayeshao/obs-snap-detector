"""
main.py — 多线程流水线主循环

线程结构：
  CaptureThread   截帧 → frame_queue
  InferenceThread 推理 → detection_queue
  Main Thread     更新覆盖层 + tkinter 事件

快捷键：
  F9  — 开/关覆盖层
  F10 — 保存带检测框的截图（snapshot_YYYYMMDD_HHMMSS.png）
  ESC — 退出
"""

import time
import threading
import queue
import ctypes
import numpy as np

from capture  import ScreenCapturer
from detector import Detector
from overlay  import Overlay
from tracker  import Tracker
from config   import (
    TOGGLE_KEY, EXIT_KEY, SNAPSHOT_KEY,
    TARGET_FPS, INFERENCE_QUEUE_SIZE, DETECTION_QUEUE_SIZE,
    TRACKER_IOU_THRESH, TRACKER_EMA_ALPHA, TRACKER_TTL, TRACKER_MIN_AGE,
    NEW_TRACK_CONF, JUMP_SCALE,
)

# ── 按键检测（GetAsyncKeyState 轮询，无全局钩子）────
_VK = {
    "f9":  0x78, "f10": 0x79, "esc": 0x1B,
    "f1":  0x70, "f2":  0x71, "f3":  0x72,
}

def _key_pressed(name: str) -> bool:
    """返回指定键当前是否被按下（短暂轮询，无钩子注册）。"""
    vk = _VK.get(name.lower())
    if vk is None:
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)

_key_prev: dict[str, bool] = {}

def _key_just_pressed(name: str) -> bool:
    """边沿检测：只在按键从抬起变为按下的那一帧返回 True。"""
    cur = _key_pressed(name)
    prev = _key_prev.get(name, False)
    _key_prev[name] = cur
    return cur and not prev

_enabled          = True
_running          = True
_request_snapshot = False

_fps_lock   = threading.Lock()
_fps_counts = {"cap": 0, "inf": 0, "ovl": 0}

_frame_queue     = queue.Queue(maxsize=INFERENCE_QUEUE_SIZE)
_detection_queue = queue.Queue(maxsize=DETECTION_QUEUE_SIZE)


def _on_toggle():
    global _enabled
    _enabled = not _enabled
    print(f"[Main] 覆盖层 {'开启' if _enabled else '关闭'}")


def _on_exit():
    global _running
    _running = False
    print("[Main] 正在退出...")


def _on_snapshot():
    global _request_snapshot
    _request_snapshot = True


# ── 截帧线程 ─────────────────────────────────────────
def _capture_loop(capturer: ScreenCapturer):
    interval = 1.0 / TARGET_FPS
    while _running:
        t0    = time.perf_counter()
        frame = capturer.grab_frame()
        if frame is not None:
            if _frame_queue.full():
                try: _frame_queue.get_nowait()
                except queue.Empty: pass
            _frame_queue.put_nowait(frame)
            with _fps_lock: _fps_counts["cap"] += 1
        sleep = interval - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)


# ── 推理线程 ─────────────────────────────────────────
def _inference_loop(detector: Detector):
    while _running:
        try:
            frame = _frame_queue.get(timeout=0.1)
            # 排空积压旧帧，始终推理最新帧（消除视角转动时的滞后感）
            while True:
                try:
                    frame = _frame_queue.get_nowait()
                except queue.Empty:
                    break
        except queue.Empty:
            continue
        detections = detector.detect(frame)
        if _detection_queue.full():
            try: _detection_queue.get_nowait()
            except queue.Empty: pass
        _detection_queue.put_nowait((frame, detections))
        with _fps_lock: _fps_counts["inf"] += 1


# ── 主循环 ───────────────────────────────────────────
def main():
    global _request_snapshot

    print("=" * 55)
    print("  OBS Snap Detector  |  学习用途  v3.0 GPU")
    print("=" * 55)

    capturer = ScreenCapturer()
    print(f"[Main] 截帧后端: {capturer.backend}")

    # 获取屏幕分辨率
    import tkinter as _tk
    _r = _tk.Tk(); _r.withdraw()
    sw, sh = _r.winfo_screenwidth(), _r.winfo_screenheight()
    _r.destroy()
    print(f"[Main] 分辨率: {sw}x{sh}")

    detector = Detector(screen_size=(sw, sh))
    overlay  = Overlay()

    print(
        f"[Main] 快捷键: {TOGGLE_KEY.upper()} 开关 "
        f"| {SNAPSHOT_KEY.upper()} 截图 "
        f"| {EXIT_KEY.upper()} 退出"
    )

    last_frame      = None
    last_detections = []
    cap_size        = None
    fps             = {"cap": 0.0, "inf": 0.0, "ovl": 0.0}
    ovl_count       = 0
    fps_timer       = time.perf_counter()
    tracker = Tracker(
        iou_thresh=TRACKER_IOU_THRESH,
        ema_alpha=TRACKER_EMA_ALPHA,
        ttl=TRACKER_TTL,
        min_age=TRACKER_MIN_AGE,
        new_track_conf=NEW_TRACK_CONF,
    )

    with capturer:
        for thr in [
            threading.Thread(target=_capture_loop,   args=(capturer,), daemon=True, name="Capture"),
            threading.Thread(target=_inference_loop, args=(detector,), daemon=True, name="Inference"),
        ]:
            thr.start()

        print("[Main] 运行中... 按 ESC 退出")

        while _running:
            t0 = time.perf_counter()

            # ── 按键轮询（无全局钩子）─────────────────
            if _key_just_pressed(TOGGLE_KEY):
                _on_toggle()
            if _key_just_pressed(SNAPSHOT_KEY):
                _on_snapshot()
            if _key_just_pressed(EXIT_KEY):
                _on_exit()

            # 获取最新检测结果，用 tracker 平滑坐标
            try:
                last_frame, raw_detections = _detection_queue.get_nowait()
                if cap_size is None and last_frame is not None:
                    cap_size = (last_frame.shape[1], last_frame.shape[0])
                    if cap_size != (sw, sh):
                        print(f"[Main] 截帧分辨率 {cap_size[0]}x{cap_size[1]} "
                              f"↓ 缩放至覆盖层 {sw}x{sh}")
                last_detections = tracker.update(raw_detections, sw, sh)
            except queue.Empty:
                pass

            overlay.update(last_detections, fps["cap"], fps["inf"], fps["ovl"], _enabled, cap_size)
            ovl_count += 1

            # 截图请求
            if _request_snapshot:
                _request_snapshot = False
                _save_snapshot(last_frame, last_detections)

            # 每秒更新 FPS
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                with _fps_lock:
                    counts = dict(_fps_counts)
                    _fps_counts.update({"cap": 0, "inf": 0, "ovl": 0})
                fps = {k: counts[k] / elapsed for k in counts}
                fps["ovl"] = ovl_count / elapsed
                ovl_count  = 0
                fps_timer  = time.perf_counter()
                print(
                    f"[Main] Cap:{fps['cap']:.0f}  Inf:{fps['inf']:.0f}  "
                    f"Ovl:{fps['ovl']:.0f}  | {len(last_detections)} 目标  "
                    f"| {'🟢' if _enabled else '🔴'}"
                )

            spent = time.perf_counter() - t0
            sleep = (1.0 / TARGET_FPS) - spent
            if sleep > 0:
                time.sleep(sleep)

    overlay.destroy()
    print("[Main] 已退出")


def _save_snapshot(frame, detections):
    """将带检测框的截图保存到当前目录。"""
    if frame is None:
        print("[Snapshot] 无可用帧，跳过")
        return
    try:
        from PIL import Image, ImageDraw
        import datetime
        from config import BOX_COLOR, PRIMARY_TARGET_COLOR, HEAD_ZONE_COLOR

        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        for i, det in enumerate(detections):
            color = tuple(int(PRIMARY_TARGET_COLOR[j:j+2], 16) for j in (1, 3, 5)) if i == 0 \
                    else tuple(int(BOX_COLOR[j:j+2], 16) for j in (1, 3, 5))
            head_c = tuple(int(HEAD_ZONE_COLOR[j:j+2], 16) for j in (1, 3, 5))
            draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=color, width=2)
            draw.rectangle(list(det.head_box), outline=head_c, width=1)
            sx, sy = det.snap_point
            draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(255, 0, 0))
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"snapshot_{ts}.png"
        img.save(fname)
        print(f"[Snapshot] 已保存: {fname}")
    except Exception as e:
        print(f"[Snapshot] 保存失败: {e}")


if __name__ == "__main__":
    main()

