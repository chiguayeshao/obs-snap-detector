"""
overlay.py — 透明全屏覆盖层

新增显示元素：
  - 头部估算框（黄色虚线）
  - snap 瞄准点红点
  - 吸附圈（屏幕中心白色虚线圆 + 准星）
  - 最近目标用不同颜色高亮
  - FPS 三合一：截帧 / 推理 / 渲染
"""

import tkinter as tk
import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detector import Detection

_TRANSPARENT_COLOR = "#010101"


class Overlay:
    def __init__(self):
        self._root = tk.Tk()
        self._setup_window()
        self._canvas = tk.Canvas(
            self._root,
            bg=_TRANSPARENT_COLOR,
            highlightthickness=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._make_click_through()
        self._root.update_idletasks()
        self._sw = self._root.winfo_screenwidth()
        self._sh = self._root.winfo_screenheight()

    def _setup_window(self):
        r = self._root
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        sw = r.winfo_screenwidth()
        sh = r.winfo_screenheight()
        r.geometry(f"{sw}x{sh}+0+0")
        r.wm_attributes("-transparentcolor", _TRANSPARENT_COLOR)
        r.configure(bg=_TRANSPARENT_COLOR)

    def _make_click_through(self):
        """设置 WS_EX_TRANSPARENT 使鼠标点击穿透覆盖层。"""
        try:
            self._root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self._root.winfo_id())
            if hwnd == 0:
                hwnd = self._root.winfo_id()
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED     = 0x00080000
            GWL_EXSTYLE       = -20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED
            )
        except Exception as e:
            print(f"[Overlay] 鼠标穿透设置失败: {e}")

    def update(
        self,
        detections: "list[Detection]",
        fps_cap: float,
        fps_inf: float,
        fps_ovl: float,
        enabled: bool,
        cap_size: "tuple[int,int] | None" = None,
    ):
        from config import (
            BOX_COLOR, BOX_WIDTH, LABEL_FONT_SIZE,
            PRIMARY_TARGET_COLOR, HEAD_ZONE_COLOR,
            SNAP_POINT_COLOR, SNAP_ZONE_COLOR,
            SNAP_ZONE_RADIUS, SHOW_SNAP_ZONE,
            FPS_COLOR, FPS_FONT_SIZE,
        )

        self._canvas.delete("all")

        if not enabled:
            self._root.update()
            return

        # ── 坐标缩放：将截帧坐标映射到 overlay 逻辑像素 ──
        cap_w = cap_size[0] if cap_size else self._sw
        cap_h = cap_size[1] if cap_size else self._sh
        rx = self._sw / cap_w
        ry = self._sh / cap_h

        def sx(x: int) -> int: return int(x * rx)
        def sy(y: int) -> int: return int(y * ry)

        cx, cy = self._sw // 2, self._sh // 2

        # ── 吸附圈 + 准星 ──────────────────────────────
        if SHOW_SNAP_ZONE and SNAP_ZONE_RADIUS > 0:
            r = SNAP_ZONE_RADIUS
            self._canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=SNAP_ZONE_COLOR, width=1, dash=(4, 6),
            )
            self._canvas.create_line(cx - 12, cy, cx + 12, cy, fill=SNAP_ZONE_COLOR, width=1)
            self._canvas.create_line(cx, cy - 12, cx, cy + 12, fill=SNAP_ZONE_COLOR, width=1)

        # ── 逐目标绘制 ─────────────────────────────────
        for i, det in enumerate(detections):
            is_primary = (i == 0)
            color = PRIMARY_TARGET_COLOR if is_primary else BOX_COLOR
            bw    = BOX_WIDTH + (1 if is_primary else 0)

            self._canvas.create_rectangle(
                sx(det.x1), sy(det.y1), sx(det.x2), sy(det.y2),
                outline=color, width=bw,
            )

            hx1, hy1, hx2, hy2 = det.head_box
            self._canvas.create_rectangle(
                sx(hx1), sy(hy1), sx(hx2), sy(hy2),
                outline=HEAD_ZONE_COLOR, width=1, dash=(3, 3),
            )

            spx, spy = det.snap_point
            spx, spy = sx(spx), sy(spy)
            dot_r    = 3 if is_primary else 2
            self._canvas.create_oval(
                spx - dot_r, spy - dot_r, spx + dot_r, spy + dot_r,
                fill=SNAP_POINT_COLOR, outline=SNAP_POINT_COLOR,
            )

            dist  = int(det.distance_to_center)
            label = f"{det.confidence:.0%}  {dist}px"
            self._canvas.create_text(
                sx(det.x1) + 4, sy(det.y1) - 2,
                text=label, fill=color,
                font=("Consolas", LABEL_FONT_SIZE, "bold"),
                anchor="sw",
            )

        # ── FPS 计数器（右上角）─────────────────────────
        fps_text = (
            f"Cap:{fps_cap:.0f}  Inf:{fps_inf:.0f}  Ovl:{fps_ovl:.0f}"
            f"  |  {len(detections)} targets"
        )
        self._canvas.create_text(
            self._sw - 10, 10,
            text=fps_text, fill=FPS_COLOR,
            font=("Consolas", FPS_FONT_SIZE, "bold"),
            anchor="ne",
        )

        self._root.update()

    def destroy(self):
        try:
            self._root.destroy()
        except Exception:
            pass

    def __init__(self):
        self._root = tk.Tk()
        self._setup_window()
        self._canvas = tk.Canvas(
            self._root,
            bg=_TRANSPARENT_COLOR,
            highlightthickness=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._make_click_through()

    def _setup_window(self):
        root = self._root
        root.overrideredirect(True)             # 无标题栏/边框
        root.attributes("-topmost", True)       # 永远置顶
        root.attributes("-fullscreen", True)    # 全屏覆盖
        root.wm_attributes("-transparentcolor", _TRANSPARENT_COLOR)  # 穿透背景色
        root.configure(bg=_TRANSPARENT_COLOR)

    def _make_click_through(self):
        """
        让覆盖层对鼠标事件透明（点击直接穿透到下层窗口）。
        通过 Windows API 设置 WS_EX_TRANSPARENT 扩展样式。
        """
        try:
            hwnd = ctypes.windll.user32.GetParent(self._root.winfo_id())
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            GWL_EXSTYLE = -20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED
            )
        except Exception as e:
            print(f"[Overlay] 鼠标穿透设置失败（非 Windows？）: {e}")

    def update(self, detections: "list[Detection]", fps: float, enabled: bool):
        """
        清空画布并重绘所有检测框。
        enabled=False 时清空覆盖层（隐藏模式）。
        """
        self._canvas.delete("all")

        if not enabled:
            self._root.update()
            return

        from config import BOX_COLOR, BOX_WIDTH, LABEL_FONT_SIZE, FPS_COLOR, FPS_FONT_SIZE

        # 画每个检测框
        for det in detections:
            # 高亮矩形框
            self._canvas.create_rectangle(
                det.x1, det.y1, det.x2, det.y2,
                outline=BOX_COLOR,
                width=BOX_WIDTH,
            )
            # 置信度标签（框左上角）
            label = f"{det.confidence:.0%}"
            self._canvas.create_text(
                det.x1 + 4, det.y1 - 2,
                text=label,
                fill=BOX_COLOR,
                font=("Consolas", LABEL_FONT_SIZE, "bold"),
                anchor="sw",
            )

        # FPS 计数器（右上角）
        self._canvas.create_text(
            self._root.winfo_screenwidth() - 10,
            10,
            text=f"FPS: {fps:.1f}  |  人数: {len(detections)}",
            fill=FPS_COLOR,
            font=("Consolas", FPS_FONT_SIZE, "bold"),
            anchor="ne",
        )

        self._root.update()

    def destroy(self):
        try:
            self._root.destroy()
        except Exception:
            pass
