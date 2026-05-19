"""
overlay.py — 透明全屏覆盖层

原理：创建一个覆盖全屏的 tkinter 窗口，
设置 transparentcolor 使背景色完全穿透，
只有我们画的线条/文字可见，叠加在任何窗口（包括游戏）上方。

窗口属性：
  - Always-on-top（-topmost True）
  - 无边框（overrideredirect）
  - 背景色 = 透明穿透色
  - 鼠标事件穿透（SetWindowLong WS_EX_TRANSPARENT）
"""

import tkinter as tk
import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detector import Detection

# 与背景色相同的颜色 → 完全透明
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
