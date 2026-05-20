"""
overlay.py — Persistent-item tkinter overlay (zero-flicker design)

Anti-flicker principle:
  NEVER call canvas.delete("all").
  Each confirmed track owns a set of canvas item IDs keyed by track_id.
  Frame updates use canvas.coords() + canvas.itemconfig() to MOVE items in-place.
  When a track disappears, its items are hidden with state='hidden' (not deleted).
  Items are reused from a free pool when a new track appears.

Three-layer flicker elimination:
  1. Pixel snap   — skip canvas.coords() update if movement < BOX_SNAP_PX pixels
  2. Primary hysteresis — only switch "nearest target" colour if new target is
                          PRIMARY_SWITCH_MARGIN px closer (prevents rapid red/green swap)
  3. Pool cap     — limit free pool to OVERLAY_MAX_POOL_SIZE to prevent canvas
                    item accumulation that degrades FPS over time
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
        self._exclude_from_capture()
        self._root.update_idletasks()
        self._sw = self._root.winfo_screenwidth()
        self._sh = self._root.winfo_screenheight()

        # track_id → {box, head, dot, label}
        self._track_items: dict[int, dict[str, int]] = {}

        # Reusable pool of hidden item groups (avoids create_* on every new track)
        self._free_pool: list[dict[str, int]] = []

        # Primary target: track ID of the nearest detected target
        self._primary_tid: int = -1

        # Stable primary-tracking state:
        # _primary_last_dist — snap_ema distance of primary when last seen
        # _primary_lost_frames — consecutive frames since primary track disappeared
        self._primary_last_dist: float = float('inf')
        self._primary_lost_frames: int = 0

        # Last drawn box coords per track — pixel-snap avoids canvas update on tiny movement
        self._last_box: dict[int, tuple[int, int, int, int]] = {}

        # Last drawn label text — skip itemconfig if label hasn't changed
        self._last_label: dict[int, str] = {}

        # Static items — created once, updated in-place
        self._init_static_items()

    def _init_static_items(self):
        from config import SNAP_ZONE_RADIUS, SNAP_ZONE_COLOR, FPS_COLOR, FPS_FONT_SIZE
        cx, cy = self._sw // 2, self._sh // 2
        r = SNAP_ZONE_RADIUS or 200

        self._snap_circle = self._canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=SNAP_ZONE_COLOR, width=1, dash=(4, 6), state='hidden',
        )
        self._ch_h = self._canvas.create_line(
            cx - 14, cy, cx + 14, cy,
            fill=SNAP_ZONE_COLOR, width=1, state='hidden',
        )
        self._ch_v = self._canvas.create_line(
            cx, cy - 14, cx, cy + 14,
            fill=SNAP_ZONE_COLOR, width=1, state='hidden',
        )
        self._fps_item = self._canvas.create_text(
            self._sw - 10, 10,
            text="", fill=FPS_COLOR,
            font=("Consolas", FPS_FONT_SIZE, "bold"),
            anchor="ne", state='hidden',
        )

    # ── Window setup ──────────────────────────────────────────────────────────

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
        """Set WS_EX_TRANSPARENT so mouse clicks pass through the overlay."""
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
                hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED,
            )
        except Exception as e:
            print(f"[Overlay] click-through setup failed: {e}")

    def _exclude_from_capture(self):
        """Prevent dxcam/DXGI from capturing this overlay window.

        WDA_EXCLUDEFROMCAPTURE (0x11) marks the window as invisible to any
        screen-capture API (DXGI Desktop Duplication, WinRT, BitBlt, etc.).
        Without this, dxcam captures the green/red boxes drawn on screen and
        feeds them into every YOLO inference frame. The overlay rectangles
        drawn over real targets alter their visual appearance, degrading YOLO
        detection quality and causing unstable track confidence for T3/T4.

        Reference: sunone_aimbot_2 overlay_exclude_from_capture config option
        (SetWindowDisplayAffinity with WDA_EXCLUDEFROMCAPTURE).
        Requires Windows 10 version 2004 (Build 19041) or later.
        """
        try:
            self._root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self._root.winfo_id())
            if hwnd == 0:
                hwnd = self._root.winfo_id()
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            result = ctypes.windll.user32.SetWindowDisplayAffinity(
                hwnd, WDA_EXCLUDEFROMCAPTURE
            )
            if result:
                print("[Overlay] Excluded from screen capture (WDA_EXCLUDEFROMCAPTURE)")
            else:
                err = ctypes.windll.kernel32.GetLastError()
                print(f"[Overlay] WDA_EXCLUDEFROMCAPTURE failed (err={err})"
                      f" — overlay visible to dxcam, may affect T3/T4 detection")
        except Exception as e:
            print(f"[Overlay] exclude-from-capture error: {e}")

    # ── Canvas item pool ──────────────────────────────────────────────────────

    def _new_item_group(self) -> dict[str, int]:
        """Create a new hidden group of canvas items for one track."""
        from config import BOX_COLOR, BOX_WIDTH, LABEL_FONT_SIZE, HEAD_ZONE_COLOR, SNAP_POINT_COLOR
        return {
            'box':   self._canvas.create_rectangle(
                         0, 0, 1, 1, outline=BOX_COLOR, width=BOX_WIDTH, state='hidden'),
            'head':  self._canvas.create_rectangle(
                         0, 0, 1, 1, outline=HEAD_ZONE_COLOR, width=1,
                         dash=(3, 3), state='hidden'),
            'dot':   self._canvas.create_oval(
                         0, 0, 1, 1, fill=SNAP_POINT_COLOR,
                         outline=SNAP_POINT_COLOR, state='hidden'),
            'label': self._canvas.create_text(
                         0, 0, text="", fill=BOX_COLOR,
                         font=("Consolas", LABEL_FONT_SIZE, "bold"),
                         anchor="sw", state='hidden'),
        }

    def _acquire_item_group(self) -> dict[str, int]:
        """Get a free item group (from pool or newly created)."""
        if self._free_pool:
            return self._free_pool.pop()
        return self._new_item_group()

    def _release_item_group(self, items: dict[str, int]):
        """Return an item group to the free pool (hidden). Cap pool to avoid canvas bloat."""
        from config import OVERLAY_MAX_POOL_SIZE
        cv = self._canvas
        if len(self._free_pool) >= OVERLAY_MAX_POOL_SIZE:
            # Pool full — delete canvas items to prevent accumulation over long sessions
            for item_id in items.values():
                cv.delete(item_id)
        else:
            for item_id in items.values():
                cv.itemconfig(item_id, state='hidden')
            self._free_pool.append(items)

    # ── Main update ───────────────────────────────────────────────────────────

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
            BOX_COLOR, BOX_WIDTH,
            PRIMARY_TARGET_COLOR, HEAD_ZONE_COLOR,
            SNAP_POINT_COLOR, SNAP_ZONE_COLOR,
            SNAP_ZONE_RADIUS, SHOW_SNAP_ZONE, FPS_COLOR,
            BOX_SNAP_PX,
        )

        cv = self._canvas

        if not enabled:
            # Hide everything without deleting
            cv.itemconfig(self._snap_circle, state='hidden')
            cv.itemconfig(self._ch_h,        state='hidden')
            cv.itemconfig(self._ch_v,        state='hidden')
            cv.itemconfig(self._fps_item,    state='hidden')
            for tid in list(self._track_items):
                self._release_item_group(self._track_items.pop(tid))
                self._last_box.pop(tid, None)
                self._last_label.pop(tid, None)
            self._primary_tid = -1
            self._primary_last_dist = float('inf')
            self._primary_lost_frames = 0
            self._root.update()
            return

        # Coordinate scaling: capture res → screen (overlay) res
        cap_w = cap_size[0] if cap_size else self._sw
        cap_h = cap_size[1] if cap_size else self._sh
        rx = self._sw / cap_w
        ry = self._sh / cap_h

        def sx(x: int) -> int: return int(x * rx)
        def sy(y: int) -> int: return int(y * ry)

        cx_s, cy_s = self._sw // 2, self._sh // 2

        # ── Static items: snap zone + crosshair ──────────────────────────────
        if SHOW_SNAP_ZONE and SNAP_ZONE_RADIUS > 0:
            r = SNAP_ZONE_RADIUS
            cv.coords(self._snap_circle, cx_s - r, cy_s - r, cx_s + r, cy_s + r)
            cv.itemconfig(self._snap_circle,
                          state='normal', outline=SNAP_ZONE_COLOR)
            cv.coords(self._ch_h, cx_s - 14, cy_s, cx_s + 14, cy_s)
            cv.itemconfig(self._ch_h, state='normal')
            cv.coords(self._ch_v, cx_s, cy_s - 14, cx_s, cy_s + 14)
            cv.itemconfig(self._ch_v, state='normal')
        else:
            cv.itemconfig(self._snap_circle, state='hidden')
            cv.itemconfig(self._ch_h,        state='hidden')
            cv.itemconfig(self._ch_v,        state='hidden')

        # ── FPS counter ───────────────────────────────────────────────────────
        fps_text = (f"Cap:{fps_cap:.0f}  Inf:{fps_inf:.0f}  Ovl:{fps_ovl:.0f}"
                    f"  |  {len(detections)} targets")
        cv.itemconfig(self._fps_item, text=fps_text, state='normal')

        # ── Release items for tracks that disappeared ─────────────────────────
        active_ids = {d.track_id for d in detections}
        for tid in [k for k in self._track_items if k not in active_ids]:
            self._release_item_group(self._track_items.pop(tid))
            self._last_box.pop(tid, None)
            self._last_label.pop(tid, None)

        # ── Primary target: stable nearest-track selection with hold & advantage ──
        # Problem: T2's track briefly dies (3-frame YOLO miss) → T3/T4 momentarily
        # become nearest → red box sweeps. Fix: two-layer protection:
        #   1. Adaptive max-age (tracker.py): established tracks survive 3× longer.
        #   2. Hold + advantage here: when primary disappears, hold for N frames AND
        #      require new nearest to be PRIMARY_ADVANTAGE_PX closer than old primary was.
        from config import (
            PRIMARY_SWITCH_MARGIN, PRIMARY_HOLD_FRAMES, PRIMARY_ADVANTAGE_PX,
        )
        prev_primary = self._primary_tid
        nearest = detections[0] if detections else None
        cur = next((d for d in detections if d.track_id == self._primary_tid), None)

        if cur is not None:
            # Primary is still alive — update its last-known distance.
            # Switch only if something clearly closer appeared (PRIMARY_SWITCH_MARGIN px).
            self._primary_lost_frames = 0
            self._primary_last_dist = cur.distance_to_center
            if nearest is not None and nearest.distance_to_center < cur.distance_to_center - PRIMARY_SWITCH_MARGIN:
                self._primary_tid = nearest.track_id
                self._primary_last_dist = nearest.distance_to_center
        elif nearest is not None:
            # Primary track disappeared. Don't immediately hand over to another target:
            # the new nearest must be significantly closer than old primary WAS, OR
            # we must have waited PRIMARY_HOLD_FRAMES frames already.
            self._primary_lost_frames += 1
            new_dist = nearest.distance_to_center
            if (new_dist < self._primary_last_dist - PRIMARY_ADVANTAGE_PX
                    or self._primary_lost_frames >= PRIMARY_HOLD_FRAMES):
                self._primary_tid = nearest.track_id
                self._primary_last_dist = new_dist
                self._primary_lost_frames = 0
            # else: keep old primary_tid — no red box shown this frame (track gone)
        else:
            self._primary_tid = -1
            self._primary_last_dist = float('inf')
            self._primary_lost_frames = 0

        # ── Update / create items for active tracks ───────────────────────────
        for det in detections:
            is_primary = (det.track_id == self._primary_tid)
            color      = PRIMARY_TARGET_COLOR if is_primary else BOX_COLOR
            bw         = BOX_WIDTH + (1 if is_primary else 0)
            dot_r      = 4 if is_primary else 2

            is_new = det.track_id not in self._track_items
            if is_new:
                self._track_items[det.track_id] = self._acquire_item_group()
            items = self._track_items[det.track_id]

            # ── Pixel snap: skip coord updates for tiny movement ──────────────
            bx1, by1 = sx(det.x1), sy(det.y1)
            bx2, by2 = sx(det.x2), sy(det.y2)
            last = self._last_box.get(det.track_id)
            coords_moved = (
                last is None
                or abs(bx1 - last[0]) > BOX_SNAP_PX
                or abs(by1 - last[1]) > BOX_SNAP_PX
                or abs(bx2 - last[2]) > BOX_SNAP_PX
                or abs(by2 - last[3]) > BOX_SNAP_PX
            )
            if coords_moved:
                # Update ALL coord-dependent items together (box, head, dot, label)
                cv.coords(items['box'],  bx1, by1, bx2, by2)
                hx1, hy1, hx2, hy2 = det.head_box
                cv.coords(items['head'], sx(hx1), sy(hy1), sx(hx2), sy(hy2))
                spx, spy = sx(det.snap_point[0]), sy(det.snap_point[1])
                cv.coords(items['dot'],
                          spx - dot_r, spy - dot_r,
                          spx + dot_r, spy + dot_r)
                cv.coords(items['label'], bx1 + 4, by1 - 2)
                self._last_box[det.track_id] = (bx1, by1, bx2, by2)

            # ── Color/outline: only update when primary status changes ─────────
            primary_changed = (
                (det.track_id == self._primary_tid) !=
                (det.track_id == prev_primary)
            )
            if is_new or primary_changed:
                cv.itemconfig(items['box'],  outline=color, width=bw, state='normal')
                cv.itemconfig(items['head'], outline=HEAD_ZONE_COLOR, state='normal')
                cv.itemconfig(items['dot'],  fill=SNAP_POINT_COLOR,
                              outline=SNAP_POINT_COLOR, state='normal')

            # ── Label: rounded values + cache to skip unchanged text ──────────
            # Round confidence to nearest 10%, distance to nearest 30px
            # → label changes only when crossing threshold, not every frame
            conf_r = int(round(det.confidence / 0.10) * 10)
            dist_r = int(round(det.distance_to_center / 30) * 30)
            if is_primary:
                label = f"{conf_r}%  {dist_r}px"
            else:
                label = f"{dist_r}px"

            label_changed = label != self._last_label.get(det.track_id)
            if is_new or label_changed or primary_changed:
                cv.itemconfig(items['label'], text=label, fill=color, state='normal')
                self._last_label[det.track_id] = label

        self._root.update()

    def destroy(self):
        try:
            self._root.destroy()
        except Exception:
            pass

