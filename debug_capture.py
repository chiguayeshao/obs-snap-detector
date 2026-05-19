"""
debug_capture.py — 调试用：截一帧 + 用低置信度跑检测，保存结果图
运行: python debug_capture.py
"""
import time, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

print("等待 3 秒，请切换到游戏画面...")
time.sleep(3)

# ── 截帧 ──────────────────────────────────────────────
try:
    import dxcam
    cam = dxcam.create(output_color="RGB")
    cam.start(video_mode=True, target_fps=30)
    time.sleep(0.5)
    frame = cam.get_latest_frame()
    cam.stop()
    print(f"dxcam 截帧成功: {frame.shape[1]}x{frame.shape[0]}")
except Exception as e:
    print(f"dxcam 失败: {e}，尝试 mss...")
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[1]
        sct_img = sct.grab(mon)
        frame = np.array(Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"))
    print(f"mss 截帧成功: {frame.shape[1]}x{frame.shape[0]}")

# 保存原始截图
Image.fromarray(frame).save("debug_raw.png")
print("原始截图已保存: debug_raw.png")

# ── 检测（低置信度，不限类别）────────────────────────
print("\n运行 YOLOv8n 检测（置信度 0.1，不限类别）...")
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

results = model(frame, conf=0.1, verbose=False)
all_boxes = []
for r in results:
    for box in r.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        name = model.names[cls]
        all_boxes.append((x1,y1,x2,y2,conf,cls,name))

print(f"\n检测到 {len(all_boxes)} 个目标（置信度 > 0.1）：")
for b in sorted(all_boxes, key=lambda x: -x[4]):
    print(f"  class={b[6]:12s} ({b[5]:2d})  conf={b[4]:.2f}  box=({b[0]},{b[1]})-({b[2]},{b[3]})")

# ── 绘制检测框并保存 ─────────────────────────────────
img  = Image.fromarray(frame)
draw = ImageDraw.Draw(img)
colors = {0: "red"}  # person=red, 其他=lime

for x1,y1,x2,y2,conf,cls,name in all_boxes:
    color = "red" if cls == 0 else "lime"
    draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
    draw.text((x1+4, y1+2), f"{name} {conf:.0%}", fill=color)

img.save("debug_detected.png")
print("\n带检测框截图已保存: debug_detected.png")
print("\n请检查两张图：")
print("  debug_raw.png      — 确认截帧内容")
print("  debug_detected.png — 确认检测结果")
