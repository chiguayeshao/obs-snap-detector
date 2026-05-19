"""
config.py — 所有可调参数
修改这里的值来定制检测行为，无需改其他文件。
"""

# ── 截帧 ──────────────────────────────────────────────
CAPTURE_REGION  = None     # None = 全屏；或 (left, top, right, bottom)
CAPTURE_MONITOR = 0        # 显示器索引：0 = 主屏幕

# ── AI 检测 ───────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.12   # 低阈值维持已有轨迹（配合 NEW_TRACK_CONF 使用）
NEW_TRACK_CONF        = 0.20   # 创建新轨迹所需最低置信度（防误检闪烁）
MODEL_NAME            = "yolo11s.pt"   # CPU 后备模型
INFERENCE_IMGSZ       = 1280   # 2K最佳；4K用960；640最快（CPU 模式）
DETECT_CLASSES        = [0]    # COCO: 0=person

# ── GPU 加速（DirectML + ONNX）────────────────────────
MODEL_ONNX_PATH       = "yolo11s_1280.onnx"  # GPU 推理模型（imgsz=1280）
USE_DIRECTML          = True   # True=GPU DirectML；False=CPU PyTorch
NMS_IOU_THRESH        = 0.45   # ONNX 后处理 NMS IoU 阈值

# ── 跟踪器参数 ────────────────────────────────────────
TRACKER_IOU_THRESH    = 0.25   # IoU 低于此值视为不同目标
TRACKER_EMA_ALPHA     = 0.25   # 坐标平滑基础系数（自适应会动态调整）
TRACKER_TTL           = 10     # 未匹配后保持显示推理帧数（10帧@50FPS≈0.2s）
TRACKER_MIN_AGE       = 2      # 新目标需连续出现几帧才显示
JUMP_SCALE            = 150.0  # 自适应 EMA 跳变参考距离（像素），越小越激进

# ── 自身手部/武器过滤 ─────────────────────────────────
# FPS 游戏中自己的手/武器永远在屏幕下方且面积很大，需过滤掉
BOTTOM_STRIP_RATIO   = 0.80   # 中心 Y > 此值的检测直接丢弃（绝对底部）
HANDS_CENTER_Y_RATIO = 0.65   # 结合高度判断：中心 Y > 此值 且高度 > 下方阈值 → 丢弃
HANDS_BOX_HEIGHT_RATIO = 0.20 # 框高 / 画面高 > 此值 且中心偏下 → 判定为自身手部

# ── 瞄准点 / 头部区域 ────────────────────────────────
HEAD_ZONE_RATIO  = 0.20   # 边界框顶部 20% = 头部区域
SNAP_ZONE_RADIUS = 300    # 屏幕中心吸附圈半径（像素），0 = 不显示
SHOW_SNAP_ZONE   = True

# ── 覆盖层颜色 ────────────────────────────────────────
BOX_COLOR            = "#00FF41"   # 普通目标框（黑客绿）
PRIMARY_TARGET_COLOR = "#FF4444"   # 最近目标框（红）
HEAD_ZONE_COLOR      = "#FFB700"   # 头部区域框（黄）
SNAP_POINT_COLOR     = "#FF0000"   # snap 瞄准点（红点）
SNAP_ZONE_COLOR      = "#FFFFFF"   # 吸附圈（白虚线）
FPS_COLOR            = "#FFFF00"   # FPS 计数器（黄）
BOX_WIDTH        = 2
LABEL_FONT_SIZE  = 11
FPS_FONT_SIZE    = 13

# ── 性能 ──────────────────────────────────────────────
TARGET_FPS           = 60      # GPU 推理支持 60fps 捕获
INFERENCE_QUEUE_SIZE = 2
DETECTION_QUEUE_SIZE = 2

# ── 快捷键 ────────────────────────────────────────────
TOGGLE_KEY   = "f9"    # 开/关覆盖层
EXIT_KEY     = "esc"   # 退出程序
SNAPSHOT_KEY = "f10"   # 保存带检测框的截图

