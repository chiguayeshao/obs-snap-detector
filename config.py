"""
config.py — 所有可调参数
修改这里的值来定制检测行为，无需改其他文件。
"""
# 注意：快捷键已更新为 F1=开关  F2=退出  F10=截图

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

# ── 跟踪器参数（ByteTrack + Kalman）─────────────────
TRACKER_IOU_THRESH    = 0.25   # IoU 匹配阈值（低于此值不匹配）
TRACKER_HIGH_CONF     = 0.25   # 第一阶段匹配用高置信度阈值
TRACKER_MAX_AGE       = 5      # CONFIRMED 轨迹最多允许连续未检测帧数（5帧@60FPS≈83ms）
TRACKER_MIN_HITS      = 1      # =1: 首次检测到即显示（无延迟），防单帧误检靠 NEW_TRACK_CONF
# 以下保留兼容旧代码
TRACKER_EMA_ALPHA     = 0.25
TRACKER_TTL           = 10
TRACKER_MIN_AGE       = 2
JUMP_SCALE            = 150.0

# ── 自身手部/武器过滤 ─────────────────────────────────
# FPS 游戏中自己的手/武器永远在屏幕下方且面积很大，需过滤掉
BOTTOM_STRIP_RATIO   = 0.80   # 中心 Y > 此值的检测直接丢弃（绝对底部）
HANDS_CENTER_Y_RATIO = 0.65   # 结合高度判断：中心 Y > 此值 且高度 > 下方阈值 → 丢弃
HANDS_BOX_HEIGHT_RATIO = 0.20 # 框高 / 画面高 > 此值 且中心偏下 → 判定为自身手部

# ── 瞄准点 / 头部区域 ────────────────────────────────
HEAD_ZONE_RATIO  = 0.20   # 边界框顶部 20% = 头部区域
SNAP_ZONE_RADIUS = 300    # 屏幕中心吸附圈半径（像素），0 = 不显示
SHOW_SNAP_ZONE   = True

# ── 覆盖层稳定参数 ───────────────────────────────────
OVERLAY_MAX_POOL_SIZE  = 15    # canvas 元素池上限，防长时间运行后画布积累太多项目拖慢渲染
PRIMARY_SWITCH_MARGIN  = 80    # 主目标切换迟滞（像素）：新目标需比当前主目标近80px才切换，防颜色闪烁
BOX_SNAP_PX            = 2     # 坐标像素捕捉阈值：变化<2px 不更新画布，消除微抖视觉噪声

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
TOGGLE_KEY   = "f1"    # 开/关覆盖层（游戏中按 F1）
EXIT_KEY     = "f2"    # 退出程序（游戏中按 F2）
SNAPSHOT_KEY = "f10"   # 保存带检测框的截图

