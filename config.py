"""
config.py — 所有可调参数
修改这里的值来定制检测行为，无需改其他文件。
"""
# 注意：快捷键已更新为 F1=开关  F2=退出  F10=截图

# ── 截帧 ──────────────────────────────────────────────
CAPTURE_REGION  = None     # None = 全屏；或 (left, top, right, bottom)
CAPTURE_MONITOR = 0        # 显示器索引：0 = 主屏幕

# ── AI 检测 ───────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.04   # 绝对过滤阈值：4%兼顾远目标检测（低置信检测进Stage2更新现有轨迹，不创建新轨迹）
NEW_TRACK_CONF        = 0.06   # 创建新轨迹所需最低置信度（6%，允许远目标T3/T4在6%+时创建轨迹）
MODEL_NAME            = "yolo11s.pt"   # CPU 后备模型
INFERENCE_IMGSZ       = 960    # 2K/960 最佳平衡；640最快（CPU 模式）
DETECT_CLASSES        = [0]    # COCO: 0=person

# ── GPU 加速（DirectML + ONNX）────────────────────────
MODEL_ONNX_PATH       = "yolo11s.onnx"  # GPU 推理模型（yolo11s imgsz=960）
USE_DIRECTML          = True   # True=GPU DirectML；False=CPU PyTorch
NMS_IOU_THRESH        = 0.70   # ONNX 后处理 NMS IoU 阈值（提高至0.70允许并排/叠放目标共存）

# ── 跟踪器参数（ByteTrack + Kalman）─────────────────
TRACKER_IOU_THRESH    = 0.20   # IoU 主匹配阈值（提高至0.20：阻止大框抢邻近目标检测(IoU≈0.14)，保留躯干-全身匹配(IoU≈0.43)）
TRACKER_HIGH_CONF     = 0.06   # 高置信分界：6%以下检测进Stage2（只更新现有轨迹），6%以上进Stage1+1b（可创建新轨迹）
TRACKER_MAX_AGE       = 15     # CONFIRMED 轨迹最多允许连续未检测帧数（15帧@42FPS≈360ms：velocity freeze使框冻结原位不漂移，远目标能存活更长）
TRACKER_MIN_HITS      = 1      # =1: 首次检测到即显示（无延迟），防单帧误检靠 NEW_TRACK_CONF
TRACKER_CENTER_DIST_FALLBACK = 160.0  # snap-point Stage 1b 回退匹配阈值（像素）：头部snap差≈50px，不同目标≥168px
TRACKER_DEDUP_DIST   = 170.0           # 去重距离（像素）：同人头部+身体bbox的snap间距≈158px<170px→去重；不同目标snap间距通常>200px不误阻
TRACKER_REID_DIST    = 60.0            # 重识别距离（像素）：已消失轨迹在此距离内重新出现则复用旧ID（60px：同一目标小于此值，相邻目标通常>100px不误识别）
TRACKER_REID_TTL     = 500             # 重识别记忆帧数：记住已消失轨迹500帧≈12秒@42fps，用于远目标重识别（检测间隔5-15秒）
TRACKER_GATE_DIST    = 150.0           # 备用参数（保留兼容）
# 以下保留兼容旧代码
TRACKER_EMA_ALPHA     = 0.25
TRACKER_TTL           = 10
TRACKER_MIN_AGE       = 2
JUMP_SCALE            = 150.0

# ── 自身手部/武器过滤 ─────────────────────────────────
# FPS 游戏中自己的手/武器永远在屏幕下方且面积很大，需过滤掉
# 注意：阈值设置保守，避免误过滤近处大目标（cy_r可达0.65-0.75）
BOTTOM_STRIP_RATIO   = 0.90   # 中心 Y > 此值的检测直接丢弃（绝对底部10%）
HANDS_CENTER_Y_RATIO = 0.80   # 结合高度判断：中心 Y > 此值 且高度 > 下方阈值 → 丢弃（0.80更激进：抓住更多手部误检）
HANDS_BOX_HEIGHT_RATIO = 0.12 # 框高 / 画面高 > 此值 且中心偏下 → 判定为自身手部

# ── 瞄准点 / 头部区域 ────────────────────────────────
HEAD_ZONE_RATIO  = 0.20   # 边界框顶部 20% = 头部区域
SNAP_ZONE_RADIUS = 300    # 屏幕中心吸附圈半径（像素），0 = 不显示
SHOW_SNAP_ZONE   = True

# ── 覆盖层稳定参数 ───────────────────────────────────
OVERLAY_MAX_POOL_SIZE  = 15    # canvas 元素池上限，防长时间运行后画布积累太多项目拖慢渲染
PRIMARY_SWITCH_MARGIN  = 50    # 主目标切换迟滞（像素）：新目标需比当前主目标近50px才切换（snap_ema稳定≈±5px，50px足够防止误切换）
PRIMARY_INHERIT_DIST   = 80    # 主目标继承距离（像素）：旧轨迹消失后，新出现轨迹若在80px内则继承主目标（再识别系统已处理同位置复用，此为兜底）
BOX_SNAP_PX            = 3     # 坐标像素捕捉阈值：变化<3px 不更新画布，消除微抖视觉噪声

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

