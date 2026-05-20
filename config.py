"""
config.py — 所有可调参数
修改这里的值来定制检测行为，无需改其他文件。
"""
# 注意：快捷键已更新为 F1=开关  F2=退出  F10=截图

# ── 截帧 ──────────────────────────────────────────────
CAPTURE_REGION  = None     # None = 全屏；或 (left, top, right, bottom)
CAPTURE_MONITOR = 0        # 显示器索引：0 = 主屏幕
CAPTURE_CROP    = 1920     # 中心裁剪尺寸（像素，正方形）：4K下取中心1920×1920，0=不裁剪
                           # 作用：减少无关区域，降低推理量；与FP16+imgsz=640配合可达100fps+

# ── AI 检测 ───────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.10   # 硬过滤：10%（参考sunone track_low_thresh=0.10）
                                # 低于此值直接丢弃，不进跟踪器，消除环境噪声假检测
NEW_TRACK_CONF        = 0.20   # 创建新轨迹最低置信度（参考sunone new_track_thresh=0.25）
                                # 低于此值的检测只能更新已有轨迹(Stage2)，不能创建新轨迹
MODEL_NAME            = "yolo11m.pt"   # CPU 后备模型
INFERENCE_IMGSZ       = 640    # 与中心裁剪1920配合：scale=1/3，目标20px，FP16下约100fps
DETECT_CLASSES        = [0]    # COCO: 0=person

# ── GPU 加速（DirectML + ONNX）────────────────────────
MODEL_ONNX_PATH       = "yolo11m_1280.onnx"  # GPU 推理模型（yolo11m imgsz=1280，4K分辨率优化，~45fps）
USE_DIRECTML          = True   # True=GPU DirectML；False=CPU PyTorch
NMS_IOU_THRESH        = 0.70   # ONNX 后处理 NMS IoU 阈值（提高至0.70允许并排/叠放目标共存）

# ── 跟踪器参数（ByteTrack + Kalman）─────────────────
TRACKER_IOU_THRESH    = 0.20   # IoU 主匹配阈值（提高至0.20：阻止大框抢邻近目标检测(IoU≈0.14)，保留躯干-全身匹配(IoU≈0.43)）
TRACKER_HIGH_CONF     = 0.20   # 高置信分界：20%以下检测进Stage2（只更新现有轨迹），20%以上进Stage1（可创建新轨迹）
TRACKER_MAX_AGE       = 3      # CONFIRMED 轨迹最多允许连续未检测帧数（3帧@42FPS≈70ms：视角转开时边框立即消失，避免残影≈1秒问题）
TRACKER_MIN_HITS      = 2      # =2: 需要连续2帧检测才显示边框，过滤单帧噪声（防止T3/T4低置信单帧检测产生的扫描闪烁效果）
TRACKER_CENTER_DIST_FALLBACK = 160.0  # snap-point Stage 1b 回退匹配阈值（像素）：头部snap差≈50px，不同目标≥168px
TRACKER_DEDUP_DIST   = 170.0           # 去重距离（像素）：同人头部+身体bbox的snap间距≈158px<170px→去重；不同目标snap间距通常>200px不误阻
TRACKER_REID_DIST    = 100.0           # 重识别距离（像素）：已消失轨迹在此距离内重新出现则复用旧ID（100px：同一目标小于此值，相邻目标通常>200px不误识别）
TRACKER_REID_TTL     = 500             # 重识别记忆帧数：记住已消失轨迹500帧≈12秒@42fps，用于远目标重识别（检测间隔5-15秒）
TRACKER_GATE_DIST    = 150.0           # 备用参数（保留兼容）
TRACKER_GHOST_MISS_LIMIT = 4          # 幽灵框限制：连续4帧（~90ms@44fps）没有Stage1匹配就强制死亡
                                       # T1/T2(75%+检测率)连续4帧miss概率<0.4%→基本不受影响；离开画面目标快速消亡
# 以下保留兼容旧代码
TRACKER_EMA_ALPHA     = 0.25
TRACKER_TTL           = 10
TRACKER_MIN_AGE       = 2
JUMP_SCALE            = 150.0

# ── 自身手部/武器过滤 ─────────────────────────────────
# FPS 游戏中自己的手/武器永远在屏幕下方，需过滤掉
BOTTOM_STRIP_RATIO    = 0.90   # 中心 Y > 此值直接丢弃（绝对底部10%）
HANDS_CENTER_Y_RATIO  = 0.72   # 结合高度判断：中心 Y > 此值 且高度 > 下方阈值 → 丢弃
HANDS_BOX_HEIGHT_RATIO = 0.08  # 框高 / 画面高 > 此值 且中心偏下 → 判定为手部
HANDS_TOP_EDGE_RATIO  = 0.60   # 框顶部 y1 > 此值 → 整个框在屏幕下半部分 → 手部/武器
#   真实目标头部（y1）始终在屏幕上半部分；玩家手部从 60% 以下开始出现

# ── 瞄准点 / 头部区域 ────────────────────────────────
HEAD_ZONE_RATIO  = 0.20   # 边界框顶部 20% = 头部区域
SNAP_ZONE_RADIUS = 300    # 屏幕中心吸附圈半径（像素），0 = 不显示
SHOW_SNAP_ZONE   = True

# ── 覆盖层稳定参数 ───────────────────────────────────
OVERLAY_MAX_POOL_SIZE  = 15    # canvas 元素池上限，防长时间运行后画布积累太多项目拖慢渲染
PRIMARY_SWITCH_MARGIN  = 10    # 主目标存在时：新目标需近10px才切换（防止等距目标间频繁切换，但允许近目标快速夺回红框）
PRIMARY_HOLD_FRAMES    = 5     # 主目标消失后：继续保持5帧再切换（防止T2短暂未检测导致T3成红框）
PRIMARY_ADVANTAGE_PX   = 50    # 主目标消失后：新最近目标需比旧主目标近50px才立即切换（否则等HOLD帧到期）
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

