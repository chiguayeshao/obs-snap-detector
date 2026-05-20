# 参考项目深度分析：SunOner/sunone_aimbot & sunone_aimbot_2

> 本文档整理自 GitHub 开源项目，供学习计算机视觉 pipeline 使用。
> 原始仓库：
> - Python版：https://github.com/SunOner/sunone_aimbot
> - C++版：https://github.com/SunOner/sunone_aimbot_2

---

## 目录

1. [sunone_aimbot（Python版）概览](#1-sunone_aimbotpython版概览)
2. [Python版完整文件结构](#2-python版完整文件结构)
3. [核心模块详解：capture.py](#3-核心模块详解capturepy)
4. [核心模块详解：overlay.py](#4-核心模块详解overlaypy)
5. [核心模块详解：visual.py](#5-核心模块详解visualpy)
6. [核心模块详解：run.py（主循环）](#6-核心模块详解runpy主循环)
7. [完整 config.ini 配置项](#7-完整-configini-配置项)
8. [sunone_aimbot_2（C++版）概览](#8-sunone_aimbot_2c版概览)
9. [C++版完整文件结构](#9-c版完整文件结构)
10. [C++版 capture 模块](#10-c版-capture-模块)
11. [C++版 overlay 模块（ImGui + D3D11）](#11-c版-overlay-模块imgui--d3d11)
12. [C++版完整 config.ini 配置项](#12-c版完整-configini-配置项)
13. [关键差异对比：Python vs C++](#13-关键差异对比python-vs-c)
14. [对本项目的改进建议](#14-对本项目的改进建议)
15. [OBS Virtual Camera 接入方式（源自 capture.py）](#15-obs-virtual-camera-接入方式源自-capturepy)

---

## 1. sunone_aimbot（Python版）概览

**仓库**：https://github.com/SunOner/sunone_aimbot  
**语言**：Python 3.12  
**Stars**：活跃维护，多次更新  
**推荐使用环境**：Python 3.12, CUDA 12.8, TensorRT 10.x, Ultralytics 8.3+

### 核心 pipeline

```
屏幕截帧（bettercam/OBS/mss）
        ↓
YOLO推理（YOLOv8/v10/v12，可用 TensorRT 加速）
        ↓
目标选择（FrameParser：最近目标 / head优先）
        ↓
[可视化显示（overlay tkinter + cv2 debug窗口）]
        ↓
[鼠标控制（WIN32/GHUB/Arduino/KMBOX）← 本项目不实现]
```

### requirements.txt

```
cuda_python
bettercam           # ← 推荐，比 dxcam 更稳定
numpy
pywin32
screeninfo
asyncio
onnxruntime
onnxruntime-gpu
pyserial
requests
opencv-python
packaging
ultralytics
keyboard
mss
supervision         # ByteTrack tracker
```

---

## 2. Python版完整文件结构

```
sunone_aimbot/
├── run.py                      # 主入口：模型加载 + 主循环
├── run_ai.bat                  # Windows 批处理启动
├── config.ini                  # 所有配置项
├── helper.py                   # 依赖检查/自动安装
├── requirements.txt
├── version                     # 版本号
└── logic/
    ├── arduino.py              # Arduino HID鼠标驱动
    ├── buttons.py              # 按键名称 ↔ VK code 映射
    ├── capture.py              # ★ 屏幕截帧（bettercam/OBS/mss）
    ├── checks.py               # 启动检查（GPU/依赖/模型）
    ├── config_watcher.py       # config.ini 热重载
    ├── frame_parser.py         # ★ 目标选择（最近距离/head优先）
    ├── game.yaml               # tracker禁用时的YOLO运行配置
    ├── ghub.py                 # Logitech GHUB 鼠标驱动
    ├── ghub_mouse.dll          # GHUB DLL
    ├── hotkeys_watcher.py      # 全局快捷键监听线程
    ├── logger.py               # 日志工具
    ├── mouse.py                # 鼠标移动平滑算法
    ├── overlay.py              # ★ tkinter 透明覆盖层
    ├── rzctl.py                # Razer 鼠标驱动
    ├── rzctl.dll               # Razer DLL
    ├── shooting.py             # 自动射击逻辑
    ├── tracker.yaml            # ByteTrack 参数
    └── visual.py               # ★ cv2 debug窗口 + overlay 绘制逻辑
```

---

## 3. 核心模块详解：capture.py

> **文件**：`logic/capture.py`  
> **关键点**：支持 3 种截帧模式，单队列+线程架构

### 架构设计

```python
class Capture(threading.Thread):
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)  # 单帧队列，只保最新帧
        # ...

    def run(self):
        while self.running:
            frame = self.capture_frame()
            if frame is not None:
                if self.frame_queue.full():
                    self.frame_queue.get()      # 丢弃旧帧
                self.frame_queue.put(frame, block=False)

    def get_new_frame(self):
        return self.frame_queue.get(timeout=1)  # 主循环调用
```

### Mode 1：bettercam（推荐）

```python
def setup_bettercam(self):
    self.bc = bettercam.create(
        device_idx=cfg.bettercam_monitor_id,
        output_idx=cfg.bettercam_gpu_id,
        output_color="BGR",
        max_buffer_len=16,
        region=self.calculate_screen_offset()   # 屏幕中心裁剪区域
    )
    if not self.bc.is_capturing:
        self.bc.start(region=..., target_fps=cfg.capture_fps)

def capture_frame(self):
    if cfg.Bettercam_capture:
        return self.bc.get_latest_frame()       # 非阻塞，直接拿最新帧
```

**要点**：
- `output_color="BGR"` → 直接给 OpenCV/YOLO 用，不需要 RGB→BGR 转换
- `max_buffer_len=16` → 防止内存溢出
- `get_latest_frame()` 返回 numpy array，失败返回 None

### Mode 2：OBS Virtual Camera（★ 核心"OBS吸附"技术）

```python
def setup_obs(self):
    # 自动搜索 OBS Virtual Camera 设备
    camera_id = self.find_obs_virtual_camera()
    
    self.obs_camera = cv2.VideoCapture(camera_id)
    self.obs_camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.detection_window_width)
    self.obs_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.detection_window_height)
    self.obs_camera.set(cv2.CAP_PROP_FPS, cfg.capture_fps)

def find_obs_virtual_camera(self):
    """遍历 DirectShow 设备，找到 OBS Virtual Camera"""
    for i in range(20):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        if cap.getBackendName() == 'DSHOW':
            logger.info(f'OBS Virtual Camera found at index {i}')
            cap.release()
            return i
    return -1

def capture_frame(self):
    if cfg.Obs_capture:
        ret_val, img = self.obs_camera.read()
        return img if ret_val else None
```

**使用条件**：需要先在 OBS Studio 中启动"虚拟摄像头"，将游戏画面推送到虚拟设备。

### Mode 3：mss（兼容模式）

```python
# 在线程内初始化，避免跨线程问题
def run(self):
    if cfg.mss_capture and self.sct is None:
        self.sct = mss.mss()
    ...

def capture_frame(self):
    if cfg.mss_capture:
        screenshot = self.sct.grab(self.monitor)
        img = np.frombuffer(screenshot.bgra, np.uint8).reshape(
            (screenshot.height, screenshot.width, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
```

**注意**：mss 不支持独占全屏（exclusive fullscreen）DX 游戏。

### 屏幕中心裁剪计算

```python
def calculate_screen_offset(self):
    """计算以屏幕中心为中心的裁剪区域"""
    left, top = self.get_primary_display_resolution()
    left = left / 2 - cfg.detection_window_width / 2
    top  = top  / 2 - cfg.detection_window_height / 2
    return (int(left), int(top), int(cfg.detection_window_width), int(cfg.detection_window_height))
```

---

## 4. 核心模块详解：overlay.py

> **文件**：`logic/overlay.py`  
> **关键点**：tkinter 透明窗口，用 queue 跨线程通信，**每帧全删重绘**

### 窗口创建

```python
def run(self, width, height):
    self.root = tk.Tk()
    self.root.overrideredirect(True)        # 无边框窗口
    self.root.attributes('-topmost', True)  # 始终置顶
    self.root.attributes('-transparentcolor', 'black')  # 黑色透明穿透

    self.canvas = Canvas(self.root, bg='black', highlightthickness=0, cursor="none")
    self.canvas.pack(fill=tk.BOTH, expand=True)

    # 阻止 overlay 窗口捕获任何鼠标/键盘事件
    self.root.bind("<Button-1>", lambda e: "break")
    self.root.bind("<Motion>", lambda e: "break")
    # ... 所有事件都绑定 "break"
```

### 跨线程更新机制

```python
# 主线程调用（非阻塞）
def draw_square(self, x1, y1, x2, y2, color='white', size=1):
    self.queue.put((self._draw_square, (x1, y1, x2, y2, color, size)))

# tkinter 主循环内执行（每 2ms 检查队列）
def process_queue(self):
    self.frame_skip_counter += 1
    if self.frame_skip_counter % 3 == 0:      # 跳帧：每 3 次只处理 1 次
        if not self.queue.empty():
            for item in self.canvas.find_all():  # ← 全删重绘（会闪烁！）
                if item != self.square_id:
                    self.canvas.delete(item)
            while not self.queue.empty():
                command, args = self.queue.get()
                command(*args)
    self.root.after(2, self.process_queue)      # 2ms 后再次调用自己
```

**⚠️ 已知问题**：sunone 的 overlay 每帧全删重绘，**我们项目用持久化 item pool 方案解决了闪烁问题**。

### 绘制 API

```python
overlay.draw_square(x1, y1, x2, y2, color='white', size=1)   # 矩形框
overlay.draw_oval(x1, y1, x2, y2, color='white', size=1)      # 椭圆
overlay.draw_line(x1, y1, x2, y2, color='white', size=1)      # 直线
overlay.draw_point(x, y, color='white', size=1)                # 点
overlay.draw_text(x, y, text, size=12, color='white')          # 文字

# 在后台线程启动 overlay
overlay.show(width=320, height=320)
```

---

## 5. 核心模块详解：visual.py

> **文件**：`logic/visual.py`  
> **关键点**：既控制 cv2 debug 窗口，也向 overlay 发送绘制命令

### 两路输出

```python
# 同一份 boxes 数据，同时输出到 cv2 和 tkinter overlay
for xyxy, cls, conf in zip(xyxy_iter, cls_iter, conf_iter):
    x0, y0, x1, y1 = map(int, map(float, xyxy))

    # cv2 debug 窗口（仅在 show_window=True 时）
    if cfg.show_window and cfg.show_boxes:
        cv2.rectangle(self.image, (x0, y0), (x1, y1), (0, 200, 0), 2)

    # tkinter overlay（仅在 show_overlay=True 时）
    if cfg.show_overlay and cfg.overlay_show_boxes:
        overlay.draw_square(x0, y0, x1, y1, 'green', 2)
```

### 标签显示位置智能处理

```python
# 防止标签超出 overlay 边界
if y0 <= 15:
    x_out = x0 - 45;  y_out = y0 + 15
else:
    x_out = x0 + 45;  y_out = y0 - 15
if x0 <= 40:
    x_out = x0 + 40
if x0 >= cfg.detection_window_width - 80:
    x_out = x0 - 40
```

### 已知的自捕获问题（注释原话）

```python
# Skip frames so that the figures do not interfere with the detector ¯\_(ツ)_/¯
self.frame_skip_counter += 1
if self.frame_skip_counter % 3 == 0:
    ...
```

sunone 通过**跳帧（每3帧处理1次）**来降低 overlay 图形被 dxcam 截到再送给模型的概率，但这只是缓解，不能根治。

---

## 6. 核心模块详解：run.py（主循环）

```python
from ultralytics import YOLO
import supervision as sv

tracker = sv.ByteTrack() if not cfg.disable_tracker else None

@torch.inference_mode()
def perform_detection(model, image, tracker):
    kwargs = dict(
        source=image,
        imgsz=cfg.ai_model_image_size,
        conf=cfg.AI_conf,
        iou=0.50,
        device=cfg.AI_device,
        half=not "cpu" in cfg.AI_device,   # FP16（GPU加速）
        max_det=20,
        verbose=False,
        stream=True
    )
    kwargs["cfg"] = "logic/tracker.yaml" if tracker else "logic/game.yaml"
    results = model.predict(**kwargs)

    if tracker:
        for res in results:
            det = sv.Detections.from_ultralytics(res)
            return tracker.update_with_detections(det)   # ByteTrack
    else:
        return next(results)

def init():
    model = YOLO(f"models/{cfg.AI_model_name}", task="detect")

    while True:
        image = capture.get_new_frame()
        if image is not None:
            if cfg.circle_capture:
                image = capture.convert_to_circle(image)   # 圆形裁剪
            if cfg.show_window or cfg.show_overlay:
                visuals.queue.put(image)
            result = perform_detection(model, image, tracker)
            if hotkeys_watcher.app_pause == 0:
                frameParser.parse(result)
```

**关键参数**：
- `half=True`：FP16 推理，GPU 速度翻倍
- `stream=True`：流式输出，节省内存
- `@torch.inference_mode()`：禁用梯度计算，等同 `no_grad()`

---

## 7. 完整 config.ini 配置项

```ini
[Detection window]
detection_window_width = 320   # 截帧宽度（像素）
detection_window_height = 320  # 截帧高度（像素）
circle_capture = True          # 圆形裁剪（减少边角干扰）

[Capture Methods]              # 只能选一个 True
capture_fps = 60
Bettercam_capture = False      # ★ 推荐，最稳定
bettercam_monitor_id = 0       # 显示器索引
bettercam_gpu_id = 0           # GPU索引
Obs_capture = False            # OBS Virtual Camera 模式
Obs_camera_id = 1              # auto 或 整数
mss_capture = True             # 默认模式，兼容性最好但最慢

[AI]
AI_model_name = sunxds_0.5.6.pt  # 模型文件名（放在 models/ 目录）
AI_model_image_size = 640        # 推理分辨率（320 更快）
AI_conf = 0.2                    # 置信度阈值
AI_device = 0                    # GPU设备 (0=cuda:0, cpu=CPU)
AI_enable_AMD = False            # ROCm/HIP AMD GPU
disable_tracker = False          # 禁用 ByteTrack

[overlay]
show_overlay = False
overlay_show_borders = True      # 显示截帧区域边框
overlay_show_boxes = False       # 显示检测框
overlay_show_target_line = False # 显示目标连线
overlay_show_labels = False      # 显示类别标签
overlay_show_conf = False        # 显示置信度

[Debug window]
show_window = False              # cv2 调试窗口
show_detection_speed = True      # 显示推理速度
show_boxes = True
debug_window_always_on_top = True
spawn_window_pos_x = 100
spawn_window_pos_y = 100
debug_window_scale_percent = 100
debug_window_screenshot_key = End

[Hotkeys]
hotkey_targeting = RightMouseButton
hotkey_exit = F2
hotkey_pause = F3
hotkey_reload_config = F4

[Mouse]                          # 本项目不使用
mouse_auto_aim = False           # ← 必须保持 False

[Aim]
body_y_offset = 0.1              # 目标点 Y 偏移（相对高度，0=中心）
hideout_targets = True
disable_headshot = False
disable_prediction = False
```

---

## 8. sunone_aimbot_2（C++版）概览

**仓库**：https://github.com/SunOner/sunone_aimbot_2  
**语言**：C++17  
**编译**：Visual Studio（.vcxproj）  
**依赖**：CUDA 13.1 / DirectML，TensorRT 10.14，OpenCV，ImGui，CppWinRT

### 为什么选 C++？

| 对比项 | Python版 | C++版 |
|--------|---------|-------|
| 推理后端 | PyTorch/Ultralytics | TensorRT (native) / DirectML |
| overlay | tkinter（不支持独占全屏） | **ImGui + D3D11（支持全屏）** |
| 截帧方式 | bettercam Python绑定 | 原生 DXGI Duplication API |
| 延迟 | 较高（GIL） | 最低（native threads） |
| 适用场景 | 学习/调试 | 实际部署 |

### 提供预编译版本

- **DML版**（通用，支持 NVIDIA/AMD/Intel 所有 GPU）
- **CUDA + TensorRT 版**（仅 NVIDIA RTX 20xx+）
- 从 Discord 下载 `ai.exe`，直接运行

---

## 9. C++版完整文件结构

```
sunone_aimbot_2/
├── sunone_aimbot_2.cpp         # 主入口
├── sunone_aimbot_2.h
├── sunone_aimbot_2.vcxproj     # Visual Studio 项目文件
├── ghub_mouse.dll              # Logitech GHUB DLL
├── capture/
│   ├── capture.cpp/.h          # 截帧总入口（模式选择）
│   ├── duplication_api_capture.cpp/.h  # DXGI Desktop Duplication
│   ├── winrt_capture.cpp/.h    # WinRT Windows.Graphics.Capture
│   ├── virtual_camera.cpp/.h   # 虚拟摄像头（类似OBS模式）
│   ├── udp_capture.cpp/.h      # UDP 网络截帧（双机方案）
│   └── capture_utils.cpp/.h
├── config/
│   ├── config.cpp/.h           # 配置读取（config.ini）
├── depth/                      # 深度估计（Depth Anything v2）
├── detector/                   # ONNX/TRT 推理
├── imgui/                      # ImGui 源码
├── include/                    # 第三方头文件
├── keyboard/
│   └── keycodes.cpp            # VK code 映射
├── mem/                        # 内存工具
├── modules/                    # 功能模块
├── mouse/                      # 鼠标移动（WIN32/GHUB/Arduino/KMBOX）
├── overlay/
│   ├── overlay.cpp/.h          # ★ D3D11 透明窗口创建
│   ├── Game_overlay.cpp/.h     # ★ 游戏内 overlay（DirectComposition）
│   ├── draw_ai.cpp             # 绘制检测框
│   ├── draw_game_overlay.cpp   # 游戏 overlay 绘制
│   ├── draw_debug.cpp          # 调试信息
│   ├── draw_mouse.cpp          # 鼠标轨迹可视化
│   ├── draw_target.cpp         # 目标高亮
│   ├── draw_stats.cpp          # 统计信息
│   ├── draw_capture.cpp        # 截帧区域可视化
│   ├── draw_depth.cpp          # 深度图可视化
│   └── draw_buttons.cpp        # 设置界面按钮
├── runtime/                    # 运行时库
├── scr/                        # 截图保存
└── tensorrt/                   # TensorRT 工具
```

---

## 10. C++版 capture 模块

C++ 版的截帧模式通过 `capture_method` 配置项切换：

| 模式 | 值 | 说明 |
|------|-----|------|
| DXGI 桌面复制 | `duplication_api` | 最快，不支持独占全屏输出到 overlay |
| WinRT 窗口捕获 | `winrt` | 支持指定窗口标题捕获 |
| 虚拟摄像头 | `virtual_camera` | OBS Virtual Camera 等 |
| UDP 网络 | `udp_capture` | 双机方案（主机截帧→另一台推理） |

### WinRT 模式（推荐用于抓游戏窗口）

```ini
capture_method = winrt
capture_target = window
capture_window_title = Delta Force   # 游戏窗口标题
capture_cursor = false
capture_borders = false
```

对应源文件：`sunone_aimbot_2/capture/winrt_capture.cpp`  
- 使用 `Windows.Graphics.Capture` API（Win10 1903+）
- 可捕获特定窗口（不受全屏/窗口模式影响）
- 相比 DXGI 有 ~1 帧延迟，但稳定性更好

### DXGI 直接捕获 + CUDA（最低延迟配置）

```ini
capture_method = duplication_api
backend = TRT
capture_use_cuda = true         # 截帧直接在 GPU 内存，不走 CPU
detection_resolution = 320
capture_fps = 120
```

条件：CUDA build + TRT backend + duplication_api + circle_mask=false + 深度mask关闭

---

## 11. C++版 overlay 模块（ImGui + D3D11）

C++ 版 overlay 的核心优势：**通过 DirectComposition 创建真正透明的 D3D11 窗口，可以覆盖在独占全屏游戏上**。

### 关键配置

```ini
game_overlay_enabled = false          # 主开关（默认关，影响性能）
game_overlay_max_fps = 0              # 0 = 不限制
game_overlay_draw_boxes = true        # 绘制检测框
game_overlay_draw_frame = true        # 绘制截帧区域边框

# 框颜色（ARGB）
game_overlay_box_a = 255
game_overlay_box_r = 0
game_overlay_box_g = 255
game_overlay_box_b = 0               # 默认绿色框
game_overlay_box_thickness = 2.0

# overlay 窗口
overlay_exclude_from_capture = true  # ★ 不被截帧工具捕获（防止自检测！）
```

### C++ overlay 创建逻辑（overlay.cpp 关键部分）

overlay 通过以下 Win32 API 创建：
1. 创建分层窗口（`WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST`）
2. 使用 DirectComposition 设置 alpha 混合
3. 用 ImGui + D3D11 渲染 UI 和检测框
4. `overlay_exclude_from_capture = true` → 调用 `SetWindowDisplayAffinity(HWND_TOPMOST, WDA_EXCLUDEFROMCAPTURE)` 使 overlay 不被 DXGI/WinRT 截帧

**关键 API**：
```cpp
SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
// 这使 overlay 在截帧中不可见，防止模型检测到自己的框
```

---

## 12. C++版完整 config.ini 配置项

### 快速示例

```ini
# 示例 A：普通 DirectML 版本（AMD/Intel GPU 或老 NVIDIA）
capture_method = duplication_api
backend = DML
detection_resolution = 320
capture_fps = 60

# 示例 B：高性能 CUDA+TRT 版本
capture_method = duplication_api
backend = TRT
capture_use_cuda = true
detection_resolution = 320
capture_fps = 120

# 示例 C：WinRT 窗口捕获
capture_method = winrt
capture_target = window
capture_window_title = Counter-Strike 2

# 示例 D：OBS虚拟摄像头
capture_method = virtual_camera
virtual_camera_name = None          # None = 自动检测
virtual_camera_width = 1920
virtual_camera_heigth = 1080
```

### AI 相关配置

```ini
backend = DML                       # DML / TRT
ai_model = sunxds_0.5.6.onnx       # 模型文件（放在 models/ 目录）
confidence_threshold = 0.10         # 置信度阈值
nms_threshold = 0.50                # NMS IOU 阈值
max_detections = 100                # 最大检测数

class_player = 0                    # 模型中玩家类的索引
class_head = 1                      # 模型中头部类的索引
```

### 深度估计（CUDA版独有）

```ini
depth_inference_enabled = false     # 深度估计主开关
depth_model_path = depth_anything_v2.engine
depth_fps = 100
depth_mask_enabled = false          # 深度过滤（过滤不在合理距离的检测）
```

---

## 13. 关键差异对比：Python vs C++

| 特性 | Python版 | C++版 |
|------|---------|-------|
| **易用性** | pip install 即可 | 需编译或下载预编译 |
| **overlay 全屏支持** | ❌ tkinter 不支持独占全屏 | ✅ D3D11+DirectComposition |
| **overlay 自捕获** | ⚠️ 通过跳帧缓解 | ✅ `WDA_EXCLUDEFROMCAPTURE` 根治 |
| **截帧速度** | bettercam ≈ 180fps | DXGI native ≈ 300fps+ |
| **推理速度** | Ultralytics ONNX DirectML | TensorRT 最快 |
| **AMD GPU 支持** | ✅ onnxruntime-directml | ✅ DML build |
| **Kalman 预测** | 基础实现 | 完整 Kalman + WindMouse |
| **深度估计** | ❌ | ✅ Depth Anything v2 |
| **双机 UDP** | ❌ | ✅ udp_capture 模式 |
| **配置热重载** | ✅ F4 | ✅ F4 |

---

## 14. 对本项目的改进建议

基于对两个参考项目的分析，以下是对 `obs-snap-detector` 的具体改进建议：

### 优先级 HIGH

#### 1. 替换 dxcam 为 bettercam

```python
# requirements.txt 改动
# 删除: dxcam
# 新增: bettercam

# capture.py 改动
import bettercam

# 初始化
self.bc = bettercam.create(
    device_idx=0,
    output_idx=0,
    output_color="BGR",
    max_buffer_len=16,
    region=(left, top, width, height)
)
self.bc.start(region=(left, top, width, height), target_fps=60)

# 截帧
frame = self.bc.get_latest_frame()
```

**理由**：dxcam 有已知的自捕获 bug（捕获到 tkinter overlay），bettercam 是社区推荐的稳定替代。

#### 2. 添加 OBS Virtual Camera 模式

```python
# config.py 新增
OBS_CAPTURE = False
OBS_CAMERA_ID = 'auto'   # 'auto' 或整数

# capture.py 新增
def setup_obs(self):
    camera_id = self._find_obs_camera() if OBS_CAMERA_ID == 'auto' else int(OBS_CAMERA_ID)
    self.obs_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    self.obs_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    self.obs_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

def _find_obs_camera(self):
    for i in range(20):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return 0
```

#### 3. 游戏必须使用"无边框窗口"模式

**tkinter overlay 的限制**：
- ✅ 支持：窗口模式 / 无边框窗口（Borderless Windowed）
- ❌ 不支持：独占全屏（Exclusive Fullscreen / Direct3D Exclusive）

在游戏中设置：`显示模式 → 无边框窗口（Borderless Window）`

### 优先级 MEDIUM

#### 4. 参考 sunone 的 frame_skip 解决 overlay 闪烁

sunone 使用跳帧来减少自捕获；我们项目已用 item pool 解决，但可以增加帧跳过逻辑进一步降低 CPU 占用。

#### 5. 增加 cv2 debug 窗口模式（不依赖 tkinter overlay）

```python
# 参考 visual.py 的双路输出设计
if cfg.SHOW_DEBUG_WINDOW:
    cv2.imshow('Debug', frame)
    cv2.waitKey(1)
```

### 优先级 LOW（高级功能）

#### 6. 长期目标：迁移到 C++ ImGui overlay（参考 sunone_aimbot_2）

对于支持全屏游戏，需要：
- Win32 分层窗口 + `WDA_EXCLUDEFROMCAPTURE`
- DirectComposition 透明
- ImGui + D3D11 渲染

对应源码：`sunone_aimbot_2/overlay/overlay.cpp`

---

## 15. OBS Virtual Camera 接入方式（源自 capture.py）

"OBS吸附"名称的由来：早期方案通过 OBS 的虚拟摄像头功能将游戏画面转为"摄像头设备"，再用 `cv2.VideoCapture` 读取。

### 完整接入步骤

**Step 1：安装 OBS Studio**
- 下载：https://obsproject.com/

**Step 2：在 OBS 中设置游戏捕获**
```
来源 → + → 游戏捕获 → 选择游戏进程
工具 → 虚拟摄像头 → 启动
```

**Step 3：Python 代码读取**

```python
import cv2

def find_obs_virtual_camera():
    """自动搜索 OBS Virtual Camera 设备索引"""
    for i in range(20):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        # OBS Virtual Camera 会出现在 DirectShow 设备列表中
        print(f"Found device {i}: {cap.getBackendName()}")
        cap.release()
        return i
    return -1

# 使用
camera_id = find_obs_virtual_camera()
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = cap.read()
    if ret:
        # frame 是 BGR numpy array，直接送 YOLO
        results = model(frame)
```

### OBS 模式 vs bettercam 模式对比

| 对比项 | bettercam | OBS Virtual Camera |
|--------|-----------|-------------------|
| 延迟 | ~5ms | ~30-50ms（OBS编码延迟）|
| CPU占用 | 低 | 中（OBS占用）|
| 全屏支持 | ✅ | ✅ |
| 依赖 | pip install | OBS Studio |
| 分辨率 | 原生 | OBS 配置 |
| 自捕获风险 | 有（需 bettercam） | 无（OBS 可排除 overlay） |

**结论**：生产环境用 bettercam；学习/调试可用 OBS 模式。

---

*文档整理时间：2025年。原始项目持续更新，请以 GitHub 最新版本为准。*
