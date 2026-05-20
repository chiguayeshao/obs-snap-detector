# OBS 吸附原理调研报告

> 调研时间：2026-05-20  
> 目标：收集开源实现 + 关键组件选型 + 已知坑点，指导项目迭代

---

## 一、核心参考项目

| 仓库 | 语言 | Stars | 截帧 | 检测 | 覆盖层 | 备注 |
|------|------|-------|------|------|--------|------|
| [SunOner/sunone_aimbot](https://github.com/SunOner/sunone_aimbot) | Python | Active | bettercam / OBS / mss | YOLOv8/v10 | tkinter | **最完整 Python 参考** |
| [SunOner/sunone_aimbot_2](https://github.com/SunOner/sunone_aimbot_2) | C++ | Active | DXGI C++ | TensorRT | ImGui + D3D11 | **独占全屏覆盖层的正确解** |
| [RootKit-Org/AI-Aimbot](https://github.com/RootKit-Org/AI-Aimbot) | Python | ~400 | bettercam | YOLOv5/ONNX/TRT | cv2.imshow | 不再维护；bettercam 用法参考 |
| [DEVILENMO/DeadEye-Auto-Aiming-System](https://github.com/DEVILENMO/DeadEye-Auto-Aiming-System) | Python | 92 | dxcam + mss | YOLOv8 + TRT | customtkinter | 中文开发者，TRT 路径硬编码需改 |
| [Lucid1ty/Yolov5ForCSGO](https://github.com/Lucid1ty/Yolov5ForCSGO) | Python | 151 | Win32 BitBlt | YOLOv5 | cv2 topmost | 学习友好；BitBlt 对全屏无效 |
| [Fragmentaim/Auto_aim](https://github.com/Fragmentaim/Auto_aim) | C++ | Small | DXGI Desktop Dup | ONNX + TRT | 无 | 核心 C++ DXGI 参考 |
| [petercunha/Pine](https://github.com/petercunha/Pine) | Python | 443 | win32api | YOLOv3 | 无 | 2019 历史项目，奠基性 |

---

## 二、OBS 吸附的真正含义（两种变体）

### Variant A — OBS Virtual Camera（`sunone_aimbot` 的 `Obs_capture` 模式）

这才是「OBS 吸附」名字的直接来源：

1. OBS Studio 开启 **Virtual Camera 插件**
2. OBS 以自己的内核级访问权限捕获游戏画面（所有全屏模式均可）
3. Python 通过 `cv2.VideoCapture` 读取 OBS 的 DirectShow 虚拟摄像头输出

```python
# sunone_aimbot/logic/capture.py 的 OBS 模式核心代码
obs_camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
obs_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
obs_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
obs_camera.set(cv2.CAP_PROP_FPS, 60)
while True:
    ret, frame = obs_camera.read()
    results = model(frame)
```

**优点**：无需 Python 端任何 DirectX 权限，兼容所有全屏模式  
**缺点**：帧率上限约 60fps，需要 OBS 保持运行

### Variant B — DXGI Desktop Duplication（主流方案）

Python 直接调用 Windows DXGI API 截取显卡输出：

```python
import bettercam
camera = bettercam.create(output_color="BGR")
camera.start(target_fps=60, video_mode=True)
frame = camera.get_latest_frame()  # numpy array, BGR
```

**优点**：无依赖、延迟最低、帧率可达 240fps+  
**缺点**：有已知的「自截自己覆盖层」bug（见下方坑点）

---

## 三、屏幕截帧库对比

| 库 | 独占全屏 | 平均 FPS | 安装难度 | 主要问题 |
|----|---------|---------|---------|---------|
| **dxcam** (ra1nty, 新版) | ✅ | 240+ | 中 | Python 3.14 comtypes bug；v0.3.0 实例复用 bug |
| **bettercam** | ✅ | ~180 | 易 | 旧版 dxcam 的 fork，API 更稳定，主流 aimbot 首选 |
| **mss** | ❌（独占全屏黑屏）| ~75 | 极易 | 只能捕获窗口/无边框模式 |
| **Win32 BitBlt** | ❌（黑屏）| ~60 | 易 | 仅窗口模式 |
| **cv2.VideoCapture（OBS）** | ✅（通过 OBS）| ≤60 | 难 | 需要 OBS 保持运行 |
| **C++ DXGI** | ✅ | 500+ | 难 | 需要 C++ 工程 |

### ⚠️ dxcam 已知 Bug（重要！）

1. **自截自身覆盖层**：dxcam 截取整个桌面，包括自己画的透明覆盖层  
   → 模型会「看到」自己画的框，造成误检测循环  
   → 解法：用独立副显示器，或用 OBS 模式（OBS 可配置只捕获游戏窗口）

2. **`comtypes.IUnknown` AttributeError**（Python 3.14 + 新版 comtypes）  
   → 解法：`pip uninstall comtypes && pip install comtypes==1.4.1`

3. **多显示器 / RDP 切换**后截帧失败  
   → 解法：重新实例化 `dxcam.create()`

4. **HDR 显示器**输出过曝  
   → 解法：关闭 Windows HDR，或在截帧后做 tonemapping

### bettercam 安装（推荐）

```bash
pip install bettercam
```

```python
import bettercam
W, H = 640, 640
camera = bettercam.create(output_color="BGR", max_buffer_len=16)
camera.start(region=(640, 220, 1280, 860), target_fps=60, video_mode=True)
frame = camera.get_latest_frame()
```

---

## 四、AI 推理性能对比

| 模型 + 推理引擎 | RTX 3080 | GTX 1660 Ti | CPU (Ryzen 7) |
|----------------|---------|------------|--------------|
| YOLOv8n 320px PyTorch | ~100fps | ~40fps | ~10fps |
| YOLOv8n 640px PyTorch | ~50fps | ~20fps | ~5fps |
| YOLOv8s 640px ONNX+DML | ~80fps | ~35fps | ~15fps |
| YOLOv8s 640px ONNX+CUDA | ~150fps | ~60fps | N/A |
| YOLOv8s 640px TensorRT | ~250fps | ~100fps | N/A |

**实践建议**：
- 无 NVIDIA GPU → `ONNX + DirectML`（支持 AMD/Intel）
- NVIDIA RTX → `ONNX + CUDA`（`onnxruntime-gpu`）
- 追求极致 → TensorRT `.engine`（setup 复杂，需 CUDA + cuDNN + TRT 版本完全匹配）

**导出 ONNX**：
```bash
# Ultralytics 一键导出
yolo export model=yolov8s.pt format=onnx imgsz=640
# 或 DirectML 专用
yolo export model=yolov8s.pt format=onnx imgsz=640 simplify=True
```

---

## 五、覆盖层方案对比

Windows 有三种全屏模式，难度完全不同：

| 游戏模式 | tkinter 透明层 | pygame | cv2 独立窗口 | C++ ImGui+D3D11 |
|---------|--------------|--------|------------|----------------|
| **窗口模式** | ✅ | ✅ | ✅ | ✅ |
| **无边框窗口** | ✅ | ✅ | ✅ | ✅ |
| **独占全屏** | ❌ | ❌ | ❌ | ✅ |

### Tier 1 — tkinter（sunone_aimbot 方案，仅无边框）

```python
# sunone_aimbot/logic/overlay.py 的核心模式
root = tk.Tk()
root.overrideredirect(True)
root.attributes('-topmost', True)
root.attributes('-transparentcolor', 'black')  # 黑色=透明
canvas = tk.Canvas(root, bg='black', highlightthickness=0)
# 画框
canvas.create_rectangle(x1, y1, x2, y2, outline='green', width=2)
root.update()
```

### Tier 2 — cv2.imshow + HWND_TOPMOST（学习项目最简方案）

不覆盖游戏，但总是置顶显示检测结果窗口：

```python
import cv2, win32gui, win32con
cv2.imshow('Detection', frame_with_boxes)
hwnd = win32gui.FindWindow(None, 'Detection')
win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
cv2.waitKey(1)
```

### Tier 3 — C++ ImGui + D3D11（独占全屏唯一有效方案）

`SunOner/sunone_aimbot_2` 的实现路径：
1. 创建 D3D11 设备和交换链
2. 用 ImGui 渲染绘制列表
3. 窗口设置 `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST`
4. 用 `AlphaBlend` 实现透明

---

## 六、已发现的主要坑点汇总

| 问题 | 现象 | 原因 | 解法 |
|------|------|------|------|
| dxcam 截到自身覆盖层 | 出现框框互套、误检测 | DXGI 捕获整个桌面合成图层 | 换 bettercam；或只截游戏窗口 HWND |
| tkinter 在独占全屏游戏不显示 | 覆盖层透明无内容 | Win32 窗口无法在 DX exclusive mode 上层渲染 | 切换游戏为无边框窗口模式 |
| 幽灵框（ghost boxes）| 画面空旷时仍有框存在 | 帧间 tracker 未能正确清除旧轨迹 | MAX_AGE=2~5，Kalman velocity zeroing |
| YOLO 把自己的手检测为人 | 屏幕下方出现大框 | COCO person 类包括手臂 | 按 y 坐标 + 框高比例过滤底部区域 |
| dxcam comtypes 报错 | `IUnknown AttributeError` | comtypes 版本不兼容 | `pip install comtypes==1.4.1` |
| keyboard 库需要管理员权限 | 快捷键无响应 | 低级键盘钩子受 UAC 限制 | 改用 `GetAsyncKeyState` 轮询（无需管理员）|
| 推理太慢导致 FPS 低 | 画面卡顿 | PyTorch CPU 推理 | 用 ONNX+DirectML 或缩小 imgsz 至 320px |
| tkinter 闪烁 | 框一闪一闪 | `canvas.delete("all")` 每帧重建 | 用 `canvas.coords()` + `canvas.itemconfig()` 原地更新 |

---

## 七、对本项目的建议

基于以上调研，推荐以下改进方向：

### 截帧
- 把 `dxcam` 替换为 `bettercam`（更稳定，主流项目的选择）
- 或添加 OBS Virtual Camera 模式（`cv2.VideoCapture`），支持独占全屏

### 检测
- 如果没有 NVIDIA GPU，用 `onnxruntime-directml` 替代 PyTorch（速度 2-3x）
- 导出命令：`yolo export model=yolo11s.pt format=onnx imgsz=640`

### 覆盖层
- 当前 tkinter 方案在独占全屏下无效，需告知用户**在游戏中切换为「无边框窗口」模式**
- 或改用 `cv2.imshow` + `HWND_TOPMOST` 作为调试窗口，不依赖覆盖层

### 参考代码
- 截帧：[RootKit-Org/AI-Aimbot:gameSelection.py](https://github.com/RootKit-Org/AI-Aimbot/blob/main/gameSelection.py)
- 覆盖层：[SunOner/sunone_aimbot:logic/overlay.py](https://github.com/SunOner/sunone_aimbot/blob/main/logic/overlay.py)
- OBS 捕获：[SunOner/sunone_aimbot:logic/capture.py](https://github.com/SunOner/sunone_aimbot/blob/main/logic/capture.py)

---

## 八、快速验证脚本（最简可用版）

这是调研中发现的最简单的 **确认可运行** 的检测脚本，适合快速验证环境：

```python
# quick_test.py
# 依赖: pip install bettercam opencv-python ultralytics
import bettercam, cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 自动下载 ~6MB
camera = bettercam.create(output_color="BGR")
camera.start(target_fps=30, video_mode=True)

print("按 Q 退出")
while True:
    frame = camera.get_latest_frame()
    if frame is None:
        continue
    results = model(frame, conf=0.5, classes=[0], verbose=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"person {conf:.0%}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow('OBS Snap Detector - Debug', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
```

> 运行后弹出 cv2 窗口，显示带检测框的实时画面。比主程序的 tkinter 覆盖层更容易调试。
