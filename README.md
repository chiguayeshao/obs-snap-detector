# obs-snap-detector

> 📡 **学习用途** — 复现 "OBS吸附" 外挂的截帧 + AI检测原理，**不包含自瞄/鼠标控制**

## 原理图

```
游戏画面 (GPU 输出)
    │
    ▼  DXGI Desktop Duplication（与 OBS 同一 API）
┌─────────────────────┐
│  capture.py         │  截取屏幕帧 → RGB numpy array
└──────────┬──────────┘
           │
    ▼  YOLOv8n (COCO person 类)
┌─────────────────────┐
│  detector.py        │  检测画面中的人，输出 bounding box
└──────────┬──────────┘
           │
    ▼  tkinter 透明覆盖层
┌─────────────────────┐
│  overlay.py         │  在屏幕上叠加高亮框（鼠标可穿透）
└─────────────────────┘
```

## 环境要求

- **Windows 10/11**（DXGI API 专属）
- Python 3.10+
- NVIDIA GPU（可选，有 GPU 推理更快）
- 以**管理员身份**运行终端（全局快捷键需要权限）

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/chiguayeshao/obs-snap-detector.git
cd obs-snap-detector

# 2. 安装依赖（建议使用虚拟环境）
pip install -r requirements.txt

# 3. 运行（以管理员身份）
python main.py
```

首次运行会自动下载 `yolov8n.pt` 模型文件（~6MB）。

## 快捷键

| 按键 | 功能 |
|------|------|
| `F9` | 开/关 覆盖层 |
| `ESC` | 退出程序 |

## 配置

编辑 `config.py` 调整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CONFIDENCE_THRESHOLD` | `0.5` | 检测置信度阈值 |
| `CAPTURE_REGION` | `None` | 截帧区域（None=全屏） |
| `BOX_COLOR` | `#00FF41` | 高亮框颜色 |
| `MODEL_NAME` | `yolov8n.pt` | YOLO 模型（n最快/s/m最准） |
| `TARGET_FPS` | `30` | 目标帧率 |

## 项目结构

```
obs-snap-detector/
├── requirements.txt   # 依赖
├── config.py          # 可调参数
├── capture.py         # 屏幕截帧（dxcam / DXGI）
├── detector.py        # YOLOv8 人体检测
├── overlay.py         # 透明覆盖层（tkinter）
└── main.py            # 主循环 + 快捷键
```

## 学习要点

- **`capture.py`** — 了解 DXGI Desktop Duplication，OBS "Display Capture" 的底层 API
- **`detector.py`** — YOLOv8 单阶段检测器，`conf` + `classes` 过滤的用法
- **`overlay.py`** — Windows `WS_EX_TRANSPARENT` 实现鼠标穿透透明窗口
- **`main.py`** — 生产者-消费者式主循环，FPS 计算

## 声明

本项目**仅供学习计算机视觉和 Windows API 原理使用**，请勿用于任何实际游戏对战。
