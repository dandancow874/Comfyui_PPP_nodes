# ComfyUI-LMStudio-Controller

[English](#english) | [中文](#chinese)

<a name="english"></a>
## 🇬🇧 English

A powerful Custom Node for ComfyUI that integrates **LM Studio**'s CLI (`lms`) to perform Vision Language Model (VLM) inference locally. 

It upgrades your ComfyUI with the ability to "see" images and videos using local models (like Qwen2-VL, LLaVA, etc.), with advanced memory management to prevent OOM errors on consumer GPUs.

### ✨ Key Features

*   **🎬 Video Analysis Support**: Seamlessly integrates with **VideoHelperSuite (VHS)** to analyze video frames.
*   **🖼️ Multi-Input Interface**: Supports **Main Image**, **Image 2**, **Image 3**, and **Video Frames** inputs simultaneously.
*   **🛡️ Anti-OOM Protection (Core)**:
    *   **Smart Downsampling**: Automatically downsamples input frames to the `max_total_images` limit (e.g., squeezes 100 frames into 8) to prevent token overflow.
    *   **Resolution Control**: Resizes large images/video frames via `max_image_side` before processing to save VRAM.
    *   **GPU Offload Control**: Manually adjust how many model layers are loaded to VRAM (0.0 - 1.0) to prevent system crashes/blue screens.
    *   **Context Length Control**: Explicitly set context window limits to manage VRAM usage.
*   **🚀 Smart Model Management**: Automatically fetches models via `lms ls`. Only reloads the model when parameters change.

### ⚙️ Prerequisites (Important)

1.  **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/).
2.  **Enable CLI Tool**: 
    *   Open LM Studio -> **"Developer"** tab.
    *   Click **"Install lms to PATH"** (Verify by running `lms` in your terminal).
3.  **Start Local Server**:
    *   In LM Studio -> **"Local Server"** tab.
    *   Click **"Start Server"** (Default port: 1234).
    *   *Note: This node relies on the Local Server API to receive images.*

### 📥 Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/dandancow874/ComfyUI-LMStudio-Controller.git
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

### 📖 Usage Guide

#### 1. Basic Image Analysis
Simply connect your `Load Image` node to the **`image`** input. You can also connect optional images to `image_2` or `image_3` for multi-image comparison.

#### 2. Video Analysis Workflow (Recommended)
To analyze videos, it is recommended to use the **VideoHelperSuite (VHS)** nodes.

1.  Add a **`Load Video (Upload)`** node (from VHS).
2.  **Crucial Setting**: Set VHS `force_size` to **Custom** (e.g., 512x512 or 768x768). This saves massive amounts of VRAM.
3.  Set VHS `frame_load_cap` to a reasonable limit (e.g., 32 or 64).
4.  Connect the `IMAGE` output of VHS to the **`video_frames`** input of this node.
5.  Set `max_total_images` on this node (e.g., 8). It will automatically sample 8 evenly distributed frames from the video.

### 🛠️ Recommended Settings

| Hardware/Model | GPU Offload | Context Length | Max Image Side | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **8B Models** (e.g., Qwen2-VL-7B) | `1.0` (Max) | `16384` | `768` or `1024` | Fast, handles high-res images well. |
| **30B+ Models** (Low VRAM) | `0.6` - `0.8` | `4096` | `512` | **Prevents Blue Screen/OOM.** |
| **30B+ Models** (24GB+ VRAM) | `0.8` - `1.0` | `8192` | `768` | Balance between quality and memory. |

---

<a name="chinese"></a>
## 🇨🇳 中文说明

这是一个功能强大的 ComfyUI 自定义节点，通过集成 **LM Studio** 的命令行工具 (`lms`)，实现本地视觉大模型 (VLM) 的推理。

它不仅让 ComfyUI 能够“看懂”图片和视频（支持 Qwen2-VL, LLaVA 等），还引入了高级的显存管理机制，防止在消费级显卡上运行大参数模型时发生爆显存（OOM）或蓝屏。

### ✨ 主要功能

*   **🎬 视频理解支持**：完美配合 **VideoHelperSuite (VHS)** 节点，接收视频流并进行反推/分析。
*   **🖼️ 多模态输入接口**：支持 **主图**、**图2**、**图3** 以及 **视频帧** (Video Frames) 同时输入。
*   **🛡️ 防爆显存机制 (核心功能)**：
    *   **智能抽帧**：无论输入多少视频帧（如100帧），自动根据 `max_total_images` 限制（如8帧）进行均匀采样，防止 Token 溢出。
    *   **分辨率控制**：通过 `max_image_side` 强制限制输入给模型的图片长边尺寸，大幅降低显存占用。
    *   **GPU Offload 控制**：支持手动调节模型加载到显存的比例 (0.0 - 1.0)。30B 大模型建议设为 0.8 以防蓝屏。
    *   **Context Length 控制**：显式限制模型的上下文窗口大小。
*   **🚀 智能加载**：通过 `lms ls` 自动获取列表。仅当模型名称或关键显存参数变化时才触发重载。

### ⚙️ 前置要求 (重要)

1.  **安装 LM Studio**: 请前往 [lmstudio.ai](https://lmstudio.ai/) 下载。
2.  **启用 CLI 工具**: 
    *   打开 LM Studio -> **"Developer" (开发者)** 选项卡。
    *   点击 **"Install lms to PATH"**（确保在终端输入 `lms` 能看到输出）。
3.  **启动本地服务 (Local Server)**:
    *   在 LM Studio 中 -> **"Local Server"** 选项卡。
    *   点击 **"Start Server"** (默认端口 1234)。
    *   *注意：本节点依赖 Local Server 接收图像 API 请求。*

### 📥 安装方法

1.  进入你的 ComfyUI 自定义节点目录：
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  克隆本项目：
    ```bash
    git clone https://github.com/dandancow874/ComfyUI-LMStudio-Controller.git
    ```
3.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
4.  重启 ComfyUI。

### 📖 使用指南

#### 1. 基础图片分析
将 `Load Image` 节点连接到本节点的 **`image`** 端口即可。如果需要对比多张图，可连接 `image_2` 或 `image_3`。

#### 2. 视频反推工作流 (推荐)
建议配合 **VideoHelperSuite (VHS)** 插件使用：

1.  添加 **`Load Video (Upload)`** 节点 (VHS)。
2.  **关键设置**：将 VHS 的 `force_size` 设为 **Custom** (例如 512x512 或 768x768)，这一步最省显存。
3.  将 `frame_load_cap` 设为适当值 (如 32 或 64)。
4.  将 VHS 的 `IMAGE` 输出连接到本节点的 **`video_frames`** 端口。
5.  在本节点上设置 `max_total_images` (例如 8)。节点会自动从视频流中均匀抽取 8 帧发送给模型。

### 🛠️ 推荐参数设置

| 硬件/模型情况 | GPU Offload | Context Length | Max Image Side | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **8B 小模型** (如 Qwen2-VL-7B) | `1.0` (拉满) | `16384` | `768` 或 `1024` | 速度快，可处理高清图，显存无压力。 |
| **30B+ 大模型** (显存紧张) | `0.6` - `0.8` | `4096` | `512` | **防止蓝屏的关键配置**。 |
| **30B+ 大模型** (24G显存) | `0.8` - `1.0` | `8192` | `768` | 画质与显存的平衡点。 |