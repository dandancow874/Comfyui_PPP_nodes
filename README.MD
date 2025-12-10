# ComfyUI-PPP-Nodes

## 🇨🇳 中文说明

我的自用节点, 包含以下功能:

*   **LM Studio**：集成 **LM Studio** 的命令行工具 (`lms`)，实现本地视觉大模型 (VLM) 的推理。
*   **Batch Walker**：支持批量加载图片（列表模式或索引模式）、按原目录结构保存图片、以及保存对应的文本/JSON 数据

## LM Studio

### ✨ 主要功能

*   **智能模型管理**：在下拉菜单切换模型时，自动检测并卸载旧模型，加载新模型。
*   **显存友好**：提供 "Unload After"（运行后卸载）选项，生成完成后立即释放显存。
*   **参数全掌控**：支持调节 Temperature, Top P, Top K, 重复惩罚 (Repetition Penalty) 等参数。
*   **CLI 深度集成**：通过 `lms ls` 自动获取你本地下载的所有模型列表。
*   **列表自动清洗**：自动过滤掉无关的系统文本，只显示清晰的模型名称。

### ⚙️ 前置要求 (重要)

1.  **安装 LM Studio**: 请前往 [lmstudio.ai](https://lmstudio.ai/) 下载。
2.  **启用 CLI 工具**: 
    *   打开 LM Studio。
    *   进入 **"Developer" (开发者)** 选项卡（或设置页面）。
    *   点击 **"Install lms to PATH"**（确保在终端输入 `lms` 能看到输出）。
3.  **启动本地服务 (Local Server)**:
    *   在 LM Studio 中，点击左侧的 **"Local Server"** 图标。
    *   点击 **"Start Server"** (默认端口 1234)。
    *   *注意：本节点使用 CLI 加载模型，但通过 API 端口发送图像请求，所以服务必须处于开启状态。*


### 🚀 使用指南

1.  在 ComfyUI 中搜索节点 **"LM Studio Vision Controller"**。
2.  连接图像 (Image) 和提示词 (Prompt)。
3.  **Model Name**: 从下拉菜单中选择一个视觉模型（列表会自动同步 LM Studio 中已下载的模型）。
4.  **Unload After**: 如果你的显存紧张，建议开启此选项，它会在任务完成后自动卸载模型。


## Batch Walker

## ✨ 主要功能

*   **📂 递归遍历**: 自动扫描文件夹及其所有子文件夹。
*   **🏗️ 保持结构**: 保存时自动计算相对路径，完美复刻源目录结构（例如 `In/A/1.png` -> `Out/A/1.png`）。
*   **🚀 两种加载模式**:
    *   **List 模式**: 一次性加载所有图片（适合小批量，下游节点需支持 List）。
    *   **Index (Single) 模式**: 配合 Batch Count 每次只加载一张，**内存占用极低**，适合处理数万张图片。
*   **💾 多功能保存**:
    *   **图片**: 支持 PNG/JPG/WEBP，支持无损/有损压缩切换。
    *   **文本**: 支持保存 TXT/MD，支持将 Python 字典自动格式化为标准 **JSON**。
*   **🔍 智能过滤**: 支持自定义后缀名过滤，支持多种分隔符。



---