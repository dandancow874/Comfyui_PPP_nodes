import torch
import numpy as np
from PIL import Image
import base64
import io
import requests
import subprocess
import json
import os
import logging
import time

logger = logging.getLogger("LMS_Controller")

class LMS_CLI_Handler:
    """
    负责与 LM Studio 命令行交互 (跨平台兼容版)
    """
    _model_cache = None
    _last_cache_time = 0
    CACHE_TTL = 10 

    @staticmethod
    def get_lms_path():
        # --- Windows 逻辑 ---
        if os.name == 'nt':
            user_home = os.path.expanduser("~")
            candidates = [
                os.path.join(user_home, ".lmstudio", "bin", "lms.exe"),
                os.path.join(user_home, "AppData", "Local", "LM-Studio", "app", "bin", "lms.exe")
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            return "lms" # 如果找不到路径，尝试直接调用命令
        
        # --- Mac/Linux 逻辑 ---
        else:
            # 在 Mac 上，只要用户点了 "Install lms to PATH"，直接用 lms 即可
            # 也可以检查一下默认路径作为兜底
            return "lms"

    @staticmethod
    def run_cmd(args, timeout=30):
        lms_path = LMS_CLI_Handler.get_lms_path()
        cmd = [lms_path] + args
        
        startupinfo = None
        # [关键修复] 只有 Windows 才需要配置 startupinfo 来隐藏黑框
        # Mac/Linux 不需要这个，加了反而会报错
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace',
                startupinfo=startupinfo # Mac下这是 None，不会报错
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)

    @classmethod
    def get_models(cls):
        # ... (get_models 的内容保持不变，直接复制之前的即可) ...
        # 为了节省篇幅，这里假设你保留了之前修正好的 get_models 代码
        if cls._model_cache and (time.time() - cls._last_cache_time < cls.CACHE_TTL):
            return cls._model_cache

        success, stdout, stderr = cls.run_cmd(["ls"], timeout=5)
        if not success:
            logger.error(f"LMS LS Error: {stderr}")
            return ["Error: lms ls failed"]

        models = []
        lines = stdout.strip().splitlines()
        BLACKLIST = {
            "size", "ram", "type", "architecture", "model", "path", 
            "llm", "llms", "embedding", "embeddings", "vision", "image",
            "name", "loading", "fetching", "downloaded", "bytes", "date",
            "publisher", "repository", "you", "have", "features", "primary", "gpu"
        }
        for line in lines:
            line = line.strip()
            if not line: continue
            if all(c in "-=*" for c in line): continue
            parts = line.split()
            if not parts: continue
            raw_name = parts[0]
            raw_lower = raw_name.lower()
            if raw_lower.rstrip(":") in BLACKLIST: continue
            if raw_lower[0].isdigit() and ("gb" in raw_lower or "mb" in raw_lower): continue
            clean_name = raw_name
            if "/" in clean_name: clean_name = clean_name.split("/")[-1]
            if clean_name.lower().endswith(".gguf"): clean_name = clean_name[:-5]
            if len(clean_name) < 2: continue
            models.append(clean_name)
        unique_models = sorted(list(set(models)))
        if not unique_models: unique_models = ["No models found"]
        cls._model_cache = unique_models
        cls._last_cache_time = time.time()
        return unique_models

    @classmethod
    def load_model(cls, model_name, identifier, gpu_ratio=1.0, context_length=2048):
        # ... (load_model 保持不变) ...
        logger.info(f"LMS: Loading '{model_name}' (GPU: {gpu_ratio}, Ctx: {context_length})...")
        gpu_arg = "max" if gpu_ratio >= 1.0 else str(gpu_ratio)
        if gpu_ratio <= 0: gpu_arg = "0"
        args = ["load", model_name, "--identifier", identifier, "--gpu", gpu_arg, "--context-length", str(context_length)]
        success, stdout, stderr = cls.run_cmd(args, timeout=180)
        if not success: logger.error(f"LMS Load Error: {stderr}")
        return success

    @classmethod
    def unload_all(cls):
        # ... (unload_all 保持不变) ...
        success, _, stderr = cls.run_cmd(["unload", "--all"], timeout=20)
        return success

class LMS_VisionController:
    _current_loaded_model = None 
    _current_gpu_ratio = 1.0
    _current_context = 2048

    def __init__(self):
        self.cli = LMS_CLI_Handler()

    @classmethod
    def INPUT_TYPES(cls):
        model_list = LMS_CLI_Handler.get_models()
        return {
            "required": {
                # [修改点1] image 已经移到了 optional，这里只保留其他必填项
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe the content of the images/video."}),
                "model_name": (model_list,),
                "max_total_images": ("INT", {"default": 8, "min": 1, "max": 64}),
                "gpu_offload": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "context_length": ("INT", {"default": 8192, "min": 512, "max": 32768}),
                "max_image_side": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # [修改点2] image 现在是可选的了！
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "video_frames": ("IMAGE",), 
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant."}),
                "base_url": ("STRING", {"default": "http://localhost:1234/v1"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    FUNCTION = "generate_content"
    CATEGORY = "PPP_nodes/LM Studio"

    def process_image(self, tensor_img, max_side):
        try:
            img_np = (tensor_img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
            width, height = pil_img.size
            if max(width, height) > max_side:
                ratio = max_side / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            # 改为 PNG 以获得更好的兼容性，虽然体积稍大
            pil_img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None

    def generate_content(self, user_prompt, model_name, max_total_images, gpu_offload, context_length, max_image_side,
                         max_tokens, temperature, seed, unload_after, 
                         image=None, image_2=None, image_3=None, video_frames=None,
                         system_prompt="", base_url="http://localhost:1234/v1", **kwargs):
        
        if "http" not in base_url: base_url = "http://localhost:1234/v1"
        IDENTIFIER = "comfy_vlm_worker"
        
        # 1. 收集图片
        all_tensors = []
        if image is not None:
            for i in range(image.shape[0]): all_tensors.append(image[i])
        if image_2 is not None:
            for i in range(image_2.shape[0]): all_tensors.append(image_2[i])
        if image_3 is not None:
            for i in range(image_3.shape[0]): all_tensors.append(image_3[i])
        if video_frames is not None:
            for i in range(video_frames.shape[0]): all_tensors.append(video_frames[i])
        
        total_count = len(all_tensors)
        
        # --- [修改点]：不再因为 total_count == 0 而报错，而是记录日志 ---
        if total_count == 0:
            logger.info("No images detected. Running in Text-Only (Chat) mode.")
        else:
            logger.info(f"Processing {total_count} images for Vision mode.")
        
        # 2. 抽帧 (仅当有图片时执行)
        final_tensors = []
        if total_count > 0:
            if total_count > max_total_images:
                indices = np.linspace(0, total_count - 1, max_total_images, dtype=int)
                final_tensors = [all_tensors[i] for i in indices]
            else:
                final_tensors = all_tensors

        # 3. 转 Base64 (仅当有图片时执行)
        image_content_list = []
        if final_tensors:
            for tensor in final_tensors:
                b64 = self.process_image(tensor, max_image_side)
                if b64:
                    image_content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "auto" # 显式添加 detail 参数
                        }
                    })

        # 4. 加载模型 (保持不变)
        needs_reload = (
            LMS_VisionController._current_loaded_model != model_name or
            abs(LMS_VisionController._current_gpu_ratio - gpu_offload) > 0.01 or 
            LMS_VisionController._current_context != context_length
        )

        if needs_reload:
            self.cli.unload_all()
            time.sleep(1.0)
            success = self.cli.load_model(model_name, IDENTIFIER, gpu_ratio=gpu_offload, context_length=context_length)
            if success:
                LMS_VisionController._current_loaded_model = model_name
                LMS_VisionController._current_gpu_ratio = gpu_offload
                LMS_VisionController._current_context = context_length
                time.sleep(2.0)
            else:
                return (f"Error: Failed to load model '{model_name}'.",)

        # 5. 构建 Payload [核心修改：区分纯文本和多模态]
        user_content = ""
        
        if len(image_content_list) > 0:
            # 视觉模式：content 是一个列表 [{"type":"text"}, {"type":"image_url"}...]
            user_content = [{"type": "text", "text": user_prompt}] + image_content_list
        else:
            # 纯文本模式：content 只是一个字符串
            # 这样兼容性最好，能支持不支持 Vision 的纯文本模型 (如 Llama 3, Mistral)
            user_content = user_prompt

        payload = {
            "model": IDENTIFIER,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False
        }

        # 6. 发送请求
        content = ""
        try:
            api_endpoint = f"{base_url.rstrip('/')}/chat/completions"
            
            # 打印日志让用户知道现在是什么模式
            mode_str = "Vision Mode" if len(image_content_list) > 0 else "Text-Only Mode"
            logger.info(f"Sending request ({mode_str})...")
            
            response = requests.post(api_endpoint, headers={"Content-Type": "application/json"}, json=payload, timeout=300)
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                else:
                    content = "Error: Empty response."
            else:
                content = f"API Error {response.status_code}: {response.text}"
                logger.error(content)
        except Exception as e:
            content = f"Connection Error: {str(e)}"
            logger.error(content)

        if unload_after:
            self.cli.unload_all()
            LMS_VisionController._current_loaded_model = None

        return (content,)

# ==========================================
# 新增功能：Prompt 管理系统
# ==========================================

# 定义 prompt 存储的根目录 (在当前节点文件夹下自动创建 prompts 文件夹)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(CURRENT_DIR, "prompts")

# 如果文件夹不存在，自动创建
if not os.path.exists(PROMPTS_DIR):
    try:
        os.makedirs(PROMPTS_DIR)
        logger.info(f"Created prompts directory at: {PROMPTS_DIR}")
    except Exception as e:
        logger.error(f"Failed to create prompts directory: {e}")

class LMS_LoadPrompt:
    """
    读取节点目录下的 prompt 文件 (.txt, .json)
    支持子文件夹，支持下拉搜索
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 每次加载节点时，遍历目录获取文件列表
        files = []
        if os.path.exists(PROMPTS_DIR):
            for root, dirs, files_in_dir in os.walk(PROMPTS_DIR):
                for file in files_in_dir:
                    if file.lower().endswith((".txt", ".json")):
                        # 获取相对路径，例如 "风格\赛博朋克.txt"
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, PROMPTS_DIR)
                        files.append(rel_path)
        
        # 排序，保证列表整齐
        files.sort()
        
        if not files:
            files = ["No prompts found.txt"]

        return {
            "required": {
                "prompt_file": (files,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_text",)
    FUNCTION = "load_file"
    CATEGORY = "PPP_nodes/Prompt"

    def load_file(self, prompt_file):
        file_path = os.path.join(PROMPTS_DIR, prompt_file)
        
        content = ""
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return ("",)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded prompt from: {prompt_file}")
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            content = f"Error reading file: {str(e)}"

        return (content,)

class LMS_SavePrompt:
    """
    保存文本到文件
    支持自动创建子文件夹 (例如输入: 词神\反推.txt)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),  # 接收来自其他节点的文本
                "filename": ("STRING", {"default": "folder/my_prompt.txt", "multiline": False}),
            },
            "optional": {
                "mode": (["overwrite", "append"],), # 覆盖模式 或 追加模式
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_text",)
    OUTPUT_NODE = True
    FUNCTION = "save_file"
    CATEGORY = "PPP_nodes/Prompt"

    def save_file(self, text, filename, mode="overwrite"):
        # 规范化路径，处理 Windows 的反斜杠
        filename = filename.replace("\\", "/")
        
        # 防止保存到父目录 (安全检查)
        if ".." in filename:
            logger.warning("Attempted path traversal. Saving to root instead.")
            filename = os.path.basename(filename)

        full_path = os.path.join(PROMPTS_DIR, filename)
        
        # 确保子文件夹存在
        directory = os.path.dirname(full_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created sub-directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory: {e}")
                return (text,)

        # 写入文件
        try:
            write_mode = 'w' if mode == "overwrite" else 'a'
            # 如果是追加模式，先加个换行符
            content_to_write = text
            if mode == "append" and os.path.exists(full_path):
                content_to_write = "\n" + text

            with open(full_path, write_mode, encoding='utf-8') as f:
                f.write(content_to_write)
            
            logger.info(f"Saved prompt to: {full_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")

        return (text,)

# ==========================================
# 注册节点 (请更新原本底部的 MAPPINGS)
# ==========================================

# 1. 找到你原本代码里的 NODE_CLASS_MAPPINGS = { ... }
# 2. 将其替换或合并为以下内容：

NODE_CLASS_MAPPINGS = {
    "LMS_VisionController": LMS_VisionController,
    "LMS_LoadPrompt": LMS_LoadPrompt,
    "LMS_SavePrompt": LMS_SavePrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMS_VisionController": "LM Studio VLM",
    "LMS_LoadPrompt": "📂 Load Prompt",
    "LMS_SavePrompt": "💾 Save Prompt"
}




