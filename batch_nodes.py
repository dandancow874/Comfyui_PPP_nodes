import os
import hashlib
import torch
import numpy as np
import json5  # 需要 pip install json5
from PIL import Image, ImageOps

try:
    import folder_paths
except Exception:
    folder_paths = None


# 默认支持的图片扩展名
DEFAULT_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
SOURCE_INFO_TYPE = "PPP_SOURCE_INFO"
BATCH_CATEGORY = "🧩 PPP_nodes/Batch Walker"


def make_source_info(path, root):
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root) if root else os.path.dirname(abs_path)
    parent = os.path.dirname(abs_path)

    try:
        rel_parent = os.path.relpath(parent, abs_root)
        if rel_parent == ".":
            rel_parent = ""
    except ValueError:
        rel_parent = ""

    return {
        "path": abs_path,
        "parent": parent,
        "filename": os.path.basename(abs_path),
        "root": abs_root,
        "relative_parent": rel_parent,
    }


def normalize_source_info(source_info, index=0):
    if isinstance(source_info, (list, tuple)):
        if not source_info:
            raise ValueError("source_info is empty.")
        source_info = source_info[min(index, len(source_info) - 1)]

    if isinstance(source_info, str):
        try:
            source_info = json5.loads(source_info)
        except Exception as exc:
            raise ValueError("source_info string is not valid JSON/JSON5.") from exc

    if not isinstance(source_info, dict):
        raise TypeError(f"source_info must be a dict, got {type(source_info).__name__}.")

    parent = source_info.get("parent") or os.path.dirname(source_info.get("path", ""))
    filename = source_info.get("filename") or os.path.basename(source_info.get("path", ""))
    root = source_info.get("root") or parent

    if not filename:
        raise ValueError("source_info must contain filename or path.")

    return parent, filename, root


def default_output_folder():
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.getcwd()


def load_image_file(path):
    img = Image.open(path)
    icc = img.info.get('icc_profile')
    img = ImageOps.exif_transpose(img)

    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        img_rgb = Image.merge('RGB', (r, g, b))
        mask = np.array(a).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
    else:
        img_rgb = img.convert('RGB')
        mask = torch.ones((img_rgb.height, img_rgb.width), dtype=torch.float32, device="cpu")

    img_np = np.array(img_rgb).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np)
    return img_tensor, mask, icc

class BatchImageLoaderRecursive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:\\path\\to\\images"}),
                "extensions": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g. .png,.jpg (Empty = All Images)"}),
            },
            "optional": {
                "batch_limit": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "label": "Limit (0=All)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", SOURCE_INFO_TYPE, "ICC_PROFILE")
    RETURN_NAMES = ("images", "masks", "source_infos", "icc_profiles")
    
    # 输出列表，允许不同尺寸图片混合
    OUTPUT_IS_LIST = (True, True, True, True)
    
    FUNCTION = "load_images"
    CATEGORY = BATCH_CATEGORY

    # 强制每次运行都检查文件夹变化
    @classmethod
    def IS_CHANGED(s, folder_path, extensions, batch_limit):
        return float("NaN")

    def load_images(self, folder_path, extensions, batch_limit):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")

        # 1. 过滤格式
        if not extensions or extensions.strip() == "":
            allowed_exts = DEFAULT_IMAGE_EXTENSIONS
            print(f"BatchLoader: No filter specified, looking for: {allowed_exts}")
        else:
            allowed_exts = tuple(ext.strip().lower() for ext in extensions.split(',') if ext.strip())
            print(f"BatchLoader: Filter active, looking for: {allowed_exts}")
        
        # 2. 递归遍历
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(allowed_exts):
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            print(f"BatchLoader: No images found in {folder_path}")
            return ([], [], [], [])

        image_paths.sort()
        total_found = len(image_paths)

        # 3. 数量限制
        if batch_limit > 0:
            image_paths = image_paths[:batch_limit]
            print(f"BatchLoader: Found {total_found} files. Limit active: loading first {len(image_paths)}.")
        else:
            print(f"BatchLoader: Found {total_found} files. Loading all.")

        images = []
        masks = []
        source_infos = []
        icc_profiles = []

        for path in image_paths:
            try:
                img_tensor, mask, icc = load_image_file(path)
                
                # 封装进 List
                images.append(img_tensor.unsqueeze(0)) 
                masks.append(mask.unsqueeze(0))
                source_infos.append(make_source_info(path, folder_path))
                icc_profiles.append(icc)

            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        if not images:
            print("BatchLoader: All found images failed to load.")
            return ([], [], [], [])

        print(f"BatchLoader: Successfully loaded {len(images)} images.")
        return (images, masks, source_infos, icc_profiles)


class BatchImageSaverRecursive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "source_info": (SOURCE_INFO_TYPE, {"forceInput": True}),
                "output_root": ("STRING", {"default": ""}),
                "format": (["auto", "png", "jpg", "webp"],),
                "compression_mode": (["lossless (无损)", "lossy (压缩)"], {"default": "lossless (无损)"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "filename_suffix": ("STRING", {"default": ""}),
                "collision_mode": (["overwrite", "skip", "rename"], {"default": "overwrite"}),
            },
            "optional": {
                "icc_profile": ("ICC_PROFILE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = BATCH_CATEGORY
    
    # 设为 False 以流式处理（每处理完一张保存一张）
    INPUT_IS_LIST = False 

    def save_images(self, images, source_info, output_root, format, compression_mode, quality, filename_suffix, collision_mode, icc_profile=None):
        
        out_dir_base = output_root.strip()
        
        suffix = filename_suffix
        mode = collision_mode
        is_lossless = "lossless" in compression_mode

        # images.shape[0] 通常为 1 (因为 INPUT_IS_LIST=False)
        for i in range(images.shape[0]):
            img_tensor = images[i]
            
            src_parent, src_filename, src_root = normalize_source_info(source_info, i)

            img_array = 255. * img_tensor.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            # 1. 计算相对路径
            try:
                rel_path = os.path.relpath(src_parent, src_root) if src_parent and src_root else ""
            except ValueError:
                rel_path = ""

            # 2. 确定保存目录
            if not out_dir_base or out_dir_base == "":
                target_folder = src_parent or default_output_folder()
            else:
                target_folder = os.path.join(out_dir_base, rel_path)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder, exist_ok=True)

            # 3. 确定保存格式
            if format == "auto":
                _, ext = os.path.splitext(src_filename)
                save_format = ext.lower().lstrip('.')
                if save_format == 'jpeg': 
                    save_format = 'jpg'
                if not save_format:
                    save_format = 'png' # Default if no extension
            else:
                save_format = format

            # 4. 构建文件名
            file_name_no_ext, _ = os.path.splitext(src_filename)
            base_new_filename = f"{file_name_no_ext}{suffix}.{save_format}"
            save_path = os.path.join(target_folder, base_new_filename)

            # 5. 冲突处理
            if os.path.exists(save_path):
                if mode == "skip":
                    print(f"Skipping (Exists): {save_path}")
                    return {} 
                elif mode == "rename":
                    counter = 1
                    while os.path.exists(save_path):
                        new_name = f"{file_name_no_ext}{suffix}_{counter}.{save_format}"
                        save_path = os.path.join(target_folder, new_name)
                        counter += 1
            
            # 6. 保存逻辑
            save_kwargs = {}
            if icc_profile is not None:
                save_kwargs["icc_profile"] = icc_profile

            try:
                if save_format == 'png':
                    # PNG 始终无损，compress_level 仅影响速度和体积
                    img_pil.save(save_path, compress_level=4, **save_kwargs)
                
                elif save_format == 'webp':
                    if is_lossless:
                        img_pil.save(save_path, lossless=True, **save_kwargs)
                    else:
                        img_pil.save(save_path, quality=quality, method=6, **save_kwargs)
                
                elif save_format == 'jpg':
                    if img_pil.mode == 'RGBA':
                        img_pil = img_pil.convert('RGB')
                    if is_lossless:
                        img_pil.save(save_path, quality=100, subsampling=0, **save_kwargs)
                    else:
                        img_pil.save(save_path, quality=quality, **save_kwargs)
                else:
                    # Fallback for other formats
                     img_pil.save(save_path, **save_kwargs)
                        
                print(f"Saved Image: {save_path}")
            except Exception as e:
                print(f"Error saving {save_path}: {e}")

        return {}


class BatchTextSaverRecursive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_data": ("STRING", {"forceInput": True}),
                "source_info": (SOURCE_INFO_TYPE, {"forceInput": True}),
                "output_root": ("STRING", {"default": ""}),
                "extension": (["txt", "json", "md"], {"default": "txt"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "collision_mode": (["overwrite", "skip", "rename"], {"default": "overwrite"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = BATCH_CATEGORY
    INPUT_IS_LIST = False 

    def save_text(self, text_data, source_info, output_root, extension, filename_suffix, collision_mode):
        
        src_parent, src_filename, src_root = normalize_source_info(source_info)
        mode = collision_mode
        suffix = filename_suffix

        try:
            rel_path = os.path.relpath(src_parent, src_root) if src_parent and src_root else ""
        except ValueError:
            rel_path = ""

        # 使用兼容性更好的写法判断路径
        out_dir_base = output_root.strip()
        if not out_dir_base or out_dir_base == "":
            target_folder = src_parent or default_output_folder()
        else:
            target_folder = os.path.join(out_dir_base, rel_path)

        if not os.path.exists(target_folder):
            os.makedirs(target_folder, exist_ok=True)

        save_extension = extension

        file_name_no_ext, _ = os.path.splitext(src_filename)
        base_new_filename = f"{file_name_no_ext}{suffix}.{save_extension}"
        save_path = os.path.join(target_folder, base_new_filename)

        if os.path.exists(save_path):
            if mode == "skip":
                print(f"Skipping Text (Exists): {save_path}")
                return {}
            elif mode == "rename":
                counter = 1
                while os.path.exists(save_path):
                    new_name = f"{file_name_no_ext}{suffix}_{counter}.{save_extension}"
                    save_path = os.path.join(target_folder, new_name)
                    counter += 1

        # 使用 JSON5 进行智能格式化
        try:
            content_to_write = ""
            
            if save_extension == 'json':
                try:
                    # 尝试用 json5 解析 (能处理单引号字典字符串)
                    if isinstance(text_data, str):
                        data_obj = json5.loads(text_data)
                    else:
                        data_obj = text_data
                        
                    # 重新 Dump 为漂亮的 JSON
                    content_to_write = json5.dumps(data_obj, indent=4, quote_keys=True)
                except Exception:
                    # 解析失败，按原文本保存
                    content_to_write = str(text_data)
            else:
                content_to_write = str(text_data)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content_to_write)
                
            print(f"Saved Text ({save_extension}): {save_path}")
        except Exception as e:
            print(f"Error saving text {save_path}: {e}")

        return {}


class SourceInfoImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        input_dir = folder_paths.get_input_directory() if folder_paths is not None else os.getcwd()
        if os.path.isdir(input_dir):
            for root, dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    if filename.lower().endswith(DEFAULT_IMAGE_EXTENSIONS):
                        full_path = os.path.join(root, filename)
                        files.append(os.path.relpath(full_path, input_dir))

        files = sorted(files)
        if not files:
            files = ["None"]

        return {
            "required": {
                "image": (files, {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", SOURCE_INFO_TYPE, "ICC_PROFILE")
    RETURN_NAMES = ("image", "mask", "source_info", "icc_profile")
    FUNCTION = "load_image"
    CATEGORY = BATCH_CATEGORY

    @classmethod
    def IS_CHANGED(s, image):
        path = s.resolve_image_path(image)
        if not path or not os.path.exists(path):
            return float("NaN")

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def resolve_image_path(image):
        if image == "None":
            return None

        if folder_paths is not None:
            return folder_paths.get_annotated_filepath(image)

        return os.path.abspath(image)

    @staticmethod
    def resolve_root_path(image, path):
        if folder_paths is None:
            return os.path.dirname(path)

        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
        temp_dir = folder_paths.get_temp_directory()

        annotated = str(image)
        if annotated.endswith("[output]"):
            return output_dir
        if annotated.endswith("[temp]"):
            return temp_dir
        return input_dir

    def load_image(self, image):
        path = self.resolve_image_path(image)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {image}")

        img_tensor, mask, icc = load_image_file(path)
        source_info = {"filename": os.path.basename(path)}
        return (img_tensor.unsqueeze(0), mask.unsqueeze(0), source_info, icc)


class BatchImageLoaderByIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:\\path\\to\\images"}),
                "extensions": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g. .png,.jpg"}),
                # index (起始索引)
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "label": "Start Index"}),
                # seed (步进/增量)
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "label": "Stepper (Auto)"}),
            },
            "optional": {
                "batch_limit": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}), 
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", SOURCE_INFO_TYPE, "INT", "ICC_PROFILE")
    RETURN_NAMES = ("image", "mask", "source_info", "file_count", "icc_profile")
    OUTPUT_IS_LIST = (False, False, False, False, False)
    FUNCTION = "load_image_by_index"
    CATEGORY = BATCH_CATEGORY

    @classmethod
    def IS_CHANGED(s, folder_path, extensions, index, seed, batch_limit):
        return float("NaN")

    def load_image_by_index(self, folder_path, extensions, index, seed, batch_limit):
        # 索引计算
        real_index = index + seed
        
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")

        if not extensions or extensions.strip() == "":
            allowed_exts = DEFAULT_IMAGE_EXTENSIONS
        else:
            normalized_input = extensions.replace(' ', ',').replace(';', ',').replace('，', ',')
            processed_exts = []
            for ext in normalized_input.split(','):
                clean_ext = ext.strip().lower()
                if clean_ext:
                    if not clean_ext.startswith('.'):
                        clean_ext = '.' + clean_ext
                    processed_exts.append(clean_ext)
            allowed_exts = tuple(processed_exts)
        
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(allowed_exts):
                    image_paths.append(os.path.join(root, file))

        image_paths.sort()

        if batch_limit > 0:
            image_paths = image_paths[:batch_limit]
            if real_index == 0:
                print(f"BatchLoader: Limit active. Pool size: {len(image_paths)}")

        total_count = len(image_paths)

        if total_count == 0:
            print(f"BatchLoader: No images found.")
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            return (empty_img, empty_mask, {}, 0, None)

        # 取模循环
        safe_index = real_index % total_count
        
        target_path = image_paths[safe_index]
        print(f"BatchLoader (Pos {real_index}): Loading {safe_index + 1}/{total_count} -> {os.path.basename(target_path)}")

        try:
            img_tensor, mask, icc = load_image_file(target_path)
            img_tensor = img_tensor.unsqueeze(0)
            mask = mask.unsqueeze(0)
            source_info = make_source_info(target_path, folder_path)

            return (img_tensor, mask, source_info, total_count, icc)

        except Exception as e:
            print(f"Error loading {target_path}: {e}")
            # --- 修复点在这里 ---
            # 创建一个空的 mask (1x64x64)
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            
            # 返回 empty_mask 而不是未定义的 mask
            return (empty_img, empty_mask, {}, total_count, None)

NODE_CLASS_MAPPINGS = {
    "SourceInfoImageLoader": SourceInfoImageLoader,
    "BatchImageLoaderRecursive": BatchImageLoaderRecursive,
    "BatchImageLoaderByIndex": BatchImageLoaderByIndex,
    "BatchImageSaverRecursive": BatchImageSaverRecursive,
    "BatchTextSaverRecursive": BatchTextSaverRecursive,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SourceInfoImageLoader": "📥 PPP Load Image",
    "BatchImageLoaderRecursive": "📂 Batch Loader (Recursive/List)",
    "BatchImageLoaderByIndex": "🔢 Batch Loader (Index/Single)",
    "BatchImageSaverRecursive": "💾 Batch Saver (Image)",
    "BatchTextSaverRecursive": "📝 Batch Saver (Text)",

}
