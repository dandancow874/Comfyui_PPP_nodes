import os
import torch
import numpy as np
import json5  # 需要 pip install json5
from PIL import Image, ImageOps


# 默认支持的图片扩展名
DEFAULT_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "masks", "file_parent_folders", "filenames", "original_root_ref")
    
    # 输出列表，允许不同尺寸图片混合
    OUTPUT_IS_LIST = (True, True, True, True, True)
    
    FUNCTION = "load_images"
    CATEGORY = "PPP_nodes/Batch Walker"

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
            return ([], [], [], [], [])

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
        parent_paths = []
        filenames = []
        root_paths = [] 

        for path in image_paths:
            try:
                img = Image.open(path)
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
                
                # 封装进 List
                images.append(img_tensor.unsqueeze(0)) 
                masks.append(mask.unsqueeze(0))
                
                parent_paths.append(os.path.dirname(path))
                filenames.append(os.path.basename(path))
                root_paths.append(folder_path)

            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        if not images:
            print("BatchLoader: All found images failed to load.")
            return ([], [], [], [], [])

        print(f"BatchLoader: Successfully loaded {len(images)} images.")
        return (images, masks, parent_paths, filenames, root_paths)


class BatchImageSaverRecursive:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "file_parent_folders": ("STRING", {"forceInput": True}),
                "filenames": ("STRING", {"forceInput": True}),
                "original_root_ref": ("STRING", {"forceInput": True}),
                "output_root": ("STRING", {"default": ""}),
                "format": (["png", "jpg", "webp"],),
                "compression_mode": (["lossless (无损)", "lossy (压缩)"], {"default": "lossless (无损)"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "filename_suffix": ("STRING", {"default": ""}),
                "collision_mode": (["overwrite", "skip", "rename"], {"default": "overwrite"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "PPP_nodes/Batch Walker"
    
    # 设为 False 以流式处理（每处理完一张保存一张）
    INPUT_IS_LIST = False 

    def save_images(self, images, file_parent_folders, filenames, original_root_ref, output_root, format, compression_mode, quality, filename_suffix, collision_mode):
        
        out_dir_base = output_root.strip()
        save_format = format
        suffix = filename_suffix
        mode = collision_mode
        is_lossless = "lossless" in compression_mode

        # images.shape[0] 通常为 1 (因为 INPUT_IS_LIST=False)
        for i in range(images.shape[0]):
            img_tensor = images[i]
            
            src_parent = file_parent_folders
            src_filename = filenames
            src_root = original_root_ref

            img_array = 255. * img_tensor.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            # 1. 计算相对路径
            try:
                rel_path = os.path.relpath(src_parent, src_root)
            except ValueError:
                rel_path = ""

            # 2. 确定保存目录
            if not out_dir_base or out_dir_base == "":
                target_folder = src_parent
            else:
                target_folder = os.path.join(out_dir_base, rel_path)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder, exist_ok=True)

            # 3. 构建文件名
            file_name_no_ext, _ = os.path.splitext(src_filename)
            base_new_filename = f"{file_name_no_ext}{suffix}.{save_format}"
            save_path = os.path.join(target_folder, base_new_filename)

            # 4. 冲突处理
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
            
            # 5. 保存逻辑
            try:
                if save_format == 'png':
                    # PNG 始终无损，compress_level 仅影响速度和体积
                    img_pil.save(save_path, compress_level=4)
                
                elif save_format == 'webp':
                    if is_lossless:
                        img_pil.save(save_path, lossless=True)
                    else:
                        img_pil.save(save_path, quality=quality, method=6)
                
                elif save_format == 'jpg':
                    if img_pil.mode == 'RGBA':
                        img_pil = img_pil.convert('RGB')
                    if is_lossless:
                        img_pil.save(save_path, quality=100, subsampling=0)
                    else:
                        img_pil.save(save_path, quality=quality)
                        
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
                "file_parent_folders": ("STRING", {"forceInput": True}),
                "filenames": ("STRING", {"forceInput": True}),
                "original_root_ref": ("STRING", {"forceInput": True}),
                "output_root": ("STRING", {"default": ""}),
                "extension": (["txt", "json", "md"], {"default": "txt"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "collision_mode": (["overwrite", "skip", "rename"], {"default": "overwrite"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "PPP_nodes/Batch Walker"
    INPUT_IS_LIST = False 

    def save_text(self, text_data, file_parent_folders, filenames, original_root_ref, output_root, extension, filename_suffix, collision_mode):
        
        src_parent = file_parent_folders
        src_filename = filenames
        src_root = original_root_ref
        mode = collision_mode
        suffix = filename_suffix

        try:
            rel_path = os.path.relpath(src_parent, src_root)
        except ValueError:
            rel_path = ""

        # 使用兼容性更好的写法判断路径
        out_dir_base = output_root.strip()
        if not out_dir_base or out_dir_base == "":
            target_folder = src_parent
        else:
            target_folder = os.path.join(out_dir_base, rel_path)

        if not os.path.exists(target_folder):
            os.makedirs(target_folder, exist_ok=True)

        file_name_no_ext, _ = os.path.splitext(src_filename)
        base_new_filename = f"{file_name_no_ext}{suffix}.{extension}"
        save_path = os.path.join(target_folder, base_new_filename)

        if os.path.exists(save_path):
            if mode == "skip":
                print(f"Skipping Text (Exists): {save_path}")
                return {}
            elif mode == "rename":
                counter = 1
                while os.path.exists(save_path):
                    new_name = f"{file_name_no_ext}{suffix}_{counter}.{extension}"
                    save_path = os.path.join(target_folder, new_name)
                    counter += 1

        # 使用 JSON5 进行智能格式化
        try:
            content_to_write = ""
            
            if extension == 'json':
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
                
            print(f"Saved Text ({extension}): {save_path}")
        except Exception as e:
            print(f"Error saving text {save_path}: {e}")

        return {}
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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "mask", "file_parent_folder", "filename", "original_root_ref", "file_count")
    OUTPUT_IS_LIST = (False, False, False, False, False, False)
    FUNCTION = "load_image_by_index"
    CATEGORY = "Batch Walker"

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
            return (empty_img, empty_mask, "", "", "", 0)

        # 取模循环
        safe_index = real_index % total_count
        
        target_path = image_paths[safe_index]
        print(f"BatchLoader (Pos {real_index}): Loading {safe_index + 1}/{total_count} -> {os.path.basename(target_path)}")

        try:
            img = Image.open(target_path)
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
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            mask = mask.unsqueeze(0)

            parent_dir = os.path.dirname(target_path)
            filename = os.path.basename(target_path)

            return (img_tensor, mask, parent_dir, filename, folder_path, total_count)

        except Exception as e:
            print(f"Error loading {target_path}: {e}")
            # --- 修复点在这里 ---
            # 创建一个空的 mask (1x64x64)
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            
            # 返回 empty_mask 而不是未定义的 mask
            return (empty_img, empty_mask, "", "", "", total_count)

NODE_CLASS_MAPPINGS = {
    "BatchImageLoaderRecursive": BatchImageLoaderRecursive,
    "BatchImageLoaderByIndex": BatchImageLoaderByIndex,
    "BatchImageSaverRecursive": BatchImageSaverRecursive,
    "BatchTextSaverRecursive": BatchTextSaverRecursive,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageLoaderRecursive": "Batch Loader (Recursive/List)",
    "BatchImageLoaderByIndex": "Batch Loader (Index/Single)",
    "BatchImageSaverRecursive": "Batch Saver (Image)",
    "BatchTextSaverRecursive": "Batch Saver (Text)",

}