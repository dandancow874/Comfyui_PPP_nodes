import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter


class SeamlessCrossMaskGenerator:
    """
    将图片变成2x2拼图，并从中心创建十字蒙版用于拼图中间的接缝

    输入：图片（可选）
    输出：2x2拼图图片，十字蒙版

    菜单选项：
    - mask大小：十字蒙版的宽度（像素）
    - Mask Blur：蒙版模糊程度（像素）
    - 2x2拼图：开关，是否生成拼图图片
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_size": (
                    "INT",
                    {
                        "default": 500,
                        "min": 0,
                        "max": 1000,
                        "step": 10,
                        "label": "Mask Size (px)",
                    },
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "label": "Mask Blur (px)",
                    },
                ),
                "create_2x2_puzzle": (
                    "BOOLEAN",
                    {"default": True, "label": "Create 2x2 Puzzle"},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate_cross_mask"
    CATEGORY = "PPP_nodes/Seamless Patch"

    def generate_cross_mask(self, mask_size, mask_blur, create_2x2_puzzle, image=None):
        # 确定输出尺寸
        if image is not None and create_2x2_puzzle:
            # 图片尺寸（处理批次维度）- 2x2拼图尺寸是原图的2倍
            if image.shape[0] > 0:
                h, w, c = image[0].shape
                output_h, output_w = h * 2, w * 2
            else:
                output_h, output_w = 2048, 2048
        else:
            # 默认尺寸
            output_h, output_w = 2048, 2048

        # 生成十字蒙版
        mask = np.zeros((output_h, output_w), dtype=np.float32)

        # 计算十字位置
        center_y, center_x = output_h // 2, output_w // 2
        half_size = mask_size // 2

        # 绘制十字
        # 水平条
        y1 = center_y - half_size
        y2 = center_y + half_size
        mask[y1:y2, :] = 1.0

        # 垂直条
        x1 = center_x - half_size
        x2 = center_x + half_size
        mask[:, x1:x2] = 1.0

        # 模糊处理
        if mask_blur > 0:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        # 处理图片输出
        if image is not None and create_2x2_puzzle and image.shape[0] > 0:
            # 创建2x2拼图 - 处理批次维度
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # 创建2x2拼图（尺寸是原图的2倍）
            w, h = img_pil.size
            puzzle = Image.new("RGB", (w * 2, h * 2))

            # 粘贴四个原图
            puzzle.paste(img_pil, (0, 0))  # 左上
            puzzle.paste(img_pil, (w, 0))  # 右上
            puzzle.paste(img_pil, (0, h))  # 左下
            puzzle.paste(img_pil, (w, h))  # 右下

            # 转换为tensor
            puzzle_np = np.array(puzzle).astype(np.float32) / 255.0
            puzzle_tensor = torch.from_numpy(puzzle_np).unsqueeze(0)
        else:
            # 如果不需要拼图或没有输入图片，创建空白画布
            blank = Image.new("RGB", (output_w, output_h), (0, 0, 0))
            blank_np = np.array(blank).astype(np.float32) / 255.0
            puzzle_tensor = torch.from_numpy(blank_np).unsqueeze(0)

        # 蒙版转换为tensor
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return (puzzle_tensor, mask_tensor)


class SeamlessPatchMerger:
    """
    将传入的图片和蒙版分割成2x2，每一张图使用对应分割的蒙版，合并成一张

    输入：
        image：需要修复的图片
        mask：十字蒙版

    输出：
        image：合并后的图片
        mask：蒙版拼接到一起，用来确认蒙版合并是否出错
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("merged_image", "composite_mask")
    FUNCTION = "merge_patches"
    CATEGORY = "PPP_nodes/Seamless Patch"

    def merge_patches(self, image, mask):
        # 转换为numpy数组 - 处理批次维度
        if image.shape[0] > 0 and mask.shape[0] > 0:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = (mask[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            img_pil = Image.fromarray(img_np)
            mask_pil = Image.fromarray(mask_np)
        else:
            # 如果没有图像或蒙版，返回默认值
            h, w = 1024, 1024
            img_pil = Image.new("RGB", (w, h), (0, 0, 0))
            mask_pil = Image.new("L", (w, h), 0)

        h, w = img_pil.size
        half_h, half_w = h // 2, w // 2

        # 分割成四个象限
        quadrants = [
            (0, 0, half_w, half_h),  # 左上
            (half_w, 0, w, half_h),  # 右上
            (0, half_h, half_w, h),  # 左下
            (half_w, half_h, w, h),  # 右下
        ]

        # 分割图片和蒙版
        img_patches = []
        mask_patches = []

        for x1, y1, x2, y2 in quadrants:
            img_patch = img_pil.crop((x1, y1, x2, y2))
            mask_patch = mask_pil.crop((x1, y1, x2, y2))

            img_patches.append(img_patch)
            mask_patches.append(mask_patch)

        # 合并图片 - 使用蒙版进行混合
        merged = Image.new("RGB", img_pil.size)

        for i, (img_patch, mask_patch, (x1, y1, x2, y2)) in enumerate(
            zip(img_patches, mask_patches, quadrants)
        ):
            # 创建临时画布
            temp = Image.new("RGB", img_pil.size)
            temp.paste(img_patch, (x1, y1))

            # 使用对应象限的蒙版进行混合
            if i == 0:  # 左上
                # 只保留左上部分的蒙版
                temp_mask = Image.new("L", img_pil.size, 0)
                temp_mask.paste(mask_patch, (x1, y1))
                # 反选蒙版（只保留象限外部）
                temp_mask = ImageOps.invert(temp_mask)
            elif i == 1:  # 右上
                temp_mask = Image.new("L", img_pil.size, 0)
                temp_mask.paste(mask_patch, (x1, y1))
                temp_mask = ImageOps.invert(temp_mask)
            elif i == 2:  # 左下
                temp_mask = Image.new("L", img_pil.size, 0)
                temp_mask.paste(mask_patch, (x1, y1))
                temp_mask = ImageOps.invert(temp_mask)
            elif i == 3:  # 右下
                temp_mask = Image.new("L", img_pil.size, 0)
                temp_mask.paste(mask_patch, (x1, y1))
                temp_mask = ImageOps.invert(temp_mask)

            # 混合到最终图片
            merged = Image.composite(merged, temp, temp_mask)

        # 合并蒙版 - 用于验证
        composite_mask = Image.new("L", img_pil.size, 0)

        for mask_patch, (x1, y1, x2, y2) in zip(mask_patches, quadrants):
            composite_mask.paste(mask_patch, (x1, y1))

        # 转换回tensor
        merged_np = np.array(merged).astype(np.float32) / 255.0
        merged_tensor = torch.from_numpy(merged_np).unsqueeze(0)

        composite_mask_np = np.array(composite_mask).astype(np.float32) / 255.0
        composite_mask_tensor = torch.from_numpy(composite_mask_np).unsqueeze(0)

        return (merged_tensor, composite_mask_tensor)


class SeamlessPuzzlePreview:
    """
    创建一个3840x2160像素的无缝拼图预览

    输入接口：
        image：传入图片，默认缩放到长边1024

    节点菜单：
        拖动滑块，缩放image大小比例，拖动后会实时预览，默认为50%
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_percent": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "label": "Scale Percentage (%)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "generate_preview"
    CATEGORY = "PPP_nodes/Seamless Patch"

    def generate_preview(self, image, scale_percent):
        # 输出尺寸（3840x2160）
        OUTPUT_WIDTH = 3840
        OUTPUT_HEIGHT = 2160

        # 转换为PIL图像 - 处理批次维度
        if image.shape[0] > 0:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
        else:
            # 如果没有图像，返回空预览
            preview = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0))
            preview_np = np.array(preview).astype(np.float32) / 255.0
            preview_tensor = torch.from_numpy(preview_np).unsqueeze(0)
            return (preview_tensor,)

        # 首先将图片缩放到长边1024
        max_side = 1024
        w, h = img_pil.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 计算缩放后的尺寸
        scale_ratio = scale_percent / 100.0

        # 计算铺满整个3840x2160的所需尺寸
        # 首先计算图片的宽高比
        img_ratio = img_pil.width / img_pil.height

        # 计算铺满预览区域的尺寸
        preview_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT

        if img_ratio > preview_ratio:
            # 图片更宽，以宽度为准
            final_w = int(OUTPUT_WIDTH * scale_ratio)
            final_h = int(final_w / img_ratio)
        else:
            # 图片更高，以高度为准
            final_h = int(OUTPUT_HEIGHT * scale_ratio)
            final_w = int(final_h * img_ratio)

        # 缩放图片
        img_scaled = img_pil.resize((final_w, final_h), Image.Resampling.LANCZOS)

        # 创建预览画布（3840x2160，黑色背景）
        preview = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0))

        # 计算平铺数量，确保铺满整个预览区域
        # 计算水平和垂直方向需要平铺的次数
        tile_w = (OUTPUT_WIDTH + final_w - 1) // final_w  # 向上取整
        tile_h = (OUTPUT_HEIGHT + final_h - 1) // final_h

        # 平铺图片
        for i in range(tile_w):
            for j in range(tile_h):
                x = i * final_w
                y = j * final_h
                preview.paste(img_scaled, (x, y))

        # 转换回tensor
        preview_np = np.array(preview).astype(np.float32) / 255.0
        preview_tensor = torch.from_numpy(preview_np).unsqueeze(0)

        return (preview_tensor,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "SeamlessCrossMaskGenerator": SeamlessCrossMaskGenerator,
    "SeamlessPatchMerger": SeamlessPatchMerger,
    "SeamlessPuzzlePreview": SeamlessPuzzlePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamlessCrossMaskGenerator": "Cross Mask Generator",
    "SeamlessPatchMerger": "Patch Merger",
    "SeamlessPuzzlePreview": "Puzzle Preview",
}
