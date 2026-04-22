import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw


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
        # 处理图片
        has_image = image is not None and image.shape[0] > 0
        if has_image:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            w, h = img_pil.size
        else:
            h, w = 512, 512
            img_pil = Image.new("RGB", (w, h), (0, 0, 0))

        if create_2x2_puzzle:
            # 2x2拼接模式：输出尺寸 = 原图×2
            output_w, output_h = w * 2, h * 2

            # 生成2x2拼图
            puzzle = Image.new("RGB", (output_w, output_h), (0, 0, 0))
            puzzle.paste(img_pil, (0, 0))
            puzzle.paste(img_pil, (w, 0))
            puzzle.paste(img_pil, (0, h))
            puzzle.paste(img_pil, (w, h))

            # 十字蒙版在中心
            mask = np.zeros((output_h, output_w), dtype=np.float32)
            half = mask_size // 2
            cy, cx = output_h // 2, output_w // 2
            mask[cy - half:cy + half, :] = 1.0
            mask[:, cx - half:cx + half] = 1.0
        else:
            # 不做拼接：输出原图，蒙版十字在中心
            output_w, output_h = w, h
            puzzle = img_pil.copy()

            # 十字蒙版在中心
            mask = np.zeros((h, w), dtype=np.float32)
            half = mask_size // 2
            cy, cx = h // 2, w // 2
            mask[cy - half:cy + half, :] = 1.0
            mask[:, cx - half:cx + half] = 1.0

        # 蒙版模糊
        if mask_blur > 0:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        puzzle_np = np.array(puzzle).astype(np.float32) / 255.0
        puzzle_tensor = torch.from_numpy(puzzle_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return (puzzle_tensor, mask_tensor)


class ImageTiler2x2:
    """
    将图片2x2拼接，中间留出间隔用于生成十字遮罩

    输入：
        image：传入图片

    选项：
        gap：间隔大小（像素，0-1000）
        extra_padding：接缝向原图边缘扩展（像素，0-1000），遮住原图边缘用于AI修复
        bg_color_r/g/b：背景色（默认绿色 0, 255, 0）

    输出：
        image：2x2拼接后的图片（输出尺寸 = 2*原图 + gap）
        mask：十字接缝蒙版（总宽度 = gap + extra_padding*2）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "label": "Gap (px)",
                    },
                ),
                "extra_padding": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "label": "Extra Padding (px)",
                    },
                ),
                "bg_color_r": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "label": "BG Red",
                    },
                ),
                "bg_color_g": (
                    "INT",
                    {
                        "default": 255,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "label": "BG Green",
                    },
                ),
                "bg_color_b": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "label": "BG Blue",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "tile_image"
    CATEGORY = "PPP_nodes/Seamless Patch"

    def tile_image(self, image, gap, extra_padding, bg_color_r, bg_color_g, bg_color_b):
        # 处理批次维度
        if image.shape[0] == 0:
            h, w = 512, 512
            img_pil = Image.new("RGB", (w, h), (0, 0, 0))
        else:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

        w, h = img_pil.size
        bg_color = (bg_color_r, bg_color_g, bg_color_b)

        # 输出尺寸: 2*原图 + gap
        out_w = w * 2 + gap
        out_h = h * 2 + gap

        # 创建画布，用背景色填充
        tiled = Image.new("RGB", (out_w, out_h), bg_color)

        # 粘贴四个原图
        tiled.paste(img_pil, (0, 0))  # 左上
        tiled.paste(img_pil, (w + gap, 0))  # 右上
        tiled.paste(img_pil, (0, h + gap))  # 左下
        tiled.paste(img_pil, (w + gap, h + gap))  # 右下

        # 如果有 extra_padding，用背景色覆盖原图边缘（扩大接缝区域）
        if extra_padding > 0:
            pad = extra_padding
            # 水平条带（覆盖原图上下边缘 + 中间间隔）
            ImageDraw.Draw(tiled).rectangle(
                [0, h - pad, out_w, h + gap + pad], fill=bg_color
            )
            # 垂直条带（覆盖原图左右边缘 + 中间间隔）
            ImageDraw.Draw(tiled).rectangle(
                [w - pad, 0, w + gap + pad, out_h], fill=bg_color
            )

        # 生成十字蒙版
        mask = np.zeros((out_h, out_w), dtype=np.float32)
        # 水平条带: y 从 h-extra_padding 到 h+gap+extra_padding
        mask[max(0, h - extra_padding) : min(out_h, h + gap + extra_padding), :] = 1.0
        # 垂直条带: x 从 w-extra_padding 到 w+gap+extra_padding
        mask[:, max(0, w - extra_padding) : min(out_w, w + gap + extra_padding)] = 1.0

        # 转换为tensor
        tiled_np = np.array(tiled).astype(np.float32) / 255.0
        tiled_tensor = torch.from_numpy(tiled_np).unsqueeze(0)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return (tiled_tensor, mask_tensor)


class SeamlessPatchMerger:
    """
    修正无缝拼贴图案：从修好的2x2图中提取边缘，贴回单个tile

    输入：
        fixed_image：2x2拼接后修复好接缝的图片
        mask：十字接缝蒙版

    处理逻辑：
        输出单个tile（原图尺寸），从四个象限各取一条边缘贴回：
        - BL上边缘 → tile上边缘（覆盖水平接缝）
        - TL下边缘 → tile下边缘（覆盖水平接缝）
        - TR左边缘 → tile左边缘（覆盖垂直接缝）
        - BR右边缘 → tile右边缘（覆盖垂直接缝）

    输出：
        image：无缝拼贴的单个tile
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fixed_image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge_patches"
    CATEGORY = "PPP_nodes/Seamless Patch"

    def merge_patches(self, fixed_image, mask):
        if fixed_image.shape[0] == 0 or mask.shape[0] == 0:
            blank = Image.new("RGB", (512, 512), (0, 0, 0))
            blank_np = np.array(blank).astype(np.float32) / 255.0
            return (torch.from_numpy(blank_np).unsqueeze(0),)

        fixed_np = (fixed_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        fixed_pil = Image.fromarray(fixed_np)
        mask_np = mask[0].cpu().numpy()

        Fx, Fy = fixed_pil.size

        # tile尺寸（象限尺寸）
        W = Fx // 2
        H = Fy // 2

        # 从修好图裁剪四个象限
        tl = fixed_pil.crop((0, 0, W, H))
        tr = fixed_pil.crop((W, 0, Fx, H))
        bl = fixed_pil.crop((0, H, W, Fy))
        br = fixed_pil.crop((W, H, Fx, Fy))

        # 从mask裁剪四个象限的蒙版
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_tl = mask_pil.crop((0, 0, W, H))
        mask_tr = mask_pil.crop((W, 0, Fx, H))
        mask_bl = mask_pil.crop((0, H, W, Fy))
        mask_br = mask_pil.crop((W, H, Fx, Fy))

        # TL作为基础，用蒙版把相邻象限的接缝内容混合进来
        result = tl.copy()
        result = Image.composite(result, bl, mask_bl)  # BL的上边缘
        result = Image.composite(result, tr, mask_tr)  # TR的左边缘
        result = Image.composite(result, br, mask_br)  # BR的右边缘

        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        return (result_tensor,)


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


class SeamlessTileCropper:
    """
    从2x2大图中心精确裁剪出无缝tile

    原理：从十字接缝交叉点裁剪 W×H 的区域，
    四条边都穿过修复过的接缝区域，自然无缝。

    输入：
        image：修复好接缝的2x2图片 (2W × 2H)

    输出：
        image：无缝tile (W × H)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_tile"
    CATEGORY = "PPP_nodes/Seamless Patch"

    def crop_tile(self, image):
        if image.shape[0] == 0:
            blank = Image.new("RGB", (512, 512), (0, 0, 0))
            blank_np = np.array(blank).astype(np.float32) / 255.0
            return (torch.from_numpy(blank_np).unsqueeze(0),)

        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        Fx, Fy = img_pil.size
        W = Fx // 2
        H = Fy // 2

        # 从中心裁剪：十字路口的四个角就是每个象限修复好的接缝部分
        # ┌───┬───┐
        # │   │   │
        # ├───┼───┤  ← 裁剪区域在正中心
        # │   │   │
        # └───┴───┘
        x1 = W // 2
        y1 = H // 2
        tile = img_pil.crop((x1, y1, x1 + W, y1 + H))

        result_np = np.array(tile).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        return (result_tensor,)


class SeamlessOffsetFilter:
    """
    模仿Photoshop位移滤镜：将图像边缘移动到中心，暴露接缝供修复

    原理：用np.roll将像素滚动，原图的四条边缘移到画面中心形成十字。
    修复中心的接缝后，再反向位移回来就得到无缝tile。

    输入：
        image：图片
        offset_x：水平位移（像素），默认为图片宽度的一半
        offset_y：垂直位移（像素），默认为图片高度的一半

    输出：
        image：位移后的图片
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "offset_x": (
                    "INT",
                    {
                        "default": 0,
                        "min": -8192,
                        "max": 8192,
                        "step": 1,
                        "label": "Horizontal Offset (px)",
                    },
                ),
                "offset_y": (
                    "INT",
                    {
                        "default": 0,
                        "min": -8192,
                        "max": 8192,
                        "step": 1,
                        "label": "Vertical Offset (px)",
                    },
                ),
                "use_half": (
                    "BOOLEAN",
                    {"default": True, "label": "Auto Half (offset_x/y ignored)"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_offset"
    CATEGORY = "PPP_nodes/Seamless Patch"

    def apply_offset(self, image, offset_x, offset_y, use_half):
        if image.shape[0] == 0:
            blank = Image.new("RGB", (512, 512), (0, 0, 0))
            blank_np = np.array(blank).astype(np.float32) / 255.0
            return (torch.from_numpy(blank_np).unsqueeze(0),)

        img_np = image[0].cpu().numpy()
        h, w, c = img_np.shape

        if use_half:
            offset_x = w // 2
            offset_y = h // 2

        # np.roll: 沿轴滚动像素
        # axis=1 (水平): 正值→像素右移，左边缘移到中心
        # axis=0 (垂直): 正值→像素下移，上边缘移到中心
        result = np.roll(img_np, shift=offset_x, axis=1)
        result = np.roll(result, shift=offset_y, axis=0)

        result_tensor = torch.from_numpy(result).unsqueeze(0)

        return (result_tensor,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "SeamlessCrossMaskGenerator": SeamlessCrossMaskGenerator,
    "ImageTiler2x2": ImageTiler2x2,
    "SeamlessPatchMerger": SeamlessPatchMerger,
    "SeamlessTileCropper": SeamlessTileCropper,
    "SeamlessOffsetFilter": SeamlessOffsetFilter,
    "SeamlessPuzzlePreview": SeamlessPuzzlePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamlessCrossMaskGenerator": "Cross Mask Generator",
    "ImageTiler2x2": "Image Tiler 2x2",
    "SeamlessPatchMerger": "Patch Merger",
    "SeamlessTileCropper": "Tile Cropper",
    "SeamlessOffsetFilter": "Offset Filter",
    "SeamlessPuzzlePreview": "Puzzle Preview",
}
