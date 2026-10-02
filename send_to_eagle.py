import json
import logging
import os
import re
import uuid
from collections import deque
from datetime import datetime

import numpy as np
import requests
from PIL import Image, PngImagePlugin


logger = logging.getLogger(__name__)
EAGLE_ADD_URL = "http://127.0.0.1:41595/api/item/addFromPath"
MODEL_FIELDS = ("ckpt_name", "unet_name", "model_name", "diffusion_model_name")
FAST_LOADER_MODEL_FIELDS = {
    "检查点模型": ("选择检查点",),
    "独立模型": ("选择扩散模型",),
    "融合模型": ("选择主模型", "选择覆盖模型"),
}


def _node(graph, node_id):
    return graph.get(str(node_id), {}) if isinstance(graph, dict) else {}


def _link(value, graph):
    if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph and isinstance(value[1], int):
        return str(value[0])
    return None


def _upstream(graph, start):
    queue = deque([(str(start), 0)])
    seen = set()
    while queue:
        node_id, distance = queue.popleft()
        if node_id in seen or node_id not in graph:
            continue
        seen.add(node_id)
        node = _node(graph, node_id)
        yield node_id, node, distance
        for value in node.get("inputs", {}).values():
            linked = _link(value, graph)
            if linked:
                queue.append((linked, distance + 1))


def _text_value(graph, value, seen=None):
    if isinstance(value, str):
        return value.strip()
    linked = _link(value, graph)
    if not linked:
        return ""
    seen = set() if seen is None else seen
    if linked in seen:
        return ""
    seen.add(linked)
    node = _node(graph, linked)
    inputs = node.get("inputs", {})
    for key in ("text", "value", "string", "prompt", "positive", "negative", "提示词"):
        if key in inputs:
            result = _text_value(graph, inputs[key], seen)
            if result:
                return result
    return ""


def _prompt_from_conditioning(graph, value):
    linked = _link(value, graph)
    if not linked:
        return ""
    for _, node, _ in _upstream(graph, linked):
        kind = node.get("class_type", "").lower()
        if "zeroout" in kind or "zero_out" in kind:
            return ""
        if "textencode" in kind or "text_encode" in kind:
            text = _text_value(graph, node.get("inputs", {}).get("text"))
            if text:
                return text
    return ""


def _basename(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1].strip()


def _lora_entries(inputs):
    if isinstance(inputs.get("lora_name"), str):
        yield inputs["lora_name"], inputs.get("strength_model", inputs.get("strength", 1)), inputs.get("strength_clip")
    for value in inputs.values():
        if isinstance(value, dict) and value.get("on", True) and isinstance(value.get("lora"), str):
            yield value["lora"], value.get("strength", value.get("strength_model", 1)), value.get("strength_clip")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("on", True) and isinstance(item.get("lora"), str):
                    yield item["lora"], item.get("strength", item.get("strength_model", 1)), item.get("strength_clip")


def _fast_loader_models(inputs):
    fields = FAST_LOADER_MODEL_FIELDS.get(inputs.get("加载模式"))
    if fields:
        for field in fields:
            for key, value in inputs.items():
                if (key == field or key.endswith("." + field)) and isinstance(value, str) and value.lower() != "none":
                    yield _basename(value)
        return

    # Older WebP exports replaced Chinese EXIF text with '?'. In this node's
    # serialized input order, the first model file is the selected base model.
    for value in inputs.values():
        if isinstance(value, str) and value.lower().endswith((".safetensors", ".ckpt", ".gguf")):
            yield _basename(value)
            return


def _model_chain(graph, start):
    queue = deque([str(start)])
    seen = set()
    models = []
    loras = []
    while queue:
        node_id = queue.popleft()
        if node_id in seen or node_id not in graph:
            continue
        seen.add(node_id)
        node = _node(graph, node_id)
        inputs = node.get("inputs", {})
        if node.get("class_type") == "fast loaderV2":
            models.extend(_fast_loader_models(inputs))
        for key in MODEL_FIELDS:
            if isinstance(inputs.get(key), str):
                models.append(_basename(inputs[key]))
                break
        for name, model_strength, clip_strength in _lora_entries(inputs):
            if name and name.lower() != "none":
                loras.append((_basename(name), model_strength, clip_strength))
        for key in ("model", "base_model", "unet", "anything", "model_in", "model1", "model2"):
            linked = _link(inputs.get(key), graph)
            if linked:
                queue.append(linked)
    return models, loras


def extract_generation_info(graph, node_id):
    """Follow the image feeding this save node, then each sampler's actual model input."""
    if not isinstance(graph, dict):
        return {"positive": "", "negative": "", "models": [], "loras": []}
    own = _node(graph, node_id)
    image_source = _link(own.get("inputs", {}).get("image"), graph)
    if not image_source:
        return {"positive": "", "negative": "", "models": [], "loras": []}

    ancestors = list(_upstream(graph, image_source))
    samplers = []
    for sid, node, distance in ancestors:
        kind = node.get("class_type", "").lower()
        inputs = node.get("inputs", {})
        if "sampler" in kind and "model" in inputs and ("positive" in inputs or "latent_image" in inputs):
            samplers.append((distance, sid, node))
    samplers.sort(key=lambda item: (item[0], item[1]))

    models, loras = [], []
    positive, negative = "", ""
    for _, _, sampler in samplers:
        inputs = sampler["inputs"]
        positive = positive or _prompt_from_conditioning(graph, inputs.get("positive"))
        negative = negative or _prompt_from_conditioning(graph, inputs.get("negative"))
        source = _link(inputs.get("model"), graph)
        if source:
            stage_models, stage_loras = _model_chain(graph, source)
            for model in stage_models:
                if model not in models:
                    models.append(model)
            for lora in stage_loras:
                if lora not in loras:
                    loras.append(lora)

    for _, node, _ in ancestors:
        if node.get("class_type") == "fast imageInputV2":
            inputs = node.get("inputs", {})
            positive = positive or _text_value(graph, inputs.get("正面提示词"))
            negative = negative or _text_value(graph, inputs.get("负面提示词"))

    if not models:
        for sid, node, _ in ancestors:
            inputs = node.get("inputs", {})
            if node.get("class_type") == "fast loaderV2" or any(key in inputs for key in MODEL_FIELDS):
                stage_models, stage_loras = _model_chain(graph, sid)
                for model in stage_models:
                    if model not in models:
                        models.append(model)
                for lora in stage_loras:
                    if lora not in loras:
                        loras.append(lora)
            else:
                for lora in _lora_entries(inputs):
                    normalized = (_basename(lora[0]), lora[1], lora[2])
                    if normalized not in loras:
                        loras.append(normalized)
    return {"positive": positive, "negative": negative, "models": models, "loras": loras}


def _strength(value):
    try:
        return f"{float(value):g}"
    except (ValueError, TypeError):
        return str(value)


def build_annotation(info):
    lines = []
    if info["positive"]:
        lines.append(info["positive"])
    lines.append("Negative prompt: " + info["negative"])
    lines.append("Model: " + ", ".join(info["models"]))
    for name, model_strength, clip_strength in info["loras"]:
        strength = f"model {_strength(model_strength)}"
        if clip_strength is not None:
            strength += f", clip {_strength(clip_strength)}"
        lines.append(f"LORA: {name} ({strength})")
    return "\n".join(lines)


def build_tags(info, manual_tags):
    tags = [*info["models"], *(name for name, _, _ in info["loras"])]
    tags.extend(tag.strip() for tag in re.split(r"[,，\n]", manual_tags or ""))
    return list(dict.fromkeys(tag for tag in tags if tag))


def _save_image(image, path, file_format, compression_mode, quality, prompt, extra_pnginfo):
    metadata = {"prompt": prompt}
    if isinstance(extra_pnginfo, dict):
        metadata.update(extra_pnginfo)
    metadata = {key: json.dumps(value, ensure_ascii=True) for key, value in metadata.items() if value is not None}

    if file_format == "png":
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            pnginfo.add_text(key, value)
        image.save(path, format="PNG", pnginfo=pnginfo, compress_level=6)
        return

    exif = Image.Exif()
    if "workflow" in metadata:
        exif[0x010F] = "workflow:" + metadata["workflow"]
    if "prompt" in metadata:
        exif[0x0110] = "prompt:" + metadata["prompt"]
    rgb = image.convert("RGB") if file_format == "jpg" else image
    if file_format == "webp":
        rgb.save(path, format="WEBP", lossless=compression_mode == "lossless", quality=quality, method=6, exif=exif)
    else:
        rgb.save(path, format="JPEG", quality=quality, subsampling=0 if quality == 100 else -1, exif=exif)


class PPPSendToEagle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "file_format": (["png", "jpg", "webp"], {"default": "webp"}),
                "compression_mode": (["lossless", "lossy"], {"default": "lossless"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "tags": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "send"
    OUTPUT_NODE = True
    CATEGORY = "PPP Nodes/Eagle"

    def send(self, image, file_format, compression_mode, quality, tags, prompt=None, extra_pnginfo=None, unique_id=None):
        import folder_paths

        if image.shape[0] == 0:
            raise ValueError("Send to Eagle: image batch is empty")
        if file_format == "jpg" and compression_mode == "lossless":
            raise ValueError("JPEG does not support lossless compression. Choose PNG/WEBP or switch to lossy.")

        info = extract_generation_info(prompt, unique_id)
        annotation = build_annotation(info)
        eagle_tags = build_tags(info, tags)
        output_dir = os.path.join(folder_paths.get_output_directory(), "PPP_Eagle")
        os.makedirs(output_dir, exist_ok=True)
        previews = []
        for index, tensor in enumerate(image):
            array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array)
            name = f"{datetime.now():%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:8]}_{index}"
            filename = f"{name}.{file_format}"
            path = os.path.join(output_dir, filename)
            _save_image(pil_image, path, file_format, compression_mode, quality, prompt, extra_pnginfo)
            try:
                response = requests.post(
                    EAGLE_ADD_URL,
                    json={"path": path, "name": name, "annotation": annotation, "tags": eagle_tags},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                if result.get("status") != "success":
                    raise RuntimeError(result.get("message") or result.get("error") or str(result))
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                raise RuntimeError(f"Send to Eagle failed; image remains at {path}: {exc}") from exc
            logger.info("Sent image to Eagle: %s", path)
            previews.append({"filename": filename, "subfolder": "PPP_Eagle", "type": "output"})
        return {"ui": {"images": previews}}


NODE_CLASS_MAPPINGS = {"send_to_eagle": PPPSendToEagle}
NODE_DISPLAY_NAME_MAPPINGS = {"send_to_eagle": "send_to_eagle"}
