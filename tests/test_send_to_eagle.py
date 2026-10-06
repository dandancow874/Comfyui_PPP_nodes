import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from send_to_eagle import PPPSendToEagle, _save_image, build_annotation, build_tags, extract_generation_info


def sample_graph():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "first.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["10", 0], "negative": ["11", 0]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "final.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3", 0], "lora_name": "styles\\look.safetensors", "strength_model": 0.8}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["10", 0], "negative": ["12", 0], "latent_image": ["2", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0]}},
        "7": {"class_type": "PPP_SendToEagle", "inputs": {"image": ["6", 0]}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": ["13", 0]}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad anatomy"}},
        "12": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["11", 0]}},
        "13": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "garden portrait"}},
        "20": {"class_type": "UNETLoader", "inputs": {"unet_name": "unrelated.safetensors"}},
    }


def dynamic_prompt_graph():
    return {
        "1": {"class_type": "Gemma4TE_ImageInfer", "inputs": {"提示词": ""}},
        "2": {"class_type": "ShowText|pysssss", "inputs": {"text": ["1", 0], "text_0": "上一次的提示词"}},
        "3": {"class_type": "fast prompts", "inputs": {"提示词": "另一条未选中的提示词"}},
        "4": {"class_type": "Any Switch (rgthree)", "inputs": {"any_01": ["2", 0], "any_03": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": ["4", 0]}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": "image-model.safetensors"}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "positive": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0]}},
        "9": {"class_type": "send_to_eagle", "inputs": {"image": ["8", 0]}},
    }


class FakeTensor:
    def cpu(self):
        return self

    def numpy(self):
        return np.ones((8, 8, 3))


class FakeBatch(list):
    @property
    def shape(self):
        return (len(self), 8, 8, 3)


class SendToEagleTests(unittest.TestCase):
    def test_follows_image_branch_and_model_chain(self):
        info = extract_generation_info(sample_graph(), "7")
        self.assertEqual(info["models"], ["final.safetensors", "first.safetensors"])
        self.assertEqual(info["loras"], [("look.safetensors", 0.8, None)])
        self.assertEqual(info["positive"], "garden portrait")
        self.assertEqual(info["negative"], "bad anatomy")
        self.assertEqual(build_tags(info, "manual，other\nmanual"), ["final.safetensors", "first.safetensors", "look.safetensors", "manual", "other"])
        self.assertIn("LORA: look.safetensors (model 0.8)", build_annotation(info))

    def test_embeds_workflow_in_each_format(self):
        graph = sample_graph()
        graph["13"]["inputs"]["value"] = "庭院里的女孩"
        image = Image.new("RGB", (8, 8), "red")
        with tempfile.TemporaryDirectory() as directory:
            for file_format in ("png", "jpg", "webp"):
                path = os.path.join(directory, "test." + file_format)
                _save_image(image, path, file_format, "lossy", 90, graph, {"workflow": {"nodes": []}})
                with Image.open(path) as saved:
                    if file_format == "png":
                        self.assertEqual(json.loads(saved.info["prompt"]), graph)
                        self.assertEqual(json.loads(saved.info["workflow"]), {"nodes": []})
                    else:
                        exif = saved.getexif()
                        self.assertEqual(json.loads(exif[0x0110][7:]), graph)
                        self.assertEqual(json.loads(exif[0x010F][9:]), {"nodes": []})

    def test_fastuse_generator_reads_its_loader(self):
        graph = {
            "3": {"class_type": "fast loaderV2", "inputs": {
                "加载模式": "独立模型",
                "加载模式.选择扩散模型": "qwen_image_2.1_int8_convrot.safetensors",
                "加载模式.Clip数量.Clip模型1": "qwen3vl_8b_int8_convrot.safetensors",
                "加载模式.选择VAE": "qwen_image_2.1_vae_bf16.safetensors",
            }},
            "23": {"class_type": "fast imageInputV2", "inputs": {
                "模型加载器": ["3", 0], "正面提示词": "一朵花", "负面提示词": "模糊",
            }},
            "8": {"class_type": "fast outputResult", "inputs": {"采样": ["23", 0]}},
            "31": {"class_type": "send_to_eagle", "inputs": {"image": ["8", 0]}},
        }
        info = extract_generation_info(graph, "31")
        self.assertEqual(info["models"], ["qwen_image_2.1_int8_convrot.safetensors"])
        self.assertEqual(info["positive"], "一朵花")
        self.assertEqual(info["negative"], "模糊")

    def test_fastuse_loader_in_older_webp_metadata(self):
        graph = {
            "3": {"class_type": "fast loaderV2", "inputs": {
                "????": "????",
                "????.??????": "qwen_image_2.1_int8_convrot.safetensors",
                "????.Clip??.Clip??1": "qwen3vl_8b_int8_convrot.safetensors",
                "????.??VAE": "qwen_image_2.1_vae_bf16.safetensors",
            }},
            "23": {"class_type": "fast imageInputV2", "inputs": {"?????": ["3", 0]}},
            "8": {"class_type": "fast outputResult", "inputs": {"??": ["23", 0]}},
            "31": {"class_type": "send_to_eagle", "inputs": {"image": ["8", 0]}},
        }
        self.assertEqual(
            extract_generation_info(graph, "31")["models"],
            ["qwen_image_2.1_int8_convrot.safetensors"],
        )

    def test_dynamic_prompt_uses_connected_runtime_text(self):
        graph = dynamic_prompt_graph()
        entry = types.SimpleNamespace(outputs=[["這是一個從極低角度仰視的近景鏡頭"]])
        cache = types.SimpleNamespace(get_local=lambda node_id: entry if node_id == "4" else None)
        execution_list = types.SimpleNamespace(output_cache=cache)
        self.assertEqual(extract_generation_info(graph, "9")["positive"], "")
        info = extract_generation_info(graph, "9", execution_list)
        self.assertEqual(info["positive"], "這是一個從極低角度仰視的近景鏡頭")
        self.assertEqual(info["models"], ["image-model.safetensors"])

    def test_send_passes_runtime_prompt_to_eagle(self):
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        entry = types.SimpleNamespace(outputs=[["本次生成的提示词"]])
        cache = types.SimpleNamespace(get_local=lambda node_id: entry if node_id == "4" else None)
        execution_list = types.SimpleNamespace(output_cache=cache)
        fake_folder_paths = types.SimpleNamespace(get_output_directory=lambda: output.name)
        response = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"status": "success"})
        with patch.dict(sys.modules, {"folder_paths": fake_folder_paths}), patch("send_to_eagle.requests.post", return_value=response) as post:
            PPPSendToEagle().send(FakeBatch([FakeTensor()]), "webp", "lossless", 95, "", prompt=dynamic_prompt_graph(), unique_id="9", execution_list=execution_list)
        self.assertTrue(post.call_args.kwargs["json"]["annotation"].startswith("本次生成的提示词\n"))
        self.assertEqual(PPPSendToEagle.INPUT_TYPES()["hidden"]["execution_list"], "EXECUTION_LIST")

    def test_send_passes_annotation_and_tags_to_eagle(self):
        graph = sample_graph()
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        fake_folder_paths = types.SimpleNamespace(get_output_directory=lambda: output.name)
        response = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"status": "success"})
        with patch.dict(sys.modules, {"folder_paths": fake_folder_paths}), patch("send_to_eagle.requests.post", return_value=response) as post:
            result = PPPSendToEagle().send(FakeBatch([FakeTensor()]), "webp", "lossless", 95, "mine", prompt=graph, extra_pnginfo={"workflow": {"nodes": []}}, unique_id="7")
        payload = post.call_args.kwargs["json"]
        self.assertTrue(os.path.exists(payload["path"]))
        self.assertEqual(payload["tags"], ["final.safetensors", "first.safetensors", "look.safetensors", "mine"])
        self.assertTrue(payload["annotation"].startswith("garden portrait\nNegative prompt: bad anatomy"))
        self.assertEqual(len(result["ui"]["images"]), 1)


if __name__ == "__main__":
    unittest.main()
