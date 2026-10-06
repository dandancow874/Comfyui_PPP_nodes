import json
import logging
import os
import uuid
from datetime import datetime

import numpy as np
import requests
from PIL import Image

from .send_to_eagle import _save_image, build_annotation, build_tags, extract_generation_info


logger = logging.getLogger(__name__)
SERPENT_MCP_URL = "http://127.0.0.1:47342/mcp"


def _setting(name):
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, name)[0]
    except (ImportError, OSError):
        return ""


class SerpentMCP:
    def __init__(self, token):
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, text/event-stream",
        })
        self.session_id = None
        self.request_id = 0

    def __enter__(self):
        response = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "ComfyUI PPP Nodes", "version": "1.0"},
            },
        })
        if "result" not in response:
            raise RuntimeError("Serpent MCP initialization failed")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = self._post({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list", "params": {}})
        names = {tool.get("name") for tool in tools.get("result", {}).get("tools", [])}
        required = {"serpent_file_import", "serpent_library_inspect", "serpent_asset_metadata_get", "serpent_asset_metadata_set", "serpent_tag_list", "serpent_tag_create", "serpent_tag_assign"}
        if not required.issubset(names):
            raise RuntimeError("Serpent MCP is missing required import or metadata tools")
        return self

    def __exit__(self, *_):
        self.session.close()

    def _next_id(self):
        self.request_id += 1
        return self.request_id

    def _post(self, payload):
        headers = {"Mcp-Session-Id": self.session_id} if self.session_id else None
        response = self.session.post(SERPENT_MCP_URL, json=payload, headers=headers, timeout=60, allow_redirects=False)
        if 300 <= response.status_code < 400:
            raise RuntimeError("Serpent MCP redirected the local request")
        response.raise_for_status()
        self.session_id = response.headers.get("Mcp-Session-Id", self.session_id)
        if not response.content:
            return {}
        if "text/event-stream" in response.headers.get("Content-Type", ""):
            messages = [json.loads(line[5:].strip()) for line in response.content.decode("utf-8").split("\n") if line.startswith("data:")]
            if not messages:
                raise RuntimeError("Serpent MCP returned an empty event stream")
            result = messages[-1]
        else:
            result = response.json()
        if "error" in result:
            raise RuntimeError(f"Serpent MCP: {result['error'].get('message', result['error'])}")
        return result

    def call(self, tool_name, library_id, **arguments):
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": {"libraryId": library_id, **arguments}},
        }
        response = self._post(payload).get("result", {})
        if response.get("isError"):
            raise RuntimeError(f"Serpent {tool_name} failed: {response.get('content')}")
        if response.get("structuredContent"):
            data = response["structuredContent"]
        else:
            content = response.get("content", [])
            text = next((part.get("text") for part in content if part.get("type") == "text"), None)
            if not text:
                raise RuntimeError(f"Serpent {tool_name} returned no result")
            data = json.loads(text)
        if not data.get("ok"):
            raise RuntimeError(f"Serpent {tool_name} failed: {data.get('error', data)}")
        return data.get("result", {})


def _tag_ids(client, library_id, names):
    if not names:
        return []
    existing = {}
    offset = 0
    while True:
        page = client.call("serpent_tag_list", library_id, limit=200, offset=offset)
        for item in page.get("items", []):
            existing[item["name"]] = item["tagId"]
        if not page.get("hasMore"):
            break
        offset += len(page["items"])
    for name in names:
        if name not in existing:
            created = client.call("serpent_tag_create", library_id, name=name)
            tag = created.get("tag", created)
            tag_id = tag.get("tagId") or tag.get("id")
            if not tag_id:
                raise RuntimeError(f"Serpent did not return an ID for tag {name!r}")
            existing[name] = tag_id
    return [existing[name] for name in names]


def add_to_serpent(client, library_id, path, annotation, tags):
    imported = client.call(
        "serpent_file_import",
        library_id,
        sourceKind="files",
        sourcePaths=[path],
        idempotencyKey=str(uuid.uuid4()),
    )
    if imported.get("status") != "completed":
        raise RuntimeError(f"Serpent import did not complete: {imported.get('status', imported)}")
    assets = imported.get("completion", {}).get("assets", [])
    if len(assets) != 1 or not assets[0].get("assetId"):
        raise RuntimeError(f"Serpent did not return one imported asset: {imported}")
    asset_id = assets[0]["assetId"]
    try:
        metadata = client.call("serpent_asset_metadata_get", library_id, assetId=asset_id)["metadata"]
        client.call("serpent_asset_metadata_set", library_id, assetId=asset_id,
                    expectedVersion=metadata["entityVersion"], description=annotation)
        tag_ids = _tag_ids(client, library_id, tags)
        if tag_ids:
            client.call("serpent_tag_assign", library_id, assetIds=[asset_id], tagIds=tag_ids)
    except Exception as exc:
        raise RuntimeError(f"Serpent imported asset {asset_id}, but metadata or tags failed: {exc}") from exc
    return asset_id


class PPPSendToSerpent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "library_id": ("STRING", {"default": _setting("PPP_SERPENT_LIBRARY_ID")}),
                "file_format": (["png", "jpg", "webp"], {"default": "webp"}),
                "compression_mode": (["lossless", "lossy"], {"default": "lossless"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "tags": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID", "execution_list": "EXECUTION_LIST"},
        }

    RETURN_TYPES = ()
    FUNCTION = "send"
    OUTPUT_NODE = True
    CATEGORY = "PPP Nodes/Serpent"

    def send(self, image, library_id, file_format, compression_mode, quality, tags,
             prompt=None, extra_pnginfo=None, unique_id=None, execution_list=None):
        import folder_paths

        token = _setting("PPP_SERPENT_MCP_TOKEN").strip()
        if not token:
            raise RuntimeError("Set PPP_SERPENT_MCP_TOKEN in the ComfyUI environment and restart ComfyUI")
        if not library_id.strip():
            raise ValueError("Serpent library_id is required")
        if image.shape[0] == 0:
            raise ValueError("Send to Serpent: image batch is empty")
        if file_format == "jpg" and compression_mode == "lossless":
            raise ValueError("JPEG does not support lossless compression. Choose PNG/WEBP or switch to lossy.")

        info = extract_generation_info(prompt, unique_id, execution_list)
        annotation = build_annotation(info)
        if len(annotation) > 10000:
            raise ValueError("Serpent description limit is 10000 characters; this prompt is longer")
        serpent_tags = build_tags(info, tags)
        output_dir = os.path.join(folder_paths.get_output_directory(), "PPP_Serpent")
        os.makedirs(output_dir, exist_ok=True)
        previews = []
        with SerpentMCP(token) as client:
            client.call("serpent_library_inspect", library_id)
            for index, tensor in enumerate(image):
                array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_image = Image.fromarray(array)
                name = f"{datetime.now():%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:8]}_{index}"
                filename = f"{name}.{file_format}"
                path = os.path.join(output_dir, filename)
                _save_image(pil_image, path, file_format, compression_mode, quality, prompt, extra_pnginfo)
                try:
                    asset_id = add_to_serpent(client, library_id, path, annotation, serpent_tags)
                except Exception as exc:
                    raise RuntimeError(f"Send to Serpent failed; image remains at {path}: {exc}") from exc
                logger.info("Sent image to Serpent: asset %s", asset_id)
                previews.append({"filename": filename, "subfolder": "PPP_Serpent", "type": "output"})
        return {"ui": {"images": previews}}


NODE_CLASS_MAPPINGS = {"send_to_serpent": PPPSendToSerpent}
NODE_DISPLAY_NAME_MAPPINGS = {"send_to_serpent": "send_to_serpent"}
