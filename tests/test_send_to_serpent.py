import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


package = types.ModuleType("ppp_test_package")
package.__path__ = [os.path.dirname(os.path.dirname(__file__))]
sys.modules.setdefault("ppp_test_package", package)
from ppp_test_package import send_to_serpent
from ppp_test_package.send_to_serpent import PPPSendToSerpent, _tag_ids, add_to_serpent


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, tool, library_id, **kwargs):
        self.calls.append((tool, library_id, kwargs))
        if tool == "serpent_file_import":
            return {"status": "completed", "completion": {"assets": [{"assetId": "asset-1"}]}}
        if tool == "serpent_asset_metadata_get":
            return {"metadata": {"entityVersion": 2}}
        if tool == "serpent_tag_list":
            return {"items": [{"name": "existing", "tagId": "tag-1"}], "hasMore": False}
        if tool == "serpent_tag_create":
            return {"id": "tag-2", "name": kwargs["name"]}
        if tool in ("serpent_asset_metadata_set", "serpent_tag_assign"):
            return {}
        raise AssertionError(tool)


class SendToSerpentTests(unittest.TestCase):
    def test_runtime_execution_list_is_hidden(self):
        self.assertEqual(PPPSendToSerpent.INPUT_TYPES()["hidden"]["execution_list"], "EXECUTION_LIST")

    def test_send_passes_runtime_prompt_to_serpent(self):
        graph = {
            "1": {"class_type": "TextNode", "inputs": {"text": "上一次的提示词"}},
            "2": {"class_type": "fast imageInputV2", "inputs": {"正面提示词": ["1", 0]}},
            "3": {"class_type": "send_to_serpent", "inputs": {"image": ["2", 0]}},
        }
        entry = types.SimpleNamespace(outputs=[["本次生成的提示词"]])
        cache = types.SimpleNamespace(get_local=lambda node_id: entry if node_id == "1" else None)
        execution_list = types.SimpleNamespace(output_cache=cache)
        fake_folder_paths = types.SimpleNamespace(get_output_directory=lambda: output)

        class FakeTensor:
            def cpu(self):
                return self

            def numpy(self):
                return np.ones((8, 8, 3))

        class FakeBatch(list):
            @property
            def shape(self):
                return (len(self), 8, 8, 3)

        with tempfile.TemporaryDirectory() as output:
            with patch.dict(sys.modules, {"folder_paths": fake_folder_paths}), \
                 patch.object(send_to_serpent, "_setting", return_value="test-token"), \
                 patch.object(send_to_serpent, "SerpentMCP"), \
                 patch.object(send_to_serpent, "add_to_serpent", return_value="asset-1") as add:
                PPPSendToSerpent().send(FakeBatch([FakeTensor()]), "library-1", "webp", "lossless", 95, "", prompt=graph, unique_id="3", execution_list=execution_list)
        self.assertTrue(add.call_args.args[3].startswith("本次生成的提示词\n"))

    def test_import_uses_explicit_library_and_sets_metadata_and_tags(self):
        client = FakeClient()
        asset = add_to_serpent(client, "library-1", r"C:\output\image.webp", "prompt\nModel: final", ["existing", "new"])
        self.assertEqual(asset, "asset-1")
        self.assertEqual([call[0] for call in client.calls], [
            "serpent_file_import", "serpent_asset_metadata_get", "serpent_asset_metadata_set",
            "serpent_tag_list", "serpent_tag_create", "serpent_tag_assign",
        ])
        self.assertTrue(all(call[1] == "library-1" for call in client.calls))
        self.assertEqual(client.calls[2][2]["expectedVersion"], 2)
        self.assertEqual(client.calls[-1][2]["tagIds"], ["tag-1", "tag-2"])

    def test_empty_tags_skip_creation(self):
        client = FakeClient()
        self.assertEqual(_tag_ids(client, "library-1", []), [])
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
