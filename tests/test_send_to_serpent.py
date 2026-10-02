import os
import sys
import types
import unittest


package = types.ModuleType("ppp_test_package")
package.__path__ = [os.path.dirname(os.path.dirname(__file__))]
sys.modules.setdefault("ppp_test_package", package)
from ppp_test_package.send_to_serpent import _tag_ids, add_to_serpent


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
