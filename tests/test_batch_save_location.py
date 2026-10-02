import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


class FakeTensor:
    def cpu(self):
        return self

    def numpy(self):
        return np.ones((4, 4, 3), dtype=np.float32)

    def unsqueeze(self, _):
        return FakeBatch([self])


class FakeBatch(list):
    @property
    def shape(self):
        return (len(self), 4, 4, 3)


module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "batch_nodes.py")
spec = importlib.util.spec_from_file_location("batch_nodes_test_module", module_path)
batch_nodes = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
    spec.loader.exec_module(batch_nodes)


class BatchSaveLocationTests(unittest.TestCase):
    def test_loader_passes_source_directory_to_both_savers(self):
        with tempfile.TemporaryDirectory() as root:
            parent = os.path.join(root, "nested")
            os.makedirs(parent)
            source = os.path.join(parent, "original.png")
            Image.new("RGB", (4, 4), "white").save(source)
            loader = batch_nodes.SourceInfoImageLoader()
            with patch.object(loader, "resolve_image_path", return_value=source), \
                 patch.object(loader, "resolve_root_path", return_value=root), \
                 patch.object(batch_nodes, "load_image_file", return_value=(FakeTensor(), FakeTensor(), None)):
                image, _, source_info, _ = loader.load_image("original.png")

            self.assertEqual(source_info["parent"], parent)
            batch_nodes.BatchImageSaverRecursive().save_images(
                image, source_info, "", "png", "lossless (无损)", 95, "_saved", "overwrite"
            )
            batch_nodes.BatchTextSaverRecursive().save_text(
                "notes", source_info, "   ", "txt", "_notes", "overwrite"
            )
            self.assertTrue(os.path.isfile(os.path.join(parent, "original_saved.png")))
            with open(os.path.join(parent, "original_notes.txt"), encoding="utf-8") as saved:
                self.assertEqual(saved.read(), "notes")

    def test_explicit_root_keeps_relative_subdirectory(self):
        with tempfile.TemporaryDirectory() as root:
            source_root = os.path.join(root, "source")
            parent = os.path.join(source_root, "nested")
            destination = os.path.join(root, "destination")
            self.assertEqual(
                batch_nodes.resolve_save_folder(parent, source_root, destination),
                os.path.join(destination, "nested"),
            )

    def test_empty_root_without_source_directory_fails(self):
        with self.assertRaisesRegex(ValueError, "source directory"):
            batch_nodes.resolve_save_folder("", "", "")


if __name__ == "__main__":
    unittest.main()
