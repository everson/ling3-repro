"""Synthetic bytes only; no engine, Torch, CUDA, original files or network."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("verify_pack", Path(__file__).with_name("verify_pack.py"))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class PackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shard = self.root / "model.safetensors"
        self.index = self.root / "model.safetensors.index.json"
        self.header = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        self.write_tensor()
        self.write_index()

    def write_tensor(self, payload=None):
        header = json.dumps(self.header).encode()
        self.shard.write_bytes(struct.pack("<Q", len(header)) + header +
                               (struct.pack("<ff", 1, 2) if payload is None else payload))

    def write_index(self, owners=None, size=8):
        self.index.write_text(json.dumps({"metadata": {"total_size": size},
                                         "weight_map": owners or {"x": self.shard.name}}))

    def manifest(self):
        path = self.root / "checksums.json"
        path.write_text(json.dumps({p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                            "size_bytes": p.stat().st_size}
                                    for p in [self.shard, self.index]}))
        return path

    def test_valid_and_cli_from_other_cwd(self):
        manifest = self.manifest()
        result = v.verify(self.root, manifest)
        self.assertEqual(result["tensors"], 1)
        self.assertEqual(result["checksummed_files"], 2)
        run = subprocess.run([sys.executable, "-I", "-S", "-B", str(Path(v.__file__).resolve()),
                              str(self.root), "--checksums", str(manifest)],
                             cwd=self.root, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), result)

    def test_payload_mutation(self):
        manifest = self.manifest()
        self.write_tensor(struct.pack("<ff", 3, 4))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            v.verify(self.root, manifest)

    def test_missing_checksum(self):
        manifest = self.manifest()
        manifest.write_text(json.dumps({self.shard.name: {}}))
        with self.assertRaisesRegex(ValueError, "inventory incomplete"):
            v.verify(self.root, manifest)

    def test_extra_and_missing_shards(self):
        extra = self.root / "extra.safetensors"
        extra.write_bytes(self.shard.read_bytes())
        with self.assertRaisesRegex(ValueError, "shard set mismatch"):
            v.verify(self.root)
        extra.unlink()
        self.shard.unlink()
        with self.assertRaisesRegex(ValueError, "shard set mismatch"):
            v.verify(self.root)

    def test_wrong_owner(self):
        self.write_index({"wrong": self.shard.name})
        with self.assertRaisesRegex(ValueError, "ownership mismatch"):
            v.verify(self.root)

    def test_duplicate_physical_tensor(self):
        extra = self.root / "extra.safetensors"
        extra.write_bytes(self.shard.read_bytes())
        self.write_index({"x": self.shard.name, "y": extra.name}, 16)
        with self.assertRaisesRegex(ValueError, "duplicate physical"):
            v.verify(self.root)

    def test_header_bounds(self):
        self.shard.write_bytes(struct.pack("<Q", v.HEADER_LIMIT + 1))
        with self.assertRaisesRegex(ValueError, "header exceeds"):
            v.verify(self.root)

    def test_truncation(self):
        self.write_tensor(b"x")
        with self.assertRaisesRegex(ValueError, "outside payload"):
            v.verify(self.root)

    def test_overlap_and_gap(self):
        self.header["y"] = dict(self.header["x"])
        self.write_tensor()
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            v.verify(self.root)
        self.header.pop("y")
        self.header["x"]["data_offsets"] = [1, 9]
        self.write_tensor(b"x" * 9)
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            v.verify(self.root)

    def test_shape_dtype_and_size(self):
        for field, value, error in [("shape", [True], "invalid shape"),
                                     ("shape", [3], "byte count"),
                                     ("dtype", "F4", "unsupported dtype")]:
            with self.subTest(field=field, value=value):
                self.header = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
                self.header["x"][field] = value
                self.write_tensor()
                with self.assertRaisesRegex(ValueError, error):
                    v.verify(self.root)

    def test_duplicate_json_key(self):
        self.index.write_text('{"weight_map":{"x":"a","x":"b"}}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            v.verify(self.root)

    def test_total_size(self):
        self.write_index(size=9)
        with self.assertRaisesRegex(ValueError, "total_size mismatch"):
            v.verify(self.root)

    def test_symlink_and_traversal(self):
        original = self.root / "payload.bin"
        self.shard.rename(original)
        self.shard.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            v.verify(self.root)
        with self.assertRaisesRegex(ValueError, "unsafe filename"):
            v.local_file(self.root, "../payload.bin")

    def test_bounded_json_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "size limit"):
            v.read_json(self.index, limit=2)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            v.decode('{"x":NaN}')


if __name__ == "__main__":
    unittest.main()
