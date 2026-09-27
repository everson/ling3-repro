#!/usr/bin/env python3
"""Bounded, stdlib-only safetensors/index/checksum verification. Not a model test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

JSON_LIMIT = 128 * 1024 * 1024
HEADER_LIMIT = 16 * 1024 * 1024
WIDTHS = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
          "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
          "I64": 8, "U64": 8, "F64": 8}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(data):
    def bad_constant(_):
        raise ValueError("nonfinite JSON constant")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def read_json(path, limit=JSON_LIMIT):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "JSON exceeds size limit")
    return decode(data)


def local_file(root, name):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name),
            "unsafe filename")
    path = root / name
    require(not path.is_symlink() and path.is_file(), "missing or symlinked file")
    return path


def tensor_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "truncated length")
        length, = struct.unpack("<Q", prefix)
        require(2 <= length <= HEADER_LIMIT, "header exceeds bounds")
        data = stream.read(length)
        require(len(data) == length, "truncated header")
        header = decode(data)
    require(isinstance(header, dict), "header must be an object")
    payload = path.stat().st_size - 8 - length
    ranges = []
    keys = set()
    for key, entry in header.items():
        if key == "__metadata__":
            require(isinstance(entry, dict) and all(isinstance(v, str) for v in entry.values()),
                    "invalid safetensors metadata")
            continue
        require(isinstance(entry, dict) and set(entry) == {"dtype", "shape", "data_offsets"},
                "invalid tensor entry")
        dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
        require(isinstance(dtype, str) and dtype in WIDTHS, "unsupported dtype")
        require(isinstance(shape, list) and len(shape) <= 32 and
                all(type(n) is int and 0 <= n <= 2**63 - 1 for n in shape), "invalid shape")
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(type(n) is int for n in offsets), "invalid offsets")
        start, end = offsets
        require(0 <= start <= end <= payload, "offset outside payload")
        require(end - start == math.prod(shape) * WIDTHS[dtype], "tensor byte count mismatch")
        ranges.append((start, end))
        keys.add(key)
    cursor = 0
    for start, end in sorted(ranges):
        require(start == cursor, "payload gap or overlap")
        cursor = end
    require(cursor == payload, "unclaimed payload bytes")
    return keys, payload


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify(root, manifest=None):
    root = Path(root)
    require(root.is_dir(), "pack directory missing")
    index_path = local_file(root, "model.safetensors.index.json")
    index = read_json(index_path)
    require(isinstance(index, dict), "invalid index")
    owners = index.get("weight_map")
    require(isinstance(owners, dict) and owners, "empty or invalid weight_map")
    require(all(isinstance(k, str) and k != "__metadata__" and isinstance(v, str)
                for k, v in owners.items()), "invalid weight_map entry")
    shards = set(owners.values())
    physical = {p.name for p in root.glob("*.safetensors")}
    require(shards == physical, "physical/index shard set mismatch")
    observed = {}
    total_bytes = 0
    for name in sorted(shards):
        keys, size = tensor_header(local_file(root, name))
        total_bytes += size
        for key in keys:
            require(key not in observed, "duplicate physical tensor")
            observed[key] = name
    require(observed == owners, "physical/index tensor ownership mismatch")
    metadata = index.get("metadata", {})
    require(isinstance(metadata, dict), "invalid index metadata")
    if "total_size" in metadata:
        require(type(metadata["total_size"]) is int and metadata["total_size"] == total_bytes,
                "index total_size mismatch")
    checked = 0
    if manifest is not None:
        expected = read_json(Path(manifest))
        require(isinstance(expected, dict) and expected, "invalid checksum manifest")
        require(shards | {index_path.name} <= set(expected), "checksum inventory incomplete")
        for name, item in expected.items():
            path = local_file(root, name)
            require(isinstance(item, dict) and set(item) == {"sha256", "size_bytes"},
                    "invalid checksum entry")
            require(type(item["size_bytes"]) is int and item["size_bytes"] >= 0 and
                    isinstance(item["sha256"], str) and
                    re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "invalid digest or size")
            require(path.stat().st_size == item["size_bytes"], "file size mismatch")
            require(checksum(path) == item["sha256"], "checksum mismatch")
            checked += 1
    return {"scope": "structure_and_supplied_checksums_only", "shards": len(shards),
            "tensors": len(observed), "payload_bytes": total_bytes,
            "checksummed_files": checked, "quality_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--checksums", type=Path,
                        help="trusted JSON filename -> {sha256, size_bytes}; includes index and every shard")
    args = parser.parse_args()
    try:
        result = verify(args.pack, args.checksums)
    except (ValueError, OSError, TypeError, OverflowError, RecursionError) as exc:
        parser.exit(1, f"verification failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
