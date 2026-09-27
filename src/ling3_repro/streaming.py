"""Reviewed-original Ling target evaluator; independent Torch, NOT a serving engine.

D01 is post-SiLU clipping, not the unclamped downloaded HF forward. Model Python
is never imported. Only config/index JSON and safetensors primitives are read.
See README.md for rounding, provenance, memory bounds and qualification limits.
The caller must exclusively own process-global numerical policy for the entire
forward/logit/capture lifecycle, including callbacks and generator suspension.
Boundary checks reject observed drift; they cannot detect another thread briefly
changing and restoring global flags during an operation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

if __package__:
    from .precision_policy import observe_precision
    from .math_oracles import (causal_conv, gated_head_norm, interleaved_rope,
                               kda_recurrence, safe_kda_decay, swiglu)
else:
    from precision_policy import observe_precision
    from math_oracles import (causal_conv, gated_head_norm, interleaved_rope,
                              kda_recurrence, safe_kda_decay, swiglu)

OFFICIAL_REVISION = "ef06d91fe382109ae82647da88ff99b0f11745b0"
OFFICIAL_REPO = "inclusionAI/Ling-3.0-flash"
FIXTURE_LABEL = "deterministic synthetic Ling fixture; NOT original model evidence"
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
REVIEWED_METADATA_SHA256 = {'config.json': '6c1bd3c25e8b2db7ed954bd08291fa706ab2596a16b2cbab397ca75d44979c3b', 'model.safetensors.index.json': 'c8cb50abb3731de0a361aa477a4b04988981bfff0935bff357658d55bfa454c1', 'tokenizer.json': '40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60'}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)


def validate_config(c):
    """Support only the reviewed target family; fixture dimensions may be smaller."""
    fixed = {
        "architectures": ["BailingMoeV3ForCausalLM"], "model_type": "bailing_hybrid",
        "hidden_act": "silu", "no_kda_lora": True, "use_kda_lora": False,
        "kda_safe_gate": True, "linear_silu": True, "use_qk_norm": True,
        "q_lora_rank": None, "rope_scaling": None, "rope_interleave": True,
        "gated_attention_proj_granularity_type": "head_wise", "router_dtype": "fp32",
        "topk_method": "noaux_tc", "score_function": "sigmoid", "scoring_func": "sigmoid",
        "norm_topk_prob": True, "moe_router_enable_expert_bias": True,
        "num_shared_experts": 1, "tie_word_embeddings": False, "group_norm_size": 1,
        "scale_router_input": False, "use_bias": False, "use_qkv_bias": False,
        "use_mla_nope": False, "use_nGPT": False, "up_proj_norm": False,
        "value_norm": False, "attention_dropout": 0.0, "embedding_dropout": 0.0,
        "output_dropout": 0.0,
    }
    for key, value in fixed.items():
        if key not in c or c[key] != value:
            raise ValueError(f"unsupported/missing config {key}: expected {value!r}")
    positive = ("hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
                "head_dim", "short_conv_kernel_size", "layer_group_size", "kv_lora_rank",
                "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "intermediate_size",
                "moe_intermediate_size", "moe_shared_expert_intermediate_size", "num_experts",
                "num_experts_per_tok", "n_group", "topk_group", "max_position_embeddings")
    for key in positive:
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(c.get("first_k_dense_replace")) is not int or not 0 <= c["first_k_dense_replace"] <= c["num_hidden_layers"]:
        raise ValueError("invalid dense layer schedule")
    if c.get("qk_head_dim") != c["qk_nope_head_dim"] + c["qk_rope_head_dim"]:
        raise ValueError("qk_head_dim mismatch")
    if c["qk_rope_head_dim"] % 2 or c.get("rotary_dim") != c["qk_rope_head_dim"]:
        raise ValueError("invalid rotary dimensions")
    if c.get("num_key_value_heads") != c["num_attention_heads"]:
        raise ValueError("MLA head replication variant unsupported")
    if c["num_experts"] % c["n_group"] or c["num_experts"] // c["n_group"] < 2:
        raise ValueError("groups require at least two experts")
    if c["topk_group"] > c["n_group"] or not 1 < c["num_experts_per_tok"] <= c["topk_group"] * (c["num_experts"] // c["n_group"]):
        raise ValueError("unsupported top-k/group geometry (top-k must exceed one)")
    for key in ("rope_theta", "rms_norm_eps", "routed_scaling_factor"):
        if not isinstance(c.get(key), (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"invalid {key}")
    if not isinstance(c.get("kda_lower_bound"), (int, float)) or not -5 <= c["kda_lower_bound"] < 0:
        raise ValueError("safe lower bound must be in [-5, 0)")
    for key in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"):
        limits = c.get(key)
        if not isinstance(limits, list) or len(limits) != c["num_hidden_layers"]:
            raise ValueError(f"{key} must match target layer count")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in limits):
            raise ValueError(f"invalid {key}")
    if type(c.get("num_nextn_predict_layers")) is not int or c["num_nextn_predict_layers"] not in (0, 1):
        raise ValueError("unreviewed extra checkpoint layers")


def expected_shapes(c):
    """Exact core tensor inventory; auxiliary precision never changes matrix layout."""
    hidden, heads, kd = c["hidden_size"], c["num_attention_heads"], c["head_dim"]
    result = {"model.word_embeddings.weight": (c["vocab_size"], hidden),
              "lm_head.weight": (c["vocab_size"], hidden), "model.norm.weight": (hidden,)}

    def mlp(prefix, intermediate):
        for name in ("gate_proj", "up_proj"):
            result[prefix + name + ".weight"] = (intermediate, hidden)
        result[prefix + "down_proj.weight"] = (hidden, intermediate)

    for layer in range(c["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            result[p + name + ".weight"] = (hidden,)
        a = p + "attention."
        if (layer + 1) % c["layer_group_size"]:
            for name in ("q", "k", "v", "f", "g"):
                result[a + name + "_proj.weight"] = (heads * kd, hidden)
            for name in ("q", "k", "v"):
                result[a + name + "_conv1d.weight"] = (heads * kd, 1, c["short_conv_kernel_size"])
            result.update({a + "b_proj.weight": (heads, hidden), a + "A_log": (heads,),
                           a + "dt_bias": (heads * kd,), a + "o_norm.weight": (kd,),
                           a + "o_proj.weight": (hidden, heads * kd)})
        else:
            result.update({
                a + "q_proj.weight": (heads * c["qk_head_dim"], hidden),
                a + "kv_a_proj_with_mqa.weight": (c["kv_lora_rank"] + c["qk_rope_head_dim"], hidden),
                a + "kv_a_layernorm.weight": (c["kv_lora_rank"],),
                a + "kv_b_proj.weight": (heads * (c["qk_nope_head_dim"] + c["v_head_dim"]), c["kv_lora_rank"]),
                a + "g_proj.weight": (heads, hidden),
                a + "dense.weight": (hidden, heads * c["v_head_dim"]),
            })
        if layer < c["first_k_dense_replace"]:
            mlp(p + "mlp.", c["intermediate_size"])
        else:
            result[p + "mlp.gate.weight"] = (c["num_experts"], hidden)
            result[p + "mlp.gate.expert_bias"] = (c["num_experts"],)
            for expert in range(c["num_experts"]):
                mlp(p + f"mlp.experts.{expert}.", c["moe_intermediate_size"])
            mlp(p + "mlp.shared_experts.", c["moe_shared_expert_intermediate_size"])
    return result


def fp32_weight(name):
    return (name.endswith((".A_log", ".dt_bias", ".expert_bias", ".o_norm.weight"))
            or "_conv1d.weight" in name or name.endswith("mlp.gate.weight"))


class TensorStore:
    """No shard cache: mmap one primitive read, copy selected data, close mmap.

    Live/peak counts measure owned compute weight bytes, NOT RSS/page cache or
    activation memory. Scopes clear tensors even if an operation raises.
    """
    def __init__(self, model_dir, config, *, fixture=False, max_weight_bytes=512 * 1024**2):
        self.root = Path(model_dir).resolve()
        self.index_path = self.root / "model.safetensors.index.json"
        self.index = read_json(self.index_path)
        self.weight_map = self.index["weight_map"]
        self.shapes = expected_shapes(config)
        missing = set(self.shapes) - self.weight_map.keys()
        extra = self.weight_map.keys() - self.shapes.keys()
        mtp_prefix = f"model.layers.{config['num_hidden_layers']}."
        ignored = {name for name in extra if config["num_nextn_predict_layers"] == 1 and name.startswith(mtp_prefix)}
        if missing or extra - ignored:
            raise ValueError(f"tensor inventory mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra - ignored)[:5]}")
        self.ignored_mtp_keys = sorted(ignored)
        self.fixture = fixture
        self.max_weight_bytes = max_weight_bytes
        if type(max_weight_bytes) is not int or max_weight_bytes <= 0:
            raise ValueError("max_weight_bytes must be positive")
        self.live_weight_bytes = self.peak_weight_bytes = self.live_tensors = self.peak_tensors = 0
        self.loads = 0
        self.observer = None  # Optional test observer; never retains tensors internally.
        self.inventory = self._audit_headers()

    def _path(self, filename):
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"unsafe shard name: {filename!r}")
        path = self.root / filename
        # HF cache symlinks may point into its blob store; only index path traversal
        # is rejected. Payload contents are still safetensors, never executable.
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint incomplete: {path}")
        return path

    def _audit_headers(self):
        shards, total_bytes, seen = [], 0, set()
        dtype_summary, auxiliary_storage = {}, {}
        by_file = {}
        for name, filename in self.weight_map.items():
            by_file.setdefault(filename, set()).add(name)
        for filename in sorted(by_file):
            path = self._path(filename)
            before = path.stat()
            with path.open("rb") as handle:
                raw_length = handle.read(8)
                if len(raw_length) != 8:
                    raise ValueError(f"truncated header: {path}")
                length = struct.unpack("<Q", raw_length)[0]
                if length > 64 * 1024**2:
                    raise ValueError("unreasonably large safetensors header")
                header = handle.read(length)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if keys != by_file[filename] or seen & keys:
                    raise ValueError(f"index/shard key closure mismatch: {filename}")
                seen.update(keys)
                for name in sorted(keys):
                    view = handle.get_slice(name)
                    shape, dtype = tuple(view.get_shape()), view.get_dtype()
                    if dtype not in ("BF16", "F32"):
                        raise ValueError(f"not original BF16/FP32 storage: {name}: {dtype}")
                    tensor_bytes = math.prod(shape) * (2 if dtype == "BF16" else 4)
                    total_bytes += tensor_bytes
                    dtype_summary.setdefault(dtype, {"tensors": 0, "bytes": 0})
                    dtype_summary[dtype]["tensors"] += 1
                    dtype_summary[dtype]["bytes"] += tensor_bytes
                    if name in self.shapes and fp32_weight(name):
                        auxiliary_storage[name] = dtype
                    if name in self.shapes:
                        if shape != self.shapes[name]:
                            raise ValueError(f"shape mismatch: {name}: {shape} != {self.shapes[name]}")
                        if not self.fixture and not fp32_weight(name) and dtype != "BF16":
                            raise ValueError(f"original matrix/norm must be BF16: {name}: {dtype}")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"shard changed during header audit: {path}")
            shards.append({"file": filename, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                           "header_sha256": hashlib.sha256(raw_length + header).hexdigest()})
        if total_bytes != self.index.get("metadata", {}).get("total_size"):
            raise ValueError("index metadata.total_size does not match tensor headers")
        return {"shards": shards, "tensor_bytes": total_bytes, "tensor_count": len(seen),
                "storage_dtype_summary": dtype_summary, "fp32_compute_auxiliary_storage": auxiliary_storage,
                "payload_hashes_verified": False, "closure": "all index/shard keys and dtypes; core shapes; MTP not evaluated"}

    def _read(self, name, device, dtype, rows=None):
        shape = self.shapes[name]
        if rows is not None:
            start, end = rows
            if len(shape) != 2 or not 0 <= start < end <= shape[0]:
                raise ValueError("invalid tensor row slice")
            shape = (end - start, shape[1])
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if self.live_weight_bytes + size > self.max_weight_bytes:
            raise MemoryError(f"weight scope exceeds {self.max_weight_bytes} bytes at {name}")
        path = self._path(self.weight_map[name])
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            source = handle.get_tensor(name) if rows is None else handle.get_slice(name)[rows[0]:rows[1]]
            tensor = source.to(device=device, dtype=dtype, copy=True)
            del source
        self.loads += 1
        self.live_weight_bytes += size
        self.live_tensors += 1
        self.peak_weight_bytes = max(self.peak_weight_bytes, self.live_weight_bytes)
        self.peak_tensors = max(self.peak_tensors, self.live_tensors)
        if self.observer is not None:
            try:
                self.observer(name, tensor)
            except Exception:
                self.live_weight_bytes -= size
                self.live_tensors -= 1
                raise
        return tensor, size

    @contextmanager
    def scope(self, names, device, dtype):
        weights, sizes = {}, []
        try:
            for name in names:
                if name in weights:
                    raise ValueError(f"duplicate scope weight: {name}")
                tensor, size = self._read(name, device, torch.float32 if fp32_weight(name) else dtype)
                weights[name] = tensor
                sizes.append(size)
                del tensor
            yield weights
        finally:
            weights.clear()
            self.live_weight_bytes -= sum(sizes)
            self.live_tensors -= len(sizes)

    @contextmanager
    def rows(self, name, start, end, device, dtype):
        tensor, size = self._read(name, device, dtype, (start, end))
        try:
            yield tensor
        finally:
            del tensor
            self.live_weight_bytes -= size
            self.live_tensors -= 1

    def assert_unchanged(self):
        for item in self.inventory["shards"]:
            stat = self._path(item["file"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                raise RuntimeError(f"checkpoint changed: {item['file']}")


@dataclass
class KDAState:
    matrix: torch.Tensor
    conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class MLAState:
    keys: torch.Tensor  # [B,T,H,QK]; expanded reference, not compressed production cache
    values: torch.Tensor


def move_state(state, device):
    if isinstance(state, KDAState):
        return KDAState(state.matrix.to(device, copy=True), tuple(v.to(device, copy=True) for v in state.conv))
    if isinstance(state, MLAState):
        return MLAState(state.keys.to(device, copy=True), state.values.to(device, copy=True))
    if state is not None:
        raise TypeError("unknown reference state")
    return None


@dataclass
class ReferenceState:
    tokens: int
    batch: int
    identity: str
    layers: dict[int, KDAState | MLAState] = field(default_factory=dict)

    def clone(self):
        return ReferenceState(self.tokens, self.batch, self.identity,
                              {i: move_state(v, "cpu") for i, v in self.layers.items()})


@dataclass
class ForwardResult:
    hidden: torch.Tensor  # CPU, post-model.norm, compute dtype
    state: ReferenceState | None
    position_start: int
    identity: str


def rms_norm(x, weight, eps):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight.to(x.dtype)


def grouped_route(x, weight, correction, groups, keep_groups, top_k, scale):
    """Independent grouped routing. Stable ties: lower group/expert ID first."""
    scores = torch.sigmoid(F.linear(x.float(), weight.float()))
    biased = scores + correction.float()
    grouped = biased.reshape(-1, groups, biased.shape[-1] // groups)
    group_scores = grouped.sort(dim=-1, descending=True, stable=True).values[..., :2].sum(-1)
    selected_groups = group_scores.argsort(dim=-1, descending=True, stable=True)[:, :keep_groups]
    allowed = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, selected_groups, True)
    eligible = biased.masked_fill(~allowed.repeat_interleave(grouped.shape[-1], dim=1), -torch.inf)
    ids = eligible.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    selected = scores.gather(1, ids)
    weights = selected / (selected.sum(-1, keepdim=True) + 1e-20) * scale
    return ids, weights


def kda_attention(x, w, c, state=None):
    batch, tokens, _ = x.shape
    heads, dim = c["num_attention_heads"], c["head_dim"]
    if state is not None and (not isinstance(state, KDAState) or state.matrix.dtype != torch.float32):
        raise ValueError("KDA requires a KDAState with FP32 matrix")
    if state is not None and (len(state.conv) != 3 or any(v.dtype != x.dtype for v in state.conv)):
        raise ValueError("KDA convolution history must use the compute dtype")
    qkv, histories = [], []
    for index, name in enumerate(("q", "k", "v")):
        projected = F.linear(x, w[name + "_proj.weight"])
        convolved, history = causal_conv(projected, w[name + "_conv1d.weight"].squeeze(1),
                                         None if state is None else state.conv[index])
        qkv.append(convolved.to(x.dtype).reshape(batch, tokens, heads, dim))
        histories.append(history.clone())  # Never retain full projection via a narrow view.
    f = F.linear(x, w["f_proj.weight"]).reshape(batch, tokens, heads, dim)
    decay = safe_kda_decay(f, w["A_log"], w["dt_bias"], c["kda_lower_bound"])
    beta = F.linear(x, w["b_proj.weight"]).float().sigmoid()
    values, matrix = kda_recurrence(*qkv, decay, beta, None if state is None else state.matrix)
    gate = F.linear(x, w["g_proj.weight"]).reshape(batch, tokens, heads, dim)
    # FLA emits model-dtype recurrence values before fused FP32 gated RMSNorm.
    output = gated_head_norm(values.to(x.dtype), w["o_norm.weight"], gate, c["rms_norm_eps"]).to(x.dtype)
    return F.linear(output.flatten(-2), w["o_proj.weight"]), KDAState(matrix, tuple(histories))


def mla_attention(x, w, c, position_start=0, state=None):
    """Expanded causal MLA, bounded by the caller's query chunk. No engine kernels."""
    batch, tokens, _ = x.shape
    heads, nope, rope, vd = (c[k] for k in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
    if state is not None and not isinstance(state, MLAState):
        raise ValueError("MLA requires MLAState")
    prior = 0 if state is None else state.keys.shape[1]
    if state is not None and (state.keys.shape != (batch, prior, heads, nope + rope)
                              or state.values.shape != (batch, prior, heads, vd)
                              or state.keys.dtype != x.dtype or state.values.dtype != x.dtype):
        raise ValueError("MLA cache shape/dtype mismatch")
    if prior != position_start:
        raise ValueError("MLA cache/position boundary mismatch")
    q = F.linear(x, w["q_proj.weight"]).reshape(batch, tokens, heads, nope + rope)
    latent_rope = F.linear(x, w["kv_a_proj_with_mqa.weight"])
    latent, kr = latent_rope.split([c["kv_lora_rank"], rope], -1)
    latent = rms_norm(latent, w["kv_a_layernorm.weight"], c["rms_norm_eps"])
    expanded = F.linear(latent, w["kv_b_proj.weight"]).reshape(batch, tokens, heads, nope + vd)
    kn, value = expanded.split([nope, vd], -1)
    positions = torch.arange(position_start, position_start + tokens, device=x.device)[None, :]
    qr = interleaved_rope(q[..., nope:], positions, c["rope_theta"]).to(x.dtype)
    kr = interleaved_rope(kr.unsqueeze(-2), positions, c["rope_theta"]).to(x.dtype)
    query = torch.cat([q[..., :nope], qr], -1)
    key = torch.cat([kn, kr.expand(batch, tokens, heads, rope)], -1)
    if state is not None:
        key = torch.cat([state.keys, key], 1)
        value = torch.cat([state.values, value], 1)
    else:
        value = value.contiguous()
    scores = (query.transpose(1, 2) @ key.transpose(1, 2).transpose(-1, -2)) * ((nope + rope) ** -0.5)
    allowed = torch.arange(key.shape[1], device=x.device)[None, :] <= positions[0, :, None]
    probabilities = scores.float().masked_fill(~allowed, -torch.inf).softmax(-1).to(x.dtype)
    attended = (probabilities @ value.transpose(1, 2)).transpose(1, 2)
    gate = F.linear(x, w["g_proj.weight"]).float().sigmoid().to(x.dtype)
    attended = attended * gate[..., None]
    output = F.linear(attended.flatten(-2), w["dense.weight"])
    return output, MLAState(key, value)


class StreamingReference:
    def __init__(self, model_dir, *, compute_dtype="bfloat16", device="cpu", fixture=False,
                 source_revision=OFFICIAL_REVISION, chunk_size=64, max_context=8192,
                 max_weight_bytes=512 * 1024**2, negative_control=None):
        if compute_dtype not in DTYPES:
            raise ValueError("compute dtype must be explicitly BF16 or FP32")
        if not fixture and (compute_dtype != "bfloat16" or source_revision != OFFICIAL_REVISION):
            raise ValueError("original qualification requires BF16 and the reviewed source pin")
        if negative_control not in (None, "pre_silu", "unclamped"):
            raise ValueError("only named negative-control clamp variants are allowed")
        if type(chunk_size) is not int or chunk_size <= 0 or type(max_context) is not int or max_context <= 0:
            raise ValueError("chunk/context limits must be positive integers")
        self.root = Path(model_dir).resolve()
        self.config = read_json(self.root / "config.json")
        validate_config(self.config)
        if fixture and self.config.get("reference_fixture") != FIXTURE_LABEL:
            raise ValueError("fixture mode requires an explicit synthetic checkpoint label")
        self.dtype = DTYPES[compute_dtype]
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU or explicitly scheduled CUDA is supported")
        self._check_precision()
        self.numerical_backend_policy = observe_precision()
        self.chunk_size, self.max_context = chunk_size, min(max_context, self.config["max_position_embeddings"])
        self.variant = negative_control or "post_silu"
        self.fixture = fixture
        if not fixture:
            for filename in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
                if sha256_file(self.root / filename) != REVIEWED_METADATA_SHA256[filename]:
                    raise ValueError(f"original differs from reviewed pinned metadata: {filename}")
        self.store = TensorStore(self.root, self.config, fixture=fixture, max_weight_bytes=max_weight_bytes)
        self.provenance = {
            "kind": FIXTURE_LABEL if fixture else "reviewed-original target reference; D01 qualified, not official HF forward",
            "source_repo": None if fixture else OFFICIAL_REPO,
            "source_revision": source_revision if not fixture else None,
            "revision_evidence": "reviewed config/index/tokenizer hash match; shard headers audited; full payload hashes NOT verified here",
            "config_sha256": sha256_file(self.root / "config.json"),
            "index_sha256": sha256_file(self.store.index_path),
            "tokenizer_sha256": sha256_file(self.root / "tokenizer.json") if (self.root / "tokenizer.json").is_file() else None,
            "streaming_sha256": sha256_file(Path(__file__)),
            "math_oracles_sha256": sha256_file(Path(__file__).with_name("math_oracles.py")),
            "design_sha256": "fefc0e43392eb9af39b2122288c916388a9303872c2936a7bd567a34ed55c8c7",  # Historical document digest; document not bundled.
            "architecture_research_sha256": "baa9d8aa27a4c9027cc9aca03abddc291933ea6064e948a94b2742ab9ff5a61f",  # Historical document digest; document not bundled.
            "compute_dtype": compute_dtype, "fp32_ops": ["norm statistics", "router GEMM/sigmoid/selection/mixture accumulation", "conv accumulation/SiLU", "KDA decay/L2/matrix/gated norm", "RoPE phases/rotation", "MLA softmax"],
            "projection_residual_dtype": compute_dtype, "kda_state_dtype": "float32",
            "mla_cache_dtype": compute_dtype, "mla_cache_layout": "expanded key/value, not absorbed",
            "mla_score_matmul_dtype": compute_dtype, "head_gemm_dtype": compute_dtype,
            "clamp": self.variant, "negative_control": negative_control is not None,
            "routing_ties": "lower group/expert ID first", "expert_accumulation": "ascending expert ID, FP32, cast before shared sum",
            "chunk_size": chunk_size, "max_context": self.max_context,
            "mtp": "NOT IMPLEMENTED; target only", "ignored_mtp_tensor_count": len(self.store.ignored_mtp_keys),
            "torch_version": torch.__version__, "device": str(self.device),
            "safetensors_version": importlib.metadata.version("safetensors"),
            "torch_cpu_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "autocast": False,
            "numerical_backend_policy": self.numerical_backend_policy,
            "precision_policy_sha256": sha256_file(Path(__file__).with_name("precision_policy.py")),
            "inventory": self.store.inventory,
        }
        identity_fields = {k: self.provenance[k] for k in ("config_sha256", "index_sha256", "compute_dtype", "clamp", "streaming_sha256", "math_oracles_sha256")}
        identity_fields.update(source_path=str(self.root), shard_identity=self.store.inventory["shards"],
                               numerical_backend_policy=self.numerical_backend_policy,
                               precision_policy_sha256=self.provenance["precision_policy_sha256"])
        self.identity = hashlib.sha256(json.dumps(identity_fields, sort_keys=True).encode()).hexdigest()

    def _check_precision(self):
        # Do not silently change caller-global backend settings or autocast.
        frozen = getattr(self, "numerical_backend_policy", None)
        if frozen is not None and observe_precision() != frozen:
            raise ValueError("reference numerical backend policy changed since construction")
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError("reference execution must be outside autocast")
        if self.device.type == "cuda" and torch.get_float32_matmul_precision() != "highest":
            raise ValueError("reference FP32 router requires highest matmul precision (TF32 disabled)")

    def _finite(self, x, stage):
        if not torch.isfinite(x).all().item():
            raise FloatingPointError(f"nonfinite reference values: {stage}")

    def _embedding(self, ids):
        hidden = torch.empty(*ids.shape, self.config["hidden_size"], dtype=self.dtype)
        # Primitive row slicing avoids copying the original 157184-row table.
        for token in sorted(set(ids.flatten().tolist())):
            with self.store.rows("model.word_embeddings.weight", token, token + 1, "cpu", self.dtype) as row:
                hidden[ids == token] = row[0]
            del row
        return hidden

    def _norm_cpu(self, x, name):
        result = torch.empty_like(x)
        with self.store.scope([name], self.device, self.dtype) as weights:
            for start in range(0, x.shape[1], self.chunk_size):
                part = x[:, start:start + self.chunk_size].to(self.device)
                result[:, start:start + self.chunk_size] = rms_norm(part, weights[name], self.config["rms_norm_eps"]).cpu()
        return result

    def _mlp(self, x, prefix, limit):
        names = [prefix + name + ".weight" for name in ("gate_proj", "up_proj", "down_proj")]
        result = torch.empty_like(x)
        with self.store.scope(names, self.device, self.dtype) as weights:
            for start in range(0, len(x), self.chunk_size):
                part = x[start:start + self.chunk_size].to(self.device)
                activated = swiglu(F.linear(part, weights[names[0]]), F.linear(part, weights[names[1]]), limit, self.variant)
                result[start:start + self.chunk_size] = F.linear(activated, weights[names[2]]).cpu()
        return result

    def _feedforward(self, x, layer):
        c, shape = self.config, x.shape
        x = x.reshape(-1, shape[-1])
        prefix = f"model.layers.{layer}.mlp."
        routed_limit = c["expert_swiglu_limit_list"][layer]
        if layer < c["first_k_dense_replace"]:
            return self._mlp(x, prefix, routed_limit).reshape(shape)
        router_names = [prefix + "gate.weight", prefix + "gate.expert_bias"]
        ids = torch.empty(len(x), c["num_experts_per_tok"], dtype=torch.long)
        weights = torch.empty_like(ids, dtype=torch.float32)
        with self.store.scope(router_names, self.device, self.dtype) as router:
            for start in range(0, len(x), self.chunk_size):
                selected, mixing = grouped_route(x[start:start + self.chunk_size].to(self.device),
                    router[router_names[0]], router[router_names[1]], c["n_group"], c["topk_group"],
                    c["num_experts_per_tok"], c["routed_scaling_factor"])
                ids[start:start + self.chunk_size], weights[start:start + self.chunk_size] = selected.cpu(), mixing.cpu()
        output = torch.zeros_like(x, dtype=torch.float32)
        for expert in ids.unique(sorted=True).tolist():
            rows, slots = torch.where(ids == expert)
            # Only one expert's weights and assigned-token outputs at a time.
            expert_output = self._mlp(x[rows], prefix + f"experts.{expert}.", routed_limit)
            output[rows] += expert_output.float() * weights[rows, slots, None]
            del expert_output
        output = output.to(self.dtype)
        shared = self._mlp(x, prefix + "shared_experts.", c["share_expert_swiglu_limit_list"][layer])
        return (output + shared).reshape(shape)

    @torch.inference_mode()
    def forward(self, input_ids, *, state=None, keep_state=False,
                capture_layers=(), capture: Callable[[str, torch.Tensor, int], None] | None = None):
        """Equal-length, unpadded batch; caller owns sequence identity and boundaries.

        Returned state is a fresh CPU copy, never mutates supplied state. Captures
        are synchronous, CPU compute dtype, post-residual at numbered layers and
        post-model.norm at 'postnorm'. Callback must not accumulate unbounded data
        or change process-global numerical policy. Caller owns that policy
        exclusively; no automatic setters or locks can protect unrelated threads.
        """
        self._check_precision()
        ids = input_ids.detach().cpu()
        if ids.dtype != torch.int64 or ids.ndim != 2 or not all(ids.shape):
            raise ValueError("input_ids must be nonempty int64 [batch,tokens], no padding")
        if ids.min().item() < 0 or ids.max().item() >= self.config["vocab_size"]:
            raise ValueError("token outside checkpoint vocabulary")
        if state is not None and (state.batch != ids.shape[0] or state.identity != self.identity
                                  or set(state.layers) != set(range(self.config["num_hidden_layers"])) or state.tokens <= 0):
            raise ValueError("state identity/batch/layer contract mismatch")
        position = 0 if state is None else state.tokens
        if position + ids.shape[1] > self.max_context:
            raise ValueError("context exceeds explicit reference memory/context bound")
        capture_layers = set(capture_layers)
        if not capture_layers <= set(range(self.config["num_hidden_layers"])):
            raise ValueError("capture layer outside target stack")
        self.store.assert_unchanged()
        hidden = self._embedding(ids)
        states = {}
        for layer in range(self.config["num_hidden_layers"]):
            self._check_precision()
            p = f"model.layers.{layer}."
            attention_prefix = p + "attention."
            names = [name for name in self.store.shapes if name.startswith(attention_prefix)]
            residual = torch.empty_like(hidden)
            layer_state = move_state(None if state is None else state.layers[layer], self.device)
            with self.store.scope(names + [p + "input_layernorm.weight"], self.device, self.dtype) as weights:
                attn_weights = {name[len(attention_prefix):]: weights[name] for name in names}
                try:
                    for start in range(0, ids.shape[1], self.chunk_size):
                        part = hidden[:, start:start + self.chunk_size].to(self.device)
                        x = rms_norm(part, weights[p + "input_layernorm.weight"], self.config["rms_norm_eps"])
                        if (layer + 1) % self.config["layer_group_size"]:
                            out, layer_state = kda_attention(x, attn_weights, self.config, layer_state)
                        else:
                            out, layer_state = mla_attention(x, attn_weights, self.config, position + start, layer_state)
                        residual[:, start:start + self.chunk_size] = (part + out).cpu()
                finally:
                    attn_weights.clear()
            if keep_state:
                states[layer] = move_state(layer_state, "cpu")
            del layer_state
            self._finite(residual, f"layer {layer} attention")
            normalized = self._norm_cpu(residual, p + "post_attention_layernorm.weight")
            hidden = residual + self._feedforward(normalized, layer)
            del normalized, residual
            self._finite(hidden, f"layer {layer} residual")
            if capture is not None and layer in capture_layers:
                capture(f"layer.{layer}", hidden.clone(), position)
                self._check_precision()
        self._check_precision()
        hidden = self._norm_cpu(hidden, "model.norm.weight")
        self._finite(hidden, "postnorm")
        if capture is not None:
            capture("postnorm", hidden.clone(), position)
            self._check_precision()
        self.store.assert_unchanged()
        self._check_precision()
        return ForwardResult(hidden, ReferenceState(position + ids.shape[1], ids.shape[0], self.identity, states) if keep_state else None, position, self.identity)

    @torch.inference_mode()
    def iter_logits(self, result, *, storage_dtype="bfloat16", rows_per_chunk=16, vocab_rows=1024):
        """Yield CPU [batch,rows,vocab] native-head logits, no full logit allocation.

        Vocab slicing bounds head weight memory; this changes GEMM geometry, not
        the mathematical head. Stored logits can be BF16 or FP32 (explicit).
        Caller must keep exclusive numerical-policy ownership across yields.
        """
        if storage_dtype not in DTYPES or rows_per_chunk <= 0 or vocab_rows <= 0:
            raise ValueError("invalid logit dtype/chunk")
        self._check_precision()
        if result.identity != self.identity or result.hidden.dtype != self.dtype:
            raise ValueError("hidden-state identity/compute dtype mismatch")
        hidden = result.hidden
        self.store.assert_unchanged()
        for start in range(0, hidden.shape[1], rows_per_chunk):
            self._check_precision()
            part = hidden[:, start:start + rows_per_chunk].to(self.device)
            logits = torch.empty(*part.shape[:-1], self.config["vocab_size"], dtype=DTYPES[storage_dtype])
            for vstart in range(0, self.config["vocab_size"], vocab_rows):
                end = min(vstart + vocab_rows, self.config["vocab_size"])
                with self.store.rows("lm_head.weight", vstart, end, self.device, self.dtype) as weight:
                    logits[..., vstart:end] = F.linear(part, weight).to(device="cpu", dtype=DTYPES[storage_dtype])
                del weight
            self._finite(logits, "logits")
            self._check_precision()
            yield result.position_start + start, logits
            # Includes the resume after the final chunk, before StopIteration.
            self._check_precision()
        self.store.assert_unchanged()
        self._check_precision()


def run_cli(args):
    if str(args.device) != "cpu" and not args.allow_gpu:
        raise ValueError("non-CPU capture requires --allow-gpu and separate scheduling authorization")
    # No tokenizer/model class loading. Frozen IDs define teacher forcing exactly.
    token_path = Path(args.tokens)
    corpus = read_json(token_path)
    if not isinstance(corpus, dict) or corpus.get("purpose") != "held-out" or not isinstance(corpus.get("sequences"), list) or not corpus["sequences"]:
        raise ValueError("token JSON requires purpose='held-out' and nonempty sequences")
    ids_seen = set()
    for seq in corpus["sequences"]:
        sid, ids = seq.get("sequence_id"), seq.get("input_ids")
        if not isinstance(sid, str) or not sid or sid in ids_seen:
            raise ValueError("sequence IDs must be unique nonempty strings")
        ids_seen.add(sid)
        if not isinstance(ids, list) or len(ids) < 2 or any(type(i) is not int for i in ids):
            raise ValueError("each sequence requires at least two literal integer token IDs")
        if type(seq.get("score_start")) is not int or not 0 <= seq["score_start"] < len(ids) - 1:
            raise ValueError("score_start must identify a row with a next-token target")
    model = StreamingReference(args.model, compute_dtype=args.compute_dtype, device=args.device,
        fixture=args.fixture, source_revision=args.source_revision, chunk_size=args.chunk_size,
        max_context=args.max_context, max_weight_bytes=args.max_weight_mib * 1024**2,
        negative_control=args.negative_control)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier result.
    manifest = {"status": "running", "reference": model.provenance,
                "capture_format_version": 2, "schema": "ling3-portable-reference-v1",
                "logit_domain": {"config_sha256": model.provenance["config_sha256"],
                                 "vocab_size": model.config["vocab_size"],
                                 "column_ids": "arange(0, vocab_size)", "vocabulary_masking": "none",
                                 "softmax_domain": "all_config_vocab_ids"},
                "capture_postnorm": bool(args.capture_layer or args.capture_postnorm),
                "token_file_sha256": sha256_file(token_path), "corpus_provenance": corpus.get("provenance"),
                "score_row_semantics": "logits at input_ids[t] predict input_ids[t+1]; last row unscored; no BOS/EOS added",
                "logit_storage_dtype": args.logit_storage_dtype, "logit_rows": args.logit_rows,
                "head_vocab_rows": args.head_vocab_rows, "capture_layers": args.capture_layer,
                "full_vocab_rows": True, "unused_vocabulary_rows_masked": False,
                "sequences": [], "files": []}
    manifest_path = output / "manifest.json"

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def persist(name, tensors, metadata):
        path = output / name
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata=metadata)
        manifest["files"].append({"file": name, "sha256": sha256_file(path), **metadata})

    write_manifest()
    try:
        for seq_index, seq in enumerate(corpus["sequences"]):
            ids = torch.tensor([seq["input_ids"]], dtype=torch.long)
            def capture(name, hidden, position):
                persist(f"sequence-{seq_index:04d}-{name}.safetensors", {"hidden": hidden, "input_ids": ids,
                    "positions": torch.arange(position, position + ids.shape[1], dtype=torch.long)},
                    {"sequence_id": seq["sequence_id"], "stage": name})
            result = model.forward(ids, capture_layers=args.capture_layer,
                                   capture=capture if args.capture_layer or args.capture_postnorm else None)
            row_count = 0
            for position, logits in model.iter_logits(result, storage_dtype=args.logit_storage_dtype,
                                                      rows_per_chunk=args.logit_rows, vocab_rows=args.head_vocab_rows):
                length = logits.shape[1]
                positions = torch.arange(position, position + length, dtype=torch.long)
                targets = torch.full((length,), -1, dtype=torch.long)
                valid = positions < ids.shape[1] - 1
                targets[valid] = ids[0, positions[valid] + 1]
                persist(f"sequence-{seq_index:04d}-logits-{position:08d}.safetensors",
                    {"logits": logits, "column_ids": torch.arange(model.config["vocab_size"], dtype=torch.long),
                     "positions": positions, "input_ids": ids[:, position:position + length],
                     "target_ids": targets, "score_mask": valid & (positions >= seq["score_start"])},
                    {"sequence_id": seq["sequence_id"], "stage": "logits", "storage_dtype": args.logit_storage_dtype})
                row_count += length
            if row_count != len(seq["input_ids"]):
                raise RuntimeError("logit row count mismatch")
            manifest["sequences"].append({**seq, "logit_rows_written": row_count})
            write_manifest()
        model.store.assert_unchanged()
        model._check_precision()
        manifest.update(status="complete", memory_accounting={"peak_owned_weight_bytes": model.store.peak_weight_bytes,
            "peak_owned_weight_tensors": model.store.peak_tensors, "live_owned_weight_bytes": model.store.live_weight_bytes,
            "tensor_loads": model.store.loads, "rss_measured": False})
        write_manifest()
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_manifest()
        raise
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path),
                      "sequences": len(manifest["sequences"]), "files": len(manifest["files"]),
                      "peak_owned_weight_bytes": model.store.peak_weight_bytes}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gpu", action="store_true", help="explicit opt-in; does not grant scheduling authorization")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True, help="held-out JSON with exact token IDs (see README)")
    parser.add_argument("--output", required=True, help="new directory; existing paths refused")
    parser.add_argument("--source-revision", default=OFFICIAL_REVISION)
    parser.add_argument("--compute-dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--logit-storage-dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--device", default="cpu", help="CPU default; cuda:0 only after parent GPU scheduling")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--max-weight-mib", type=int, default=512)
    parser.add_argument("--logit-rows", type=int, default=16)
    parser.add_argument("--head-vocab-rows", type=int, default=1024)
    parser.add_argument("--capture-layer", type=int, action="append", default=[])
    parser.add_argument("--capture-postnorm", action="store_true")
    parser.add_argument("--fixture", action="store_true", help="requires labelled synthetic config; NOT original qualification")
    parser.add_argument("--negative-control", choices=("pre_silu", "unclamped"), help="labels all output as a wrong-variant negative control")
    run_cli(parser.parse_args())


if __name__ == "__main__":
    main()
