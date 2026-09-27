"""CPU-only deterministic SYNTHETIC checkpoints, not original model qualification.

The full-model oracle below does not call streaming/math_oracles functions.
It uses scalar/token loops, explicit transition matrices, complex RoPE and
Python-sorted routing to test different arithmetic/selection implementations.
"""
import copy
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
from ling3_repro import streaming as ref


def fixture_config():
    c = {'architectures': ['BailingMoeV3ForCausalLM'], 'attention_dropout': 0.0, 'auto_map': {'AutoConfig': 'configuration_bailing_moe_v3.BailingMoeV3Config', 'AutoModel': 'modeling_bailing_moe_v3.BailingMoeV3Model', 'AutoModelForCausalLM': 'modeling_bailing_moe_v3.BailingMoeV3ForCausalLM'}, 'embedding_dropout': 0.0, 'eos_token_id': 156895, 'expert_swiglu_limit_list': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4], 'first_k_dense_replace': 2, 'gated_attention_proj_granularity_type': 'head_wise', 'group_norm_size': 1, 'head_dim': 128, 'hidden_act': 'silu', 'hidden_size': 2560, 'initializer_range': 0.02, 'intermediate_size': 6144, 'kda_lower_bound': -5.0, 'kda_safe_gate': True, 'kv_lora_rank': 512, 'layer_group_size': 6, 'linear_silu': True, 'max_position_embeddings': 262144, 'max_window_layers': 20, 'moe_intermediate_size': 768, 'moe_router_enable_expert_bias': True, 'moe_shared_expert_intermediate_size': 768, 'mtp_loss_scaling_factor': 0, 'mtp_use_kda': False, 'n_group': 8, 'no_kda_lora': True, 'norm_topk_prob': True, 'num_attention_heads': 32, 'num_experts': 512, 'num_experts_per_tok': 8, 'num_hidden_layers': 42, 'num_key_value_heads': 32, 'num_kv_heads_for_linear_attn': 0, 'num_nextn_predict_layers': 1, 'num_shared_experts': 1, 'output_dropout': 0.0, 'output_router_logits': False, 'pad_token_id': 156892, 'partial_rotary_factor': 0.5, 'q_lora_rank': None, 'qk_head_dim': 192, 'qk_nope_head_dim': 128, 'qk_rope_head_dim': 64, 'rms_norm_eps': 1e-06, 'rope_interleave': True, 'rope_scaling': None, 'rope_theta': 6000000, 'rotary_dim': 64, 'routed_scaling_factor': 2.5, 'router_dtype': 'fp32', 'scale_router_input': False, 'score_function': 'sigmoid', 'scoring_func': 'sigmoid', 'seq_aux': True, 'share_expert_swiglu_limit_list': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 7, 7], 'short_conv_kernel_size': 4, 'tie_word_embeddings': False, 'topk_group': 4, 'topk_method': 'noaux_tc', 'transformers_version': '4.45.0', 'up_proj_norm': False, 'use_bias': False, 'use_cache': True, 'use_kda_lora': False, 'use_mla_nope': False, 'use_nGPT': False, 'use_qk_norm': True, 'use_qkv_bias': False, 'v_head_dim': 128, 'value_norm': False, 'vocab_size': 157184, 'model_type': 'bailing_hybrid', 'torch_dtype': 'bfloat16'}
    c.update(reference_fixture=ref.FIXTURE_LABEL, hidden_size=8, vocab_size=19,
             num_hidden_layers=4, layer_group_size=2, num_attention_heads=2,
             num_key_value_heads=2, head_dim=3, kv_lora_rank=5, qk_nope_head_dim=3,
             qk_rope_head_dim=4, qk_head_dim=7, rotary_dim=4, v_head_dim=2,
             intermediate_size=9, moe_intermediate_size=5, moe_shared_expert_intermediate_size=7,
             first_k_dense_replace=1, num_experts=6, n_group=3, topk_group=2,
             num_experts_per_tok=3, num_nextn_predict_layers=0, max_position_embeddings=2048,
             expert_swiglu_limit_list=[0, .35, .2, .3],
             share_expert_swiglu_limit_list=[0, .5, 0, .4])
    return c


def make_fixture(root, dtype=torch.float32):
    root.mkdir(parents=True, exist_ok=True)
    c = fixture_config()
    generator = torch.Generator().manual_seed(481519)
    weights = {}
    for name, shape in ref.expected_shapes(c).items():
        value = torch.randn(shape, generator=generator) * .32
        if "norm.weight" in name:
            value = 1 + value * .3
        if name.endswith(".A_log"):
            value = torch.tensor([-.7, .4])
        if name.endswith(".expert_bias"):
            value = torch.tensor([-1.2, -1.6, -1.3, -1.7, -1.1, -1.8])
        weights[name] = value.to(torch.float32 if ref.fp32_weight(name) else dtype)
    # Several shards exercise cross-shard scopes, while every tensor is tiny.
    names = sorted(weights)
    weight_map = {}
    for shard in range(3):
        filename = f"model-{shard:02d}.safetensors"
        tensors = {name: weights[name] for name in names[shard::3]}
        save_file(tensors, str(root / filename), metadata={"fixture": ref.FIXTURE_LABEL})
        weight_map.update({name: filename for name in tensors})
    (root / "config.json").write_text(json.dumps(c, indent=2))
    total = sum(v.numel() * v.element_size() for v in weights.values())
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
    return c, weights


def oracle_norm(x, weight, eps):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * weight


def oracle_route(x, weight, correction, c):
    output_ids, output_weights = [], []
    experts_per_group = c["num_experts"] // c["n_group"]
    for token in x:
        scores = [torch.sigmoid(torch.dot(token, expert)) for expert in weight]
        biased = [float(score + bias) for score, bias in zip(scores, correction)]
        group_values = []
        for group in range(c["n_group"]):
            members = list(range(group * experts_per_group, (group + 1) * experts_per_group))
            group_values.append(sum(sorted((biased[i] for i in members), reverse=True)[:2]))
        keep = sorted(range(c["n_group"]), key=lambda group: (-group_values[group], group))[:c["topk_group"]]
        eligible = [i for i in range(c["num_experts"]) if i // experts_per_group in keep]
        ids = sorted(eligible, key=lambda i: (-biased[i], i))[:c["num_experts_per_tok"]]
        selected = torch.stack([scores[i] for i in ids])
        output_ids.append(ids)
        output_weights.append(selected / (selected.sum() + 1e-20) * c["routed_scaling_factor"])
    return torch.tensor(output_ids), torch.stack(output_weights)


def oracle_rotate(x, position, theta):
    pairs = torch.complex(x[..., ::2], x[..., 1::2])
    angles = torch.tensor([position * theta ** (-2 * i / x.shape[-1]) for i in range(x.shape[-1] // 2)], dtype=x.dtype)
    product = pairs * torch.complex(angles.cos(), angles.sin())
    return torch.stack([product.real, product.imag], -1).flatten(-2)


def oracle_kda(x, w, c, state=None):
    batch, tokens, _ = x.shape
    heads, dim, width = c["num_attention_heads"], c["head_dim"], c["short_conv_kernel_size"]
    s = torch.zeros(batch, heads, dim, dim, dtype=x.dtype) if state is None else state.matrix.to(x.dtype).clone()
    history = [torch.zeros(batch, heads * dim, width - 1, dtype=x.dtype) for _ in range(3)] if state is None else [h.to(x.dtype).clone() for h in state.conv]
    output = []
    for t in range(tokens):
        qkv = []
        for index, label in enumerate(("q", "k", "v")):
            raw = x[:, t] @ w[label + "_proj.weight"].T
            window = torch.cat([history[index], raw[..., None]], -1)
            # Explicit convolution coefficient loop, no conv/unfold helper.
            convolved = sum(window[..., k] * w[label + "_conv1d.weight"][:, 0, k] for k in range(width))
            qkv.append((convolved * torch.sigmoid(convolved)).reshape(batch, heads, dim))
            history[index] = window[..., 1:].clone()
        q, k, v = qkv
        q = q / torch.sqrt((q*q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k*k).sum(-1, keepdim=True) + 1e-6)
        decay = (x[:, t] @ w["f_proj.weight"].T).reshape(batch, heads, dim) + w["dt_bias"].reshape(heads, dim)
        decay = torch.exp(c["kda_lower_bound"] * torch.sigmoid(w["A_log"].exp()[None, :, None] * decay))
        beta = torch.sigmoid(x[:, t] @ w["b_proj.weight"].T)
        # Matrix-form recurrence differs from evaluator's prediction/error update.
        eye = torch.eye(dim, dtype=x.dtype)
        transform = (eye - beta[..., None, None] * k[..., :, None] * k[..., None, :]) @ torch.diag_embed(decay)
        s = transform @ s + beta[..., None, None] * k[..., :, None] * v[..., None, :]
        y = ((q / math.sqrt(dim))[..., None, :] @ s).squeeze(-2)
        g = (x[:, t] @ w["g_proj.weight"].T).reshape(batch, heads, dim)
        y = oracle_norm(y, w["o_norm.weight"], c["rms_norm_eps"]) * torch.sigmoid(g)
        output.append(y.flatten(-2) @ w["o_proj.weight"].T)
    return torch.stack(output, 1), ref.KDAState(s, tuple(history))


def oracle_mla(x, w, c, position_start=0, state=None):
    heads, nope, rope, vd = (c[k] for k in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
    keys = [] if state is None else list(state.keys.to(x.dtype).unbind(1))
    values = [] if state is None else list(state.values.to(x.dtype).unbind(1))
    output = []
    for t in range(x.shape[1]):
        q = (x[:, t] @ w["q_proj.weight"].T).reshape(x.shape[0], heads, nope + rope)
        both = x[:, t] @ w["kv_a_proj_with_mqa.weight"].T
        latent = oracle_norm(both[:, :c["kv_lora_rank"]], w["kv_a_layernorm.weight"], c["rms_norm_eps"])
        expanded = (latent @ w["kv_b_proj.weight"].T).reshape(x.shape[0], heads, nope + vd)
        qr = oracle_rotate(q[..., nope:], position_start + t, c["rope_theta"])
        kr = oracle_rotate(both[:, c["kv_lora_rank"]:], position_start + t, c["rope_theta"])
        query = torch.cat([q[..., :nope], qr], -1)
        keys.append(torch.cat([expanded[..., :nope], kr[:, None, :].expand(-1, heads, -1)], -1))
        values.append(expanded[..., nope:])
        scores = torch.stack([(query * key).sum(-1) / math.sqrt(nope + rope) for key in keys], -1)
        probabilities = torch.exp(scores - scores.max(-1, keepdim=True).values)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        attended = sum(probabilities[..., i, None] * value for i, value in enumerate(values))
        gate = torch.sigmoid(x[:, t] @ w["g_proj.weight"].T)
        output.append((attended * gate[..., None]).flatten(-2) @ w["dense.weight"].T)
    return torch.stack(output, 1), ref.MLAState(torch.stack(keys, 1), torch.stack(values, 1))


def oracle_mlp(x, w, prefix, limit, variant="post_silu"):
    raw_gate = x @ w[prefix + "gate_proj.weight"].T
    up = x @ w[prefix + "up_proj.weight"].T
    if limit and variant == "pre_silu":
        raw_gate = torch.minimum(raw_gate, torch.tensor(limit, dtype=x.dtype))
    activated = raw_gate * torch.sigmoid(raw_gate)
    if limit and variant != "unclamped":
        if variant == "post_silu":
            activated = torch.minimum(activated, torch.tensor(limit, dtype=x.dtype))
        up = torch.maximum(torch.minimum(up, torch.tensor(limit, dtype=x.dtype)), torch.tensor(-limit, dtype=x.dtype))
    return (activated * up) @ w[prefix + "down_proj.weight"].T


def oracle_model(ids, w, c, variant="post_silu"):
    # Float64 fixture truth avoids inheriting evaluator FP32 reduction order.
    w = {name: value.double() for name, value in w.items()}
    x = w["model.word_embeddings.weight"][ids]
    captures, states = {}, {}
    for layer in range(c["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        y = oracle_norm(x, w[p + "input_layernorm.weight"], c["rms_norm_eps"])
        aw = {name[len(p + "attention."):]: value for name, value in w.items() if name.startswith(p + "attention.")}
        function = oracle_kda if (layer + 1) % c["layer_group_size"] else oracle_mla
        attention, states[layer] = function(y, aw, c)
        x = x + attention
        y = oracle_norm(x, w[p + "post_attention_layernorm.weight"], c["rms_norm_eps"])
        limit = c["expert_swiglu_limit_list"][layer]
        if layer < c["first_k_dense_replace"]:
            ffn = oracle_mlp(y, w, p + "mlp.", limit, variant)
        else:
            flat = y.reshape(-1, y.shape[-1])
            ids_e, mixing = oracle_route(flat, w[p + "mlp.gate.weight"], w[p + "mlp.gate.expert_bias"], c)
            ffn = torch.empty_like(flat)
            for row, token in enumerate(flat):
                routed = sum(mixing[row, slot] * oracle_mlp(token, w, p + f"mlp.experts.{expert}.", limit, variant)
                             for slot, expert in enumerate(ids_e[row].tolist()))
                ffn[row] = routed + oracle_mlp(token, w, p + "mlp.shared_experts.", c["share_expert_swiglu_limit_list"][layer], variant)
            ffn = ffn.reshape(y.shape)
        x = x + ffn
        captures[layer] = x.clone()
    x = oracle_norm(x, w["model.norm.weight"], c["rms_norm_eps"])
    return x, x @ w["lm_head.weight"].T, captures, states


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU fixture tests must not initialize CUDA")
        cls.temp = tempfile.TemporaryDirectory(prefix="ling-reference-", dir=os.environ["TMPDIR"])
        cls.base = Path(cls.temp.name)
        cls.config, cls.weights = make_fixture(cls.base / "fp32")
        make_fixture(cls.base / "bf16", torch.bfloat16)
        cls.ids = torch.tensor([[1, 6, 2, 11, 3, 15, 4], [10, 3, 7, 1, 12, 6, 2]], dtype=torch.long)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self, **kwargs):
        return ref.StreamingReference(self.base / "fp32", fixture=True, compute_dtype="float32", **kwargs)

    def close(self, actual, expected, atol=3e-5, rtol=3e-5):
        torch.testing.assert_close(actual.double(), expected.double(), atol=atol, rtol=rtol)

    def all_logits(self, model, result):
        return torch.cat([x for _, x in model.iter_logits(result, storage_dtype="float32", rows_per_chunk=3, vocab_rows=7)], 1)

    def test_full_fp32_fixture_against_independent_float64_model(self):
        expected, logits, layers, _ = oracle_model(self.ids, self.weights, self.config)
        observed = {}
        model = self.model(chunk_size=3)
        result = model.forward(self.ids, capture_layers=range(4), capture=lambda name, x, pos: observed.update({name: x}))
        self.close(result.hidden, expected)
        self.close(self.all_logits(model, result), logits)
        for layer, tensor in layers.items():
            self.close(observed[f"layer.{layer}"], tensor)
        self.close(observed["postnorm"], expected)
        self.assertIsNone(result.state)

    def test_kda_nonzero_asymmetric_state_and_history_against_matrix_oracle(self):
        c = self.config
        aw = {name.split("attention.", 1)[1]: value for name, value in self.weights.items() if name.startswith("model.layers.0.attention.")}
        generator = torch.Generator().manual_seed(195)
        x = torch.randn(2, 7, 8, generator=generator)
        prior = ref.KDAState(torch.randn(2, 2, 3, 3, generator=generator),
                            tuple(torch.randn(2, 6, 3, generator=generator) for _ in range(3)))
        original = ref.move_state(prior, "cpu")
        expected, expected_state = oracle_kda(x.double(), {k:v.double() for k,v in aw.items()}, c, prior)
        output, state = ref.kda_attention(x, aw, c, prior)
        self.close(output, expected)
        self.close(state.matrix, expected_state.matrix)
        parts, continuation = [], prior
        for start in (0, 2, 4, 6):
            part, continuation = ref.kda_attention(x[:, start:start+2], aw, c, continuation)
            parts.append(part)
        self.close(torch.cat(parts, 1), expected)
        self.close(continuation.matrix, expected_state.matrix)
        torch.testing.assert_close(prior.matrix, original.matrix, rtol=0, atol=0)
        for current, wanted, old, original_old in zip(state.conv, expected_state.conv, prior.conv, original.conv):
            self.close(current, wanted)
            torch.testing.assert_close(old, original_old, rtol=0, atol=0)
            self.assertLessEqual(current.untyped_storage().nbytes(), current.numel() * current.element_size())
        self.assertEqual(state.matrix.dtype, torch.float32)

    def test_mla_nonzero_prefix_complex_rope_and_head_gate(self):
        c = self.config
        aw = {name.split("attention.", 1)[1]: value for name, value in self.weights.items() if name.startswith("model.layers.1.attention.")}
        generator = torch.Generator().manual_seed(813)
        x = torch.randn(2, 7, 8, generator=generator)
        prior = ref.MLAState(torch.randn(2, 3, 2, 7, generator=generator), torch.randn(2, 3, 2, 2, generator=generator))
        expected, expected_state = oracle_mla(x.double(), {k:v.double() for k,v in aw.items()}, c, 3, prior)
        output, state = ref.mla_attention(x, aw, c, 3, prior)
        self.close(output, expected)
        self.close(state.keys, expected_state.keys)
        self.close(state.values, expected_state.values)
        pieces, continuation = [], prior
        for t in range(x.shape[1]):
            part, continuation = ref.mla_attention(x[:, t:t+1], aw, c, t+3, continuation)
            pieces.append(part)
        self.close(torch.cat(pieces, 1), expected)
        self.assertEqual(prior.keys.shape[1], 3)
        with self.assertRaisesRegex(ValueError, "boundary"):
            ref.mla_attention(x, aw, c, 2, prior)
        no_gate = dict(aw, **{"g_proj.weight": torch.zeros_like(aw["g_proj.weight"])})
        wrong, _ = ref.mla_attention(x, no_gate, c, 3, prior)
        self.assertGreater((wrong - output).abs().max().item(), .01)

    def test_token_chunk_and_nonzero_continuation_equivalence(self):
        model = self.model(chunk_size=3)
        whole = model.forward(self.ids, keep_state=True)
        prefix = model.forward(self.ids[:, :2], keep_state=True)
        original_prefix = prefix.state.clone()
        parts, state = [prefix.hidden], prefix.state
        for t in range(2, self.ids.shape[1]):
            result = model.forward(self.ids[:, t:t+1], state=state, keep_state=True)
            parts.append(result.hidden)
            state = result.state
        self.close(torch.cat(parts, 1), whole.hidden)
        self.assertEqual(state.tokens, self.ids.shape[1])
        for layer, value in state.layers.items():
            expected = whole.state.layers[layer]
            if isinstance(value, ref.KDAState):
                self.close(value.matrix, expected.matrix)
                self.close(prefix.state.layers[layer].matrix, original_prefix.layers[layer].matrix, atol=0, rtol=0)
                for actual, wanted in zip(value.conv, expected.conv):
                    self.close(actual, wanted)
            else:
                self.close(value.keys, expected.keys)
                self.close(value.values, expected.values)
        suffix = model.forward(self.ids[:, 2:], state=prefix.state, keep_state=True)
        self.close(suffix.hidden, whole.hidden[:, 2:])
        self.assertEqual(suffix.position_start, 2)

    def test_chunk_boundaries_lengths_and_batch_order(self):
        for length in (1, 2, 3, 4, 7, 31, 32, 33, 127, 128, 129):
            with self.subTest(length=length):
                ids = (torch.arange(length)[None, :] * 7 + 3) % 19
                a = self.model(chunk_size=3).forward(ids)
                b = self.model(chunk_size=32).forward(ids)
                self.close(a.hidden, b.hidden)
        model = self.model(chunk_size=3)
        original = model.forward(self.ids)
        permuted = model.forward(self.ids.flip(0))
        self.close(original.hidden, permuted.hidden.flip(0))
        self.close(original.hidden[:1], model.forward(self.ids[:1]).hidden)

    def test_group_routing_bias_negative_mask_ties_and_underflow(self):
        c = dict(self.config)
        x = torch.tensor([[1., -.2], [.3, .7]])
        weight = torch.tensor([[1., -.3], [-2., .1], [.4, .7], [-.1, .2], [3., -1.], [-3., .4]])
        bias = torch.tensor([-2., -2., -1.8, -1.9, -2.5, -2.3])
        ids, mixing = ref.grouped_route(x, weight, bias, 3, 2, 3, 2.5)
        expected_ids, expected_weights = oracle_route(x.double(), weight.double(), bias.double(), c)
        torch.testing.assert_close(ids, expected_ids)
        self.close(mixing, expected_weights, atol=1e-6, rtol=1e-6)
        self.close(mixing.sum(-1), torch.full((2,), 2.5))
        unbiased_selection, _ = ref.grouped_route(x, weight, torch.zeros_like(bias), 3, 2, 3, 2.5)
        self.assertFalse(torch.equal(ids, unbiased_selection))
        ties, _ = ref.grouped_route(torch.zeros(1,2), torch.zeros(6,2), torch.zeros(6), 3, 2, 3, 2.5)
        self.assertEqual(ties.tolist(), [[0, 1, 2]])
        _, zeros = ref.grouped_route(torch.ones(1,2), torch.full((6,2), -1000.), bias, 3, 2, 3, 2.5)
        self.assertTrue(torch.equal(zeros, torch.zeros_like(zeros)))

    def test_clamp_variants_are_explicit_negative_controls(self):
        model = self.model(chunk_size=3)
        correct = self.all_logits(model, model.forward(self.ids))
        for variant in ("pre_silu", "unclamped"):
            negative = self.model(chunk_size=3, negative_control=variant)
            wrong = self.all_logits(negative, negative.forward(self.ids))
            _, oracle_logits, _, _ = oracle_model(self.ids, self.weights, self.config, variant)
            self.close(wrong, oracle_logits)
            self.assertGreater((wrong - correct).abs().max().item(), .001)
            self.assertTrue(negative.provenance["negative_control"])
        # Distinct shared schedule matters independently of routed limits.
        modified = dict(self.config, share_expert_swiglu_limit_list=self.config["expert_swiglu_limit_list"])
        _, conflated, _, _ = oracle_model(self.ids, self.weights, modified)
        self.assertGreater((conflated.float() - correct).abs().max().item(), .001)
        self.assertEqual(model.provenance["clamp"], "post_silu")
        self.assertFalse(model.provenance["negative_control"])

    def test_bf16_fixture_storage_state_and_exact_fresh_replay(self):
        kwargs = dict(fixture=True, compute_dtype="bfloat16", chunk_size=3)
        first = ref.StreamingReference(self.base / "bf16", **kwargs)
        result = first.forward(self.ids, keep_state=True)
        self.assertEqual(result.hidden.dtype, torch.bfloat16)
        self.assertEqual(result.state.layers[0].matrix.dtype, torch.float32)
        self.assertEqual(result.state.layers[1].keys.dtype, torch.bfloat16)
        logits = torch.cat([x for _,x in first.iter_logits(result, rows_per_chunk=3, vocab_rows=7)], 1)
        cache_path = self.base / "bf16-stored-logits.safetensors"
        save_file({"logits": logits, "input_ids": self.ids}, str(cache_path),
                  metadata={"qualification": ref.FIXTURE_LABEL, "storage_dtype": "bfloat16"})
        with safe_open(str(cache_path), framework="pt", device="cpu") as handle:
            stored = handle.get_tensor("logits").clone()
            torch.testing.assert_close(handle.get_tensor("input_ids"), self.ids, rtol=0, atol=0)
        torch.testing.assert_close(logits, stored, rtol=0, atol=0)
        second = ref.StreamingReference(self.base / "bf16", **kwargs)
        replay = torch.cat([x for _,x in second.iter_logits(second.forward(self.ids), rows_per_chunk=3, vocab_rows=7)], 1)
        self.assertEqual(logits.dtype, torch.bfloat16)
        torch.testing.assert_close(logits, replay, atol=0, rtol=0)
        logp, logq = stored.double().log_softmax(-1), replay.double().log_softmax(-1)
        self.assertEqual(float((logp.exp() * (logp-logq)).sum()), 0.)
        wrong_row = logq.roll(1, dims=1)
        self.assertGreater(float((logp.exp() * (logp-wrong_row)).sum(-1).mean()), 1e-4)
        self.assertEqual(first.provenance["head_gemm_dtype"], "bfloat16")

    def test_bounded_weight_lifecycle_and_failure_cleanup(self):
        model = self.model(chunk_size=3, max_weight_bytes=8192)
        refs = []
        model.store.observer = lambda name, tensor: refs.append(weakref.ref(tensor))
        result = model.forward(self.ids, keep_state=True)
        self.all_logits(model, result)
        gc.collect()
        self.assertEqual(model.store.live_weight_bytes, 0)
        self.assertEqual(model.store.live_tensors, 0)
        self.assertTrue(all(r() is None for r in refs))
        self.assertLess(model.store.peak_weight_bytes, model.store.inventory["tensor_bytes"])
        self.assertLessEqual(model.store.peak_weight_bytes, 8192)
        self.assertLess(model.store.peak_tensors, model.store.inventory["tensor_count"])
        names = ["model.layers.0.attention.q_proj.weight", "model.layers.0.attention.k_proj.weight"]
        model.store.max_weight_bytes = 200
        with self.assertRaises(MemoryError):
            with model.store.scope(names, "cpu", torch.float32):
                self.fail("budget should fail during the second read")
        self.assertEqual(model.store.live_weight_bytes, 0)
        model.store.max_weight_bytes = 8192
        def fail(*args):
            raise RuntimeError("injected capture failure")
        with self.assertRaisesRegex(RuntimeError, "injected capture"):
            model.forward(self.ids, capture_layers=[0], capture=fail)
        gc.collect()
        self.assertTrue(all(r() is None for r in refs))
        self.assertEqual(model.store.live_weight_bytes, 0)

    def test_loader_and_input_fail_closed(self):
        model = self.model()
        for ids in (torch.tensor([[1.]]), torch.tensor([[19]]), torch.empty(1,0,dtype=torch.long)):
            with self.assertRaises(ValueError):
                model.forward(ids)
        with self.assertRaises(ValueError):
            self.model(max_context=2).forward(self.ids)
        with self.assertRaises(ValueError):
            model.forward(self.ids, capture_layers=[4])
        state = model.forward(self.ids[:, :2], keep_state=True).state
        with self.assertRaises(ValueError):
            self.model(negative_control="unclamped").forward(self.ids[:, 2:], state=state)
        with self.assertRaises(ValueError):
            ref.StreamingReference(self.base / "fp32", compute_dtype="float32")
        with self.assertRaises(ValueError):
            ref.StreamingReference(self.base / "fp32", source_revision="main")
        for key, value in (("rope_interleave", False), ("q_lora_rank", 2), ("expert_swiglu_limit_list", [0]),
                           ("topk_group", 4), ("head_dim", True), ("num_experts_per_tok", 1)):
            bad = dict(self.config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                ref.validate_config(bad)
        path = self.base / "malformed"
        make_fixture(path)
        index_path = path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        name = next(iter(index["weight_map"]))
        index["weight_map"][name] = "../outside.safetensors"
        index_path.write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            ref.StreamingReference(path, fixture=True, compute_dtype="float32")
        make_fixture(path)
        (path / "model-00.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            ref.StreamingReference(path, fixture=True, compute_dtype="float32")
        make_fixture(path)
        changed = ref.StreamingReference(path, fixture=True, compute_dtype="float32")
        with (path / "model-00.safetensors").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            changed.forward(self.ids)


    def test_real_cli_capture_identity_hashes_and_storage(self):
        tokens = self.base / "heldout.json"
        corpus = {"purpose": "held-out", "provenance": "synthetic fixture only; not a real held-out corpus",
                  "sequences": [{"sequence_id": "fixture/A", "input_ids": self.ids[0].tolist(), "score_start": 2},
                                {"sequence_id": "fixture-B", "input_ids": self.ids[1].tolist(), "score_start": 0}]}
        tokens.write_text(json.dumps(corpus))
        output = self.base / "cli-output"
        command = [sys.executable, "-B", "-m", "ling3_repro.streaming", "--model", str(self.base / "fp32"),
                   "--fixture", "--compute-dtype", "float32", "--logit-storage-dtype", "float32", "--device", "cpu",
                   "--tokens", str(tokens), "--output", str(output), "--chunk-size", "3", "--logit-rows", "3",
                   "--head-vocab-rows", "7", "--capture-layer", "0", "--capture-layer", "3"]
        process = subprocess.run(command, text=True, capture_output=True, env=dict(os.environ), timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        response = json.loads(process.stdout)
        self.assertEqual(response["status"], "complete")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["token_file_sha256"], ref.sha256_file(tokens))
        self.assertEqual(manifest["memory_accounting"]["live_owned_weight_bytes"], 0)
        self.assertFalse(manifest["reference"]["inventory"]["payload_hashes_verified"])
        self.assertEqual(manifest["reference"]["mtp"], "NOT IMPLEMENTED; target only")
        for entry in manifest["files"]:
            self.assertEqual(ref.sha256_file(output / entry["file"]), entry["sha256"])
        for i, seq in enumerate(corpus["sequences"]):
            arrays, seen = [], []
            for path in sorted(output.glob(f"sequence-{i:04d}-logits-*.safetensors")):
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    positions = handle.get_tensor("positions")
                    values = handle.get_tensor("logits")
                    arrays.append(values)
                    seen.extend(positions.tolist())
                    torch.testing.assert_close(handle.get_tensor("input_ids"), self.ids[i:i+1, positions])
                    target = handle.get_tensor("target_ids")
                    for row, position in enumerate(positions.tolist()):
                        self.assertEqual(target[row].item(), seq["input_ids"][position+1] if position+1 < len(seq["input_ids"]) else -1)
                    expected_mask = (positions >= seq["score_start"]) & (positions < len(seq["input_ids"])-1)
                    torch.testing.assert_close(handle.get_tensor("score_mask"), expected_mask)
                    self.assertEqual(handle.metadata()["sequence_id"], seq["sequence_id"])
            self.assertEqual(seen, list(range(len(seq["input_ids"]))))
            model = self.model(chunk_size=3)
            fresh = self.all_logits(model, model.forward(self.ids[i:i+1]))
            torch.testing.assert_close(torch.cat(arrays, 1), fresh, rtol=0, atol=0)
        from ling3_repro.diagnostic import load_capture, score
        loaded, sequences, logits = load_capture(output)
        self.assertEqual(loaded["schema"], "ling3-portable-reference-v1")
        self.assertEqual(sequences, corpus["sequences"])
        self.assertEqual(score(output, output)["kl_ref_candidate"]["mean"], 0)
        repeat = subprocess.run(command, text=True, capture_output=True, timeout=60)
        self.assertNotEqual(repeat.returncode, 0)
        self.assertIn("FileExistsError", repeat.stderr)

    def test_no_engine_remote_import_or_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(any(name == "exllamav3" or name.startswith("exllamav3.") for name in sys.modules))
        self.assertFalse(any(name == "transformers" or name.startswith("transformers.") for name in sys.modules))
        self.assertFalse(any("modeling_bailing" in name for name in sys.modules))

    def test_precision_identity_guards_and_observer_failure(self):
        model = self.model(chunk_size=3)
        result = model.forward(self.ids)
        other = self.model(negative_control="unclamped")
        with self.assertRaisesRegex(ValueError, "identity"):
            next(other.iter_logits(result))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast"):
                model.forward(self.ids)
            with self.assertRaisesRegex(ValueError, "autocast"):
                next(model.iter_logits(result))
        def failing_observer(name, tensor):
            raise RuntimeError("observer failure")
        model.store.observer = failing_observer
        with self.assertRaisesRegex(RuntimeError, "observer failure"):
            model.forward(self.ids)
        self.assertEqual(model.store.live_weight_bytes, 0)
        self.assertEqual(model.store.live_tensors, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
