"""Independent Torch math oracles for Ling qualification, not an inference backend.

No imports from ExLlamaV3, Transformers modeling, FLA or downloaded model code.
Operations use explicit FP32 (or requested float64) and expose state/precision.
The production clamp decision is post-SiLU; other variants are negative controls.
"""
import math
import torch
import torch.nn.functional as F


def swiglu(gate, up, limit = 0.0, variant = "post_silu"):
    if variant not in ("post_silu", "pre_silu", "unclamped"):
        raise ValueError(variant)
    if limit > 0 and variant == "pre_silu":
        return F.silu(gate.clamp(max = limit)) * up.clamp(-limit, limit)
    activated = F.silu(gate)
    if limit > 0 and variant == "post_silu":
        activated = activated.clamp(max = limit)
        up = up.clamp(-limit, limit)
    return activated * up


def safe_kda_decay(f, a_log, dt_bias, lower_bound = -5.0):
    if not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("safe KDA lower_bound must be finite and negative")
    if f.ndim != 4 or a_log.numel() != f.shape[-2] or dt_bias.numel() != f.shape[-2] * f.shape[-1]:
        raise ValueError("safe KDA expects f[B,T,H,D], A_log[H], dt_bias[H*D]")
    a = a_log.float().reshape(f.shape[-2], 1).exp()
    bias = dt_bias.float().reshape(f.shape[-2:])
    return lower_bound * torch.sigmoid(a * (f.float() + bias))


def causal_conv(x, weight, state = None):
    """x[B,T,C], weight[C,K], oldest-to-newest history[B,C,K-1]."""
    batch, tokens, channels = x.shape
    width = weight.shape[-1]
    if weight.shape[0] != channels or width < 1:
        raise ValueError("convolution shape mismatch")
    history = x.new_zeros(batch, channels, width - 1) if state is None else state.clone()
    if history.shape != (batch, channels, width - 1):
        raise ValueError("convolution state shape mismatch")
    output = []
    for i in range(tokens):
        window = torch.cat([history, x[:, i, :, None]], dim = -1)
        output.append(F.silu((window.float() * weight.float()[None]).sum(-1)))
        history = window[..., 1:]
    result = torch.stack(output, dim = 1) if output else x.new_empty(batch, 0, channels)
    return result, history.contiguous()


def kda_recurrence(q, k, v, log_decay, beta, state = None, normalize = True):
    """Sequential delta rule with key-axis decay before residual, FP32 matrix state.

    q,k[B,T,H,Dk], v[B,T,H,Dv], log_decay[B,T,H,Dk], beta[B,T,H].
    Nonzero prior state is accepted and never mutated by the oracle.
    """
    batch, tokens, heads, kd = q.shape
    if k.shape != q.shape or log_decay.shape != q.shape or beta.shape != q.shape[:-1]:
        raise ValueError("KDA q/k/gate shape mismatch")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("KDA value shape mismatch")
    q, k, v = q.float(), k.float(), v.float()
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim = True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim = True) + 1e-6)
    q = q / math.sqrt(kd)
    shape = (batch, heads, kd, v.shape[-1])
    s = torch.zeros(shape, device = q.device, dtype = torch.float) if state is None else state.float().clone()
    if s.shape != shape:
        raise ValueError("KDA state shape mismatch")
    output = []
    for i in range(tokens):
        s = s * log_decay[:, i].float().exp().unsqueeze(-1)
        predicted = (s * k[:, i].unsqueeze(-1)).sum(-2)
        correction = beta[:, i].float().unsqueeze(-1) * (v[:, i] - predicted)
        s = s + k[:, i].unsqueeze(-1) * correction.unsqueeze(-2)
        output.append((s * q[:, i].unsqueeze(-1)).sum(-2))
    result = torch.stack(output, 1) if output else v.new_empty(batch, 0, heads, v.shape[-1])
    return result, s


def gated_head_norm(x, weight, gate, epsilon = 1e-6):
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + epsilon) * weight.float() * gate.float().sigmoid()


def interleaved_rope(x, positions, theta):
    """Adjacent complex pairs on the already-selected rotary slice."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("rotary slice must be even")
    inv = theta ** (-torch.arange(0, dim, 2, device = x.device, dtype = torch.float) / dim)
    phase = positions.float()[..., None] * inv
    while phase.ndim < x.ndim:
        phase = phase.unsqueeze(-2)
    even, odd = x.float()[..., ::2], x.float()[..., 1::2]
    return torch.stack([even * phase.cos() - odd * phase.sin(),
                        odd * phase.cos() + even * phase.sin()], -1).flatten(-2)


def head_gated_attention(q, k, v, gate, causal = True):
    """Expanded dense attention oracle, never reused as the production MLA path."""
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        qt, kt = q.shape[-2], k.shape[-2]
        # Query suffix positions permit testing decode against a longer cached key prefix.
        allowed = torch.arange(kt, device = q.device)[None, :] <= \
            (torch.arange(qt, device = q.device) + kt - qt)[:, None]
        scores = scores.masked_fill(~allowed, -float("inf"))
    values = (scores.softmax(-1) @ v).transpose(1, 2)
    return values * gate.float().sigmoid().unsqueeze(-1)
