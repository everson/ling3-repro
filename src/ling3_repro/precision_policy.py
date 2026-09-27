"""Observed numerical policy and explicit setup for a dedicated reference worker.

Import and observation do not initialize CUDA or change caller-global settings.
The strict setter is opt-in, requires a new process, and is never called by the
reference evaluator itself. It does not establish undocumented kernel behavior.
"""
import os
import torch

STRICT_NAME = 'ling-reference-strict-bf16-v1'
BOOL_FIELDS = (
    'matmul_allow_tf32', 'allow_bf16_reduced_precision_reduction',
    'allow_fp16_reduced_precision_reduction', 'deterministic_algorithms',
    'deterministic_warn_only', 'cudnn_deterministic', 'cudnn_benchmark',
    'cudnn_allow_tf32',
)
STRICT_FLAGS = dict(zip(BOOL_FIELDS, (False, False, False, True, False, True, False, False)))


def observe_precision():
    return {
        'schema': 1,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    }


def validate_observation(value):
    expected = set(BOOL_FIELDS) | {'schema', 'float32_matmul_precision', 'cublas_workspace_config'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('invalid numerical backend policy fields')
    if type(value['schema']) is not int or value['schema'] != 1:
        raise ValueError('invalid numerical backend policy schema')
    if any(type(value[k]) is not bool for k in BOOL_FIELDS):
        raise ValueError('invalid numerical backend boolean policy')
    if value['float32_matmul_precision'] not in ('highest', 'high', 'medium'):
        raise ValueError('invalid FP32 matmul precision')
    workspace = value['cublas_workspace_config']
    if workspace is not None and not isinstance(workspace, str):
        raise ValueError('invalid cuBLAS workspace declaration')


def require_strict(value):
    validate_observation(value)
    expected = {**STRICT_FLAGS, 'schema': 1, 'float32_matmul_precision': 'highest',
                'cublas_workspace_config': ':4096:8'}
    if value != expected:
        raise ValueError('strict reference numerical policy not active')


def configure_strict():
    if torch.cuda.is_initialized():
        raise RuntimeError('strict reference policy must precede CUDA initialization')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python')
    # Use the established legacy precision-API family consistently, not mixed
    # with per-backend fp32_precision APIs.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    value = observe_precision()
    require_strict(value)
    return value
