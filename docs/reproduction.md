# Completed 8-bit path: reproducibility boundary

This collection describes the **completed original → EXL3 8/8/8 conversion**,
its integrity checks and a completed bounded diagnostic. It does not package
unfinished low-bit or trace revisions. Historical results are not reexecuted
results. External review currently holds GPU/conversion work; these instructions
are documentation, not launch authorization.

## Pinned inputs and runtime

- Original: `inclusionAI/Ling-3.0-flash`, revision
  `ef06d91fe382109ae82647da88ff99b0f11745b0`.
- Public engine: <https://github.com/everson/exllamav3>, commit
  `a7b05152924da3b2a88d9c2cff0a4e2d6703157b`.
- Historical conversion recorded engine HEAD
  `f4db6980e6a74d9e7dbfb6e18376f65bb2454368` plus an independently frozen tree.
  Do **not** relabel that measurement as execution of the public commit. The
  public pinned converter's argument definitions were inspected for this recipe;
  conversion equivalence at that commit has not been rerun here.
- Historical Python 3.12.3, Torch 2.14.0+cu130, CUDA runtime 13.0,
  CUDA build toolkit 13.1, RTX A6000. Use an isolated engine environment, never
  mutate a serving installation. This repository does not install the engine.

## CLI transcription, not the original guarded orchestrator

Choose absolute values for `ENGINE`, `PYTHON`, `SOURCE`, `CALIBRATION`, `WORK`
and `OUTPUT`. `ENGINE` is the pinned local checkout and `PYTHON` its provisioned
Python interpreter. `SOURCE` is a verified local copy of the pinned original.
`WORK` and `OUTPUT` must be new, separate directories. `A6000_UUID` is the locally
verified physical A6000 UUID; it must not select some other GPU. No device ID is
bundled. Provisioning/downloads and resource admission are deliberately separate.

After review and explicit GPU authorization, the matching converter CLI is:

```sh
(
set -eu
: "${ENGINE:?}" "${PYTHON:?}" "${SOURCE:?}" "${CALIBRATION:?}" "${WORK:?}" "${OUTPUT:?}" "${A6000_UUID:?}"
test "$(git -C "$ENGINE" rev-parse HEAD)" = a7b05152924da3b2a88d9c2cff0a4e2d6703157b
# Inspect git status too: HEAD alone does not establish an unchanged tree.
test -f "$CALIBRATION"
test ! -e "$WORK" && test ! -e "$OUTPUT"
cd "$ENGINE"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$A6000_UUID" \
EXL3_GDN_PROJ_FP32=0 EXL3_GDN_GATE_FP32=1 EXL3_GDN_CONV_TOKEN_MAJOR=1 \
EXL3_QKV_SLICE=1 EXL3_BC_GDN=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -B convert.py -i "$SOURCE" -w "$WORK" -o "$OUTPUT" \
  -b 8 -hb 8 -mb 8 -hq -cd "$CALIBRATION" -cr 250 -cc 2048 \
  -ss 8192 -d 0 -dr 1 -cb mul1 --out_scales always -cpi 120
)
```

The historical calibration had 250 rows of 2048 real valid tokens each. It is not
bundled: **no redistribution-cleared calibration corpus is included**. User-supplied
calibration must be screened for privacy, source/transitive rights, correct Ling
tokenizer and valid lengths, and overlap against protected evaluation material.
Do not silently fall back to the engine's default corpus. Different calibration
means a different conversion, not a bit-identical replay of these results.

This concise CLI does **not** reproduce the omitted historical guard. That guard
set and monitored strict Torch precision/determinism, disabled reduced FP16/BF16
reductions and TF32, enforced no recovery/CPU factorization, observed compiler
ownership before merge, bound code/calibration/source inputs, and checked work
versus compiled tensor bytes plus finite floating payloads. Environment variables
alone do not reproduce these policies. The portable validator below does not
implement them. In particular the CLI is not a replacement admission mechanism
for the historical run. Do not claim its output has those guarantees.

Native auxiliaries remain native. All 63,201 eligible matrices were assigned
8 bits: decoder 61,657, head 1, MTP 1,543. MTP is intentionally **uncalibrated**
(`mtp_calibration=none`, authorized MTP fallback); decoder/head fallback was not
admitted. Neither conversion nor this exception qualifies MTP generation.

## Completion metadata and integrity

Converter exit zero is insufficient: preparation can report an error and exit
zero. Require an actual compiled pack, config, tokenizer assets, shard index and
`quantization_config.json`. The completed pack recorded `quant_method=exl3`,
`version=1.5.1`, `bits=head_bits=mtp_bits=8`, `codebook=mul1`,
`out_scales=always`, calibration `rows=250, cols=2048`, and `tensor_storage`.
Its storage metadata covered 61,658 target quantized leaves but listed **zero MTP
quantized leaves**; therefore it is not a sufficient all-model/MTP coverage oracle.
Historical physical ownership and the independent per-linear ledger established
MTP coverage. Retain equivalent independent evidence for a new conversion.
Configuration values assert settings, not numerical accuracy or model quality.

From any working directory, with `REPRO` pointing at this checkout:

```sh
python3 -I -S -B "$REPRO/tools/verify_pack.py" "$OUTPUT"
python3 -I -S -B "$REPRO/tools/verify_pack.py" "$OUTPUT" --checksums "$CHECKSUMS"
python3 -I -S -B "$REPRO/tools/test_verify_pack.py" -v
```

`CHECKSUMS` is a trusted, separately obtained JSON object mapping each basename to
`{"sha256": "<64 lowercase hex characters>", "size_bytes": <integer>}`.
It must include every shard and the index; include config/tokenizer/quantization
metadata too when verifying those assets. Original checkpoints should be checked
against independent pinned LFS sizes/digests. A manifest created from the same
untrusted output proves consistency, not authenticity. No trusted weight manifest
is bundled with this source-only package.

The tool checks duplicate JSON keys, supported dtype/shape byte counts, bounded
headers, exact contiguous payload offsets, unique tensor ownership, physical/index
shard closure, optional index `total_size`, and supplied whole-file SHA-256/size.
JSON reads are capped at 128 MiB, each header at 16 MiB; hashes stream in 1 MiB
chunks. Memory still scales with bounded parsed headers/index. Only flat basenames
and regular non-symlink files are supported. Unsupported dtypes fail explicitly;
this is intentionally not a universal safetensors implementation.

Without `--checksums`, **payload bytes are not read or hashed**. With checksums,
only the listed files are hashed (every shard/index is required; unrelated assets
are not an all-directory inventory). There is no finite-value scan, EXL3 numerical
validation, expected architecture census, tokenizer validation, or work-to-output
comparison. Keep inputs quiescent: this tool is not a concurrent/adversarial file
mutation defense or a future-immutability guarantee. Its CPU fixtures are synthetic
format tests, not inference or a fresh scan of the historical weights.

## Evidence and redistribution

[measured-results.json](measured-results.json) contains only selected aggregate
values read from completed historical receipts: conversion worker report,
artifact validator report, offload load report and successful candidate diagnostic
comparison. These private source receipts and raw probes are deliberately not
bundled; descriptive evidence labels are not downloadable public artifacts.
This limits independent replay from the repository alone. No generated text,
reversible token rows, logits, personal IDs or machine-specific paths are included.

The bounded diagnostic uses 384 scored positions from two prior probes, full
157,184-column normalization, original BF16 reference versus EXL3 FP16/mixed
execution. It mixes weight, backend, cache and rounding differences. Exact fresh
candidate replay and sensitive wrong-row controls are evidence of that diagnostic's
self-consistency, not representative quality. Historical original-reference process
identity limitations remain; they are not retroactively repaired. No throughput
benchmark, production readiness, low-bit result, or MTP acceptance claim follows.

Source code licensing does not grant rights to weights or third-party calibration.
Obtain weights under the pinned model's applicable license and preserve its notices;
review upstream engine terms separately. Screen every proposed corpus component,
including incorporated material, before use or redistribution.
