# Ling-3.0-Flash experiment companion

Source-only companion to [the experimental ExLlamaV3 implementation](https://github.com/everson/exllamav3/tree/a7b05152924da3b2a88d9c2cff0a4e2d6703157b). One current implementation per tool; no archived scripts or model artifacts.

## Included

- Layer-streamed original-weight reference evaluator and independent mathematical fixtures.
- Teacher-forced candidate capture API and full-vocabulary CPU KL/NLL/top-1 scorer.
- Standalone safetensors/index integrity checker with optional trusted checksums.
- [Pinned conversion recipe and evidence boundaries](docs/reproduction.md).
- [Selected historical measured results](docs/measured-results.json), explicitly separate from tests of this portable package.

**This is not a turnkey replay of the historical experiment.** The portable evaluator and new capture schemas have CPU fixture coverage. Real-model/GPU execution of this adaptation has not been rerun. The candidate API requires a caller-loaded ExLlama model/cache; a qualified portable offload launcher is not included. Historical calibration/probe inputs are not redistributed, so exact historical metrics cannot be reproduced from this checkout alone. No unfinished lower-bit conversion or trace pipeline is included.

## CPU setup and checks

Use an isolated Python environment with Torch and safetensors. Tested locally with Python 3.12, Torch 2.14.0+cu130 and safetensors 0.8.0; declared minimum dependency versions are not a tested compatibility matrix. Installing Torch may require selecting its appropriate distribution for your machine.

```sh
# REPRO is this checkout's absolute path, chosen by you.
python -m pip install -e "$REPRO"
# These commands also work from outside the checkout after installation.
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s "$REPRO/tests" -v
python -I -S -B "$REPRO/tools/test_verify_pack.py" -v
python -m ling3_repro.streaming --help
python -m ling3_repro.diagnostic --help
```

Tests generate small synthetic tensors in temporary directories; they need no weights, engine import or GPU. They check equations, BF16 replay, actual CLI capture/scoring, row identities, full-vocabulary normalization, malformed captures and pack corruption. They do not establish real-model fidelity.

## Reference capture and scoring

Supply your own licensed, screened exact token sequences in a JSON object with `sequences`, each containing `sequence_id`, `input_ids` and zero-based `score_start`. Inputs must use the pinned model's tokenizer. Next-token targets are derived from the sequence; the final input position has no target. No real token corpus is bundled. Inspect the CLI before choosing memory/context limits: full original evaluation is resource-intensive even though weights stream.

```sh
python -m ling3_repro.streaming --model "$MODEL" --tokens "$TOKENS" --output "$REFERENCE"
python -m ling3_repro.diagnostic --reference "$REFERENCE" --candidate "$CANDIDATE" --output "$SCORES"
```

The reference defaults to CPU/BF16. GPU requires an explicit device plus `--allow-gpu`; that flag is not resource admission. Output locations must be new. `capture_candidate` in `ling3_repro.diagnostic` accepts an already-loaded model/cache and writes the candidate schema; CPU tests exercise its transport contract, not native loading. It assumes a 2,048-token cache contract. Do not use it as an unqualified long-context runner.

Compare independent fresh captures and run `--wrong-rows` for a within-sequence row-roll sensitivity control. That control differs from historical row-reversal controls; their numbers are not interchangeable. The scorer materializes bounded captures in CPU memory. Capture hashes and declared engine/model identities are consistency metadata, not independent authentication of model payloads or runtime identity.

## Privacy, publication and scope

No weights, calibration/probe corpus, reversible real tokens, logits, raw receipts, generated completions, credentials or local machine identifiers are included. Generated outputs belong outside the repository or in ignored `outputs/`. Review new files before staging: `.gitignore` is not a secret detector. Check both contents and Git author metadata before publishing.

Source inspection and CPU tests are available now; real-model reproduction still requires independently obtained inputs, engine provisioning, resource admission and additional integration work. Source-code rights do not grant rights to model weights or corpus material. No blanket license for third-party inputs is asserted.
