"""New portable diagnostic schema; not a reader for archived R2 evidence.

FP64 CPU scoring retains every head column. Candidate capture is an API for an
explicitly caller-loaded engine/cache; this module never imports an engine.
"""
import argparse
import json
from pathlib import Path
import hashlib
import torch
from safetensors.torch import load_file, save_file
from .streaming import read_json

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def row_identity(seq, start, count, vocab):
    ids = torch.tensor(seq['input_ids'], dtype=torch.int64)
    pos = torch.arange(start, start + count, dtype=torch.int64)
    targets = torch.full((count,), -1, dtype=torch.int64)
    valid = pos < len(ids) - 1
    targets[valid] = ids[pos[valid] + 1]
    return {'positions': pos, 'input_ids': ids[None, start:start + count],
            'target_ids': targets, 'score_mask': valid & (pos >= seq['score_start']),
            'column_ids': torch.arange(vocab, dtype=torch.int64)}

def validate_rows(data, seq, start, count, dtype, vocab):
    expected = row_identity(seq, start, count, vocab)
    require(set(data) == {'logits', *expected}, 'tensor key closure')
    logits = data['logits']
    require(logits.shape == (1, count, vocab), 'batch/row/full vocabulary shape')
    require(logits.dtype == dtype and bool(torch.isfinite(logits).all()), 'dtype/nonfinite logits')
    for k, t in expected.items():
        require(data[k].dtype == t.dtype and torch.equal(data[k], t), 'row/domain identity: ' + k)

def forward_rows(model, cache, seq, vocab, cache_tokens=2048):
    """Teacher-force one token per call, never sample or mask the raw model head.
    Explicit CPU test adapters implement model/cache transport only.
    """
    state = cache.get_new_state()
    try:
        for position, token in enumerate(seq['input_ids']):
            require(state.position == position, 'recurrent/cache position drift before forward')
            params = {'attn_mode': 'flash_attn', 'cache': cache,
                      'batch_shape': (1, cache_tokens), 'past_len': position,
                      'recurrent_states': [state], 'last_tokens_only': 1}
            logits = model.forward(torch.tensor([[token]], dtype=torch.int64), params)
            require(state.position == position + 1, 'engine failed to advance recurrent state')
            require(logits.shape == (1, 1, vocab), 'raw head must retain all columns')
            require(logits.dtype == torch.float16, 'unexpected native logit dtype; do not silently cast')
            require(bool(torch.isfinite(logits).all()), 'nonfinite raw logits')
            yield position, logits.detach().cpu().contiguous()
    finally:
        cache.release_state(state)

def capture_sequence(model, cache, seq, index, output, vocab):
    files, pending = [], []
    with torch.inference_mode():
        for position, row in forward_rows(model, cache, seq, vocab):
            pending.append(row)
            if len(pending) == 16 or position == len(seq['input_ids']) - 1:
                start = position + 1 - len(pending)
                name = f'sequence-{index:04d}-logits-{start:08d}.safetensors'
                data = {**row_identity(seq, start, len(pending), vocab), 'logits': torch.cat(pending, dim=1)}
                validate_rows(data, seq, start, len(pending), torch.float16, vocab)
                save_file(data, str(output / name))
                files.append({'file': name, 'sha256': sha(output / name), 'sequence_id': seq['sequence_id'],
                              'start': start, 'count': len(pending)})
                pending.clear()
    return files

def metrics(ref, candidate, seq, wrong_rows=False):
    """FP64 CPU normalization/reduction; natural logs; explicit within-sequence row-roll control."""
    require(ref.shape == candidate.shape and ref.ndim == 3 and ref.shape[0] == 1, 'comparison shape')
    require(ref.shape[1] == len(seq['input_ids']), 'comparison sequence length')
    require(bool(torch.isfinite(ref).all()) and bool(torch.isfinite(candidate).all()), 'comparison nonfinite')
    rows = []
    n = len(seq['input_ids'])
    for t in range(seq['score_start'], n - 1):
        qrow = (t + 1) % n if wrong_rows else t
        lp = torch.log_softmax(ref[0, t].double(), dim=-1)
        lq = torch.log_softmax(candidate[0, qrow].double(), dim=-1)
        target = seq['input_ids'][t + 1]
        rows.append({'sequence_id': seq['sequence_id'], 'position': t, 'candidate_position': qrow,
                     'input_id': seq['input_ids'][t], 'target_id': target,
                     'kl_ref_candidate': float((lp.exp() * (lp - lq)).sum()),
                     'reference_nll': float(-lp[target]), 'candidate_nll': float(-lq[target]),
                     'reference_top1': int(lp.argmax()), 'candidate_top1': int(lq.argmax()),
                     'top1_agreement': bool(lp.argmax() == lq.argmax())})
    return rows

def aggregate(rows):
    require(bool(rows), 'no scored rows')
    out = {'scored_rows': len(rows), 'top1_agreement': sum(r['top1_agreement'] for r in rows) / len(rows)}
    for k in ('kl_ref_candidate', 'reference_nll', 'candidate_nll'):
        x = torch.tensor([r[k] for r in rows], dtype=torch.float64)
        out[k] = {'mean': float(x.mean()), 'min': float(x.min()), 'max': float(x.max()),
                  **{f'p{p}': float(torch.quantile(x, p / 100)) for p in (50, 90, 95, 99)}}
    out['worst_kl_positions'] = sorted(rows, key=lambda r: r['kl_ref_candidate'], reverse=True)[:20]
    return out

SCHEMA = "ling3-portable-candidate-v1"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def validate_sequences(seqs, vocab):
    require(isinstance(seqs, list) and bool(seqs), "nonempty sequences required")
    seen = set()
    for seq in seqs:
        sid = seq.get("sequence_id")
        require(isinstance(sid, str) and sid and sid not in seen, "sequence identity")
        seen.add(sid)
        ids = seq.get("input_ids")
        require(isinstance(ids, list) and len(ids) >= 2 and
                all(type(i) is int and 0 <= i < vocab for i in ids), "token IDs")
        require(type(seq.get("score_start")) is int and
                0 <= seq["score_start"] < len(ids) - 1, "score_start")


def load_capture(directory):
    """Validate complete ordered tensor coverage before CPU diagnostic scoring.

    Hashes bind bytes to the supplied manifest, not to a trusted external source.
    This does not authenticate model provenance or infer correct model logits.
    """
    root = Path(directory).resolve(strict=True)
    m = read_json(root / "manifest.json")
    reference = m.get("schema") == "ling3-portable-reference-v1"
    require(reference or m.get("schema") == SCHEMA, "unsupported capture schema")
    require(m.get("status") == "complete", "capture incomplete")
    domain = m.get("logit_domain", {})
    vocab = domain.get("vocab_size")
    require(type(vocab) is int and vocab > 0, "vocabulary size")
    require(domain.get("column_ids") == "arange(0, vocab_size)" and
            domain.get("vocabulary_masking") == "none" and
            domain.get("softmax_domain") == "all_config_vocab_ids", "full unmasked column domain")
    dtype = m.get("logit_storage_dtype")
    require(dtype in DTYPES, "storage dtype")
    if reference:
        require(m["reference"]["compute_dtype"] == "bfloat16" or
                m["reference"]["kind"].startswith("deterministic synthetic"), "reference compute")
    else:
        require(m.get("policy", {}).get("compute") == "ExLlamaV3 EXL3 FP16/mixed; NOT original BF16",
                "candidate compute policy")
        require(dtype == "float16", "candidate storage")
    token_hash = m.get("token_file_sha256")
    require(isinstance(token_hash, str) and len(token_hash) == 64 and
            all(c in "0123456789abcdef" for c in token_hash), "token hash")
    seqs = [{k: s[k] for k in ("sequence_id", "input_ids", "score_start")} for s in m["sequences"]]
    validate_sequences(seqs, vocab)
    files = m["files"]
    require(isinstance(files, list) and bool(files), "file inventory")
    names, logfiles = set(), []
    for entry in files:
        name = entry["file"]
        require(isinstance(name, str) and name == Path(name).name and
                name.endswith(".safetensors") and name not in names, "safe unique capture filename")
        names.add(name)
        path = root / name
        require(not path.is_symlink() and path.is_file(), "regular capture file")
        require(sha(path) == entry["sha256"], "payload hash")
        if not reference or entry.get("stage") == "logits":
            logfiles.append(entry)
    require(names == {p.name for p in root.glob("*.safetensors")}, "tensor file closure")
    cursor, arrays = 0, {}
    for i, seq in enumerate(seqs):
        start, chunks = 0, []
        while start < len(seq["input_ids"]):
            require(cursor < len(logfiles), "missing logit chunk")
            entry = logfiles[cursor]
            data = load_file(str(root / entry["file"]), device="cpu")
            count = data["logits"].shape[1] if data["logits"].ndim == 3 else 0
            require(count > 0 and start + count <= len(seq["input_ids"]), "chunk range")
            require(entry["sequence_id"] == seq["sequence_id"] and
                    entry["file"] == f"sequence-{i:04d}-logits-{start:08d}.safetensors", "chunk order")
            if not reference:
                require(entry["start"] == start and entry["count"] == count, "candidate chunk metadata")
            validate_rows(data, seq, start, count, DTYPES[dtype], vocab)
            chunks.append(data["logits"])
            cursor += 1
            start += count
        arrays[seq["sequence_id"]] = torch.cat(chunks, 1)
    require(cursor == len(logfiles), "extra logit chunk")
    return m, seqs, arrays


def capture_candidate(model, cache, *, tokens, output, vocab, engine_commit, model_identity):
    """Caller owns engine construction, authorization, precision and source audit.

    Uses the exercised single-token rectangular-cache adapter (2048 capacity).
    Provenance arguments are declarations, not independently verified attestations.
    """
    require(isinstance(engine_commit, str) and len(engine_commit) == 40 and
            all(c in "0123456789abcdef" for c in engine_commit), "engine commit")
    require(isinstance(model_identity, str) and model_identity, "model identity")
    corpus = read_json(tokens)
    require(corpus.get("purpose") == "held-out", "held-out tokens required")
    seqs = [{k: s[k] for k in ("sequence_id", "input_ids", "score_start")} for s in corpus["sequences"]]
    validate_sequences(seqs, vocab)
    require(all(len(s["input_ids"]) <= 2048 for s in seqs), "cache capacity")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    m = {"schema": SCHEMA, "status": "running", "sequences": seqs, "files": [],
         "token_file_sha256": sha(tokens), "logit_storage_dtype": "float16",
         "logit_domain": {"vocab_size": vocab, "column_ids": "arange(0, vocab_size)",
                          "vocabulary_masking": "none", "softmax_domain": "all_config_vocab_ids"},
         "policy": {"compute": "ExLlamaV3 EXL3 FP16/mixed; NOT original BF16",
                    "cache_tokens": 2048, "actual_forward_tokens": 1},
         "declared_engine_commit": engine_commit, "declared_model_identity": model_identity}
    def persist():
        (output / "manifest.json").write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
    persist()
    try:
        for i, seq in enumerate(seqs):
            m["files"].extend(capture_sequence(model, cache, seq, i, output, vocab))
        m["status"] = "complete"
        persist()
        load_capture(output)
    except Exception:
        m["status"] = "failed"
        persist()
        raise
    return m


def score(reference, candidate, *, wrong_rows=False):
    rm, rs, ra = load_capture(reference)
    cm, cs, ca = load_capture(candidate)
    require(rs == cs and rm["token_file_sha256"] == cm["token_file_sha256"], "cross-capture token identity")
    require(rm["logit_domain"]["vocab_size"] == cm["logit_domain"]["vocab_size"], "cross-capture columns")
    rows = []
    exact = True
    for seq in rs:
        a, b = ra[seq["sequence_id"]], ca[seq["sequence_id"]]
        exact = exact and a.dtype == b.dtype and torch.equal(a, b)
        rows.extend(metrics(a, b, seq, wrong_rows=wrong_rows))
    return {"schema": "ling3-portable-score-v1", "diagnostic_only": True,
            "same_dtype_exact_logits": exact, "wrong_rows": "next-row-roll-within-sequence" if wrong_rows else None,
            "reference_manifest_sha256": sha(Path(reference) / "manifest.json"),
            "candidate_manifest_sha256": sha(Path(candidate) / "manifest.json"), **aggregate(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, help="new JSON file")
    parser.add_argument("--wrong-rows", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), "output exists")
    result = score(args.reference, args.candidate, wrong_rows=args.wrong_rows)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
