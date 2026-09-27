"""CPU synthetic transport and scoring tests, never native-engine qualification."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file
from ling3_repro import diagnostic as d


class Cache:
    def get_new_state(self):
        self.state = type('State', (), {'position': 0})()
        return self.state

    def release_state(self, state):
        assert state is self.state
        self.released = True


class Model:
    def forward(self, ids, params):
        state = params['recurrent_states'][0]
        assert params['batch_shape'] == (1, 2048)
        assert params['past_len'] == state.position
        state.position += 1
        # Position-distinct synthetic logits; no inference claim.
        return torch.tensor([[[float(ids.item()), -float(ids.item()), 0.5, 1.0]]], dtype=torch.float16)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokens = self.root / 'tokens.json'
        self.seq = {'sequence_id': 'synthetic', 'input_ids': [0, 1, 2, 3] * 5, 'score_start': 1}
        self.tokens.write_text(json.dumps({'purpose': 'held-out', 'sequences': [self.seq]}))
        self.a = self.root / 'a'
        self.cache = Cache()
        d.capture_candidate(Model(), self.cache, tokens=self.tokens, output=self.a,
                            vocab=4, engine_commit='a' * 40, model_identity='synthetic transport only')

    def rewrite(self, manifest):
        (self.a / 'manifest.json').write_text(json.dumps(manifest))

    def test_candidate_roundtrip_and_exact_replay(self):
        self.assertTrue(self.cache.released)
        m, seqs, arrays = d.load_capture(self.a)
        self.assertEqual(seqs, [self.seq])
        expected = torch.tensor([[[float(t), -float(t), .5, 1.] for t in self.seq['input_ids']]], dtype=torch.float16)
        self.assertTrue(torch.equal(arrays['synthetic'], expected))
        result = d.score(self.a, self.a)
        self.assertEqual(result['scored_rows'], 18)
        self.assertEqual(result['kl_ref_candidate']['mean'], 0)
        self.assertEqual(result['top1_agreement'], 1)
        self.assertTrue(result['same_dtype_exact_logits'])
        self.assertGreater(d.score(self.a, self.a, wrong_rows=True)['kl_ref_candidate']['mean'], 0.1)
        with self.assertRaises(FileExistsError):
            d.capture_candidate(Model(), Cache(), tokens=self.tokens, output=self.a,
                                vocab=4, engine_commit='a' * 40, model_identity='synthetic')

    def test_reference_schema_and_distinct_compute(self):
        m, _, _ = d.load_capture(self.a)
        m['schema'] = 'ling3-portable-reference-v1'
        m['reference'] = {'compute_dtype': 'bfloat16', 'kind': 'synthetic transport'}
        m['logit_storage_dtype'] = 'bfloat16'
        for entry in m['files']:
            path = self.a / entry['file']
            data = load_file(str(path))
            data['logits'] = data['logits'].bfloat16()
            save_file(data, str(path))
            entry.update(sha256=d.sha(path), stage='logits')
        self.rewrite(m)
        d.load_capture(self.a)
        self.assertEqual(d.score(self.a, self.a)['kl_ref_candidate']['mean'], 0)

    def test_manifest_and_payload_rejection(self):
        original, _, _ = d.load_capture(self.a)
        for mutate, message in [
            (lambda m: m.update(status='running'), 'incomplete'),
            (lambda m: m.update(schema='ling-candidate-diagnostic-r1'), 'schema'),
            (lambda m: m['logit_domain'].update(vocabulary_masking='tokenizer'), 'domain'),
            (lambda m: m['files'].reverse(), 'order'),
            (lambda m: m['files'][0].update(file='../outside.safetensors'), 'filename'),
            (lambda m: m['files'][0].update(sha256='0' * 64), 'hash'),
        ]:
            m = copy.deepcopy(original)
            mutate(m)
            self.rewrite(m)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                d.load_capture(self.a)
        self.rewrite(original)
        entry = original['files'][0]
        path = self.a / entry['file']
        data = load_file(str(path))
        data['target_ids'][0] = 3
        save_file(data, str(path))
        entry['sha256'] = d.sha(path)
        self.rewrite(original)
        with self.assertRaisesRegex(ValueError, 'identity'):
            d.load_capture(self.a)

    def test_unmapped_column_included_and_independent_kl(self):
        seq = {'sequence_id': 'x', 'input_ids': [0, 1], 'score_start': 0}
        a = torch.zeros(1, 2, 4)
        b = a.clone()
        b[0, 0, 3] = torch.log(torch.tensor(3.0))
        result = d.metrics(a, b, seq)[0]
        import math
        expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
        self.assertAlmostEqual(result['kl_ref_candidate'], expected, places=7)
        self.assertEqual(result['candidate_top1'], 3)

    def test_real_score_cli(self):
        output = self.root / 'score.json'
        command = [sys.executable, '-B', '-m', 'ling3_repro.diagnostic',
                   '--reference', str(self.a), '--candidate', str(self.a), '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())['scored_rows'], 18)
        self.assertNotEqual(subprocess.run(command, capture_output=True, timeout=60).returncode, 0)


if __name__ == '__main__':
    unittest.main()
