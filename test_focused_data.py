import copy
import json
from pathlib import Path
import tempfile
import unittest

import focused_data as data


def sample(repo='example/tool', language='go', identity='example__tool-1'):
    extension = 'go' if language == 'go' else 'ts'
    return {'instance_id': identity, 'repo': repo, 'language': language, 'license': 'MIT',
        'base_commit': 'a'*40, 'problem_statement': 'The counter must increment instead of decrementing.',
        'patch': f'diff --git a/count.{extension} b/count.{extension}\nindex abc..def 100644\n--- a/count.{extension}\n+++ b/count.{extension}\n@@ -1,3 +1,3 @@\n func count() {{\n- return x - 1\n+ return x + 1\n }}\n',
        'test_patch':'diff --git a/test b/test\n+assert count(3) == 4\n',
        'FAIL_TO_PASS':['counter'], 'PASS_TO_PASS':['other'],
        'image_name':'docker.io/swerebenchv2/example-tool:1-aaaaaaa',
        'install_config': {'test_cmd':'go test ./...','install':[]},
        'meta': {'llm_metadata':{'code':'A','detected_issues':{'B1':False}}}}


class FocusedDataTests(unittest.TestCase):
    def test_solution_is_not_in_user_prompt_and_provenance_is_explicit(self):
        row=data.convert(sample(), set())
        self.assertNotIn('return x + 1',row['messages'][1]['content'])
        self.assertIn('return x - 1',row['messages'][1]['content'])
        self.assertIn('return x + 1',row['messages'][2]['content'])
        self.assertEqual(row['verification']['kind'],'publisher_execution')
        self.assertFalse(row['verification']['locally_replayed'])
        self.assertEqual(len(row['provenance']['source_row_sha256']),64)

    def test_language_fidelity_and_missing_evidence_fail_closed(self):
        for key,value in [('language','javascript'),('FAIL_TO_PASS',[]),('license','custom-check-github')]:
            row=sample();row[key]=value
            with self.assertRaises(ValueError): data.convert(row,set())
        row=sample();row['meta']['llm_metadata']['code']='B2'
        with self.assertRaises(ValueError):data.convert(row,set())

    def test_patch_parser_rejects_traversal_rename_binary_and_test_edits(self):
        for replacement in ['../evil.go','foo_test.go','test/foo.go']:
            row=sample();row['patch']=row['patch'].replace('count.go',replacement)
            with self.assertRaises(ValueError):data.convert(row,set())
        row=sample();row['patch']+='Binary files differ\n'
        with self.assertRaises(ValueError):data.convert(row,set())

    def test_benchmark_overlap_is_excluded(self):
        row=sample();row['problem_statement']=' '.join('word'+str(i) for i in range(20))
        with self.assertRaises(ValueError):data.convert(row,data.shingles(row['problem_statement']))
        row=sample();row['repo']='openai/human-eval'
        with self.assertRaises(ValueError):data.convert(row,set())

    def test_forks_and_case_variants_share_split(self):
        self.assertEqual(data.family('Owner/Tool'),data.family('fork/tool'))
        self.assertEqual(data.split_for('Owner/Tool'),data.split_for('fork/tool'))

    def test_selection_deduplicates_patch_and_caps_repository(self):
        originals=[]
        for n in range(5):
            row=sample(identity=f'example__tool-{n}')
            if n:row['patch']=row['patch'].replace('x + 1',f'x + {n+1}')
            originals.append(data.convert(row,set()))
        originals.append(copy.deepcopy(originals[0]))
        chosen=data.select(originals,per_language=10,per_repository=2)
        self.assertEqual(len(chosen),2)
        self.assertEqual(len({r['task_id'] for r in chosen}),2)

    def test_bundle_is_atomic_digest_bound_and_has_separate_splits(self):
        rows=[data.convert(sample(),set()),data.convert(sample('other/project','ts','other__project-3'),set())]
        with tempfile.TemporaryDirectory() as folder:
            data.write_bundle(Path(folder),rows,{'test_source':'pinned'})
            for split in ('train','dev','test'):
                p=Path(folder)/(split+'.jsonl');manifest=json.loads(Path(str(p)+'.manifest.json').read_text())
                self.assertEqual(manifest['dataset_sha256'],data.file_sha256(p))
                for line in p.read_text().splitlines():self.assertEqual(json.loads(line)['split'],split)
            data.write_bundle(Path(folder),rows,{'test_source':'pinned'})
            p=Path(folder)/'train.jsonl';p.write_text('corrupt')
            with self.assertRaises(ValueError):data.write_bundle(Path(folder),rows,{'test_source':'pinned'})

if __name__=='__main__':unittest.main()
