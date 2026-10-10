import json
from pathlib import Path
import tempfile
import unittest
from focused_report import compare, report
from focused_evaluation import summarize


class ReportTests(unittest.TestCase):
    def test_pairs_gains_and_losses_instead_of_hiding_them_in_total(self):
        before = {'go': [{'id':'a','passed':True},{'id':'b','passed':False}]}
        after = {'go': [{'id':'a','passed':False},{'id':'b','passed':True}]}
        value=compare(before,after)
        self.assertEqual(value['go'], {'before':1,'after':1,'total':2,'gained':['b'],'lost':['a']})

    def test_unmatched_or_incomplete_audit_cannot_be_compared(self):
        with self.assertRaises(ValueError):
            compare({'go':[{'id':'a','passed':True}]}, {'go':[{'id':'b','passed':True}]})
        with self.assertRaises(ValueError):
            compare({'go':[{'id':'a','passed':True}]}, {'go':[{'id':'a','error':'timeout'}]})

    def test_report_rejects_inconsistent_totals_and_changed_training_base(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for model in ('old-student','qwen35-base','qwen35-trained'):
                for suite in ('audit','repairs'):
                    target=root/(model+'-'+suite);target.mkdir()
                    binding=dict(datasets='fixed',max_output_tokens=1024,temperature=0,thinking=False,
                        batch_size=2,prompt='fixed',container_image='fixed',sources={'eval':'fixed'},
                        engine='qwen35',model='qwen35',revision='fixed',precision='bf16',runtime='fixed',suite=suite)
                    rows={'go':[{'id':'a','passed':True}]}
                    value=summarize(binding,{'go':[{'id':'a'}]},rows)
                    (target/'summary.json').write_text(json.dumps(value))
                    (target/'go.json').write_text(json.dumps(rows['go']))
            self.assertFalse(report(root)['automatically_promoted'])
            target=root/'qwen35-trained-audit/summary.json'
            original=json.loads(target.read_text())
            target.write_text(json.dumps(dict(original,passed=999)))
            with self.assertRaisesRegex(ValueError,'Aggregate'):
                report(root)
            original['binding']['revision']='different'
            target.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'settings differ: revision'):
                report(root)


if __name__ == '__main__': unittest.main()
