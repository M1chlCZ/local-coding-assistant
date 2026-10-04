"""Keep the recipe trial separate from matched evaluation and checkpoint resume."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import learning_session as learning
from train_adapter import atomic_json,parser as training_parser


class RecipeTests(unittest.TestCase):
    def test_multilingual_warm_update_uses_smaller_rate_without_changing_old_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=learning.Session(Path(tmp)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(worker.lock.close);self.addCleanup(worker.gpu_lock.close)
            worker.state.update(round=2,best_adapter='/accepted',polyglot=True)
            args=training_parser().parse_args(worker.training_command(worker.path/'round-002')[2:])
            self.assertEqual(args.learning_rate,2.5e-6)
            self.assertEqual(args.max_steps,5)
            self.assertEqual(args.max_length,4096)
            self.assertEqual(str(args.warm_start),'/accepted')
            worker.state['polyglot']=False
            args=training_parser().parse_args(worker.training_command(worker.path/'round-002')[2:])
            self.assertEqual(args.learning_rate,5e-6)

    def test_teacher_deadline_changes_without_changing_quality_checks(self):
        calls=[]
        def solve(task,base,**settings):
            calls.append(settings)
            return {'id':task['id'],'repository':task['repository'],'split':task['split'],'passed':True,'patch':{'solution.go':'verified'}}
        fake=types.SimpleNamespace(solve=solve,grade=lambda task,answer:{'passed':True})
        original=learning.LIMITS.copy()
        with tempfile.TemporaryDirectory() as tmp:
            worker=learning.Session(Path(tmp)/'session',6,Path('/teacher.gguf'),research=True)
            self.addCleanup(worker.lock.close);self.addCleanup(worker.gpu_lock.close)
            worker.state.update(round=2,polyglot=True,polyglot_baseline_complete=True)
            folder=worker.path/'round-002';folder.mkdir()
            task={'id':'fresh-go-r2','repository':'fresh-go','split':'train','prompt':'Repair solution.go',
                'language':'go','source_reference_passed':True}
            atomic_json(folder/'tasks.json',[task])
            with patch.dict('sys.modules',{'recursive_agent':fake}),patch.object(worker,'server',return_value=True),patch.object(worker,'interrupted',return_value=False):
                worker.collect(folder)
            self.assertEqual(calls[0]['seconds'],180)
            self.assertEqual(calls[0]['calls'],8)
            self.assertEqual(calls[0]['output_tokens'],4096)
            report=json.loads((folder/'teacher.json').read_text())
            self.assertEqual(report['limits']['seconds'],180)
            self.assertEqual(worker.state['passed'],1)
            self.assertEqual(worker.state['phase'],'export')
            self.assertEqual(learning.LIMITS,original)
            self.assertEqual(learning.LIMITS['seconds'],120)


if __name__=='__main__':unittest.main()
