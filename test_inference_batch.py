"""Bounded batches keep independent prompts, response order and task accounting."""
import ast
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class InferenceBatchTests(unittest.TestCase):
    def bridge(self):
        source=ast.parse(Path('research/adapter_eval_server.py').read_text())
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in source.body),
                        'The model bridge must be import-safe for batch validation')
        return importlib.import_module('research.adapter_eval_server')

    def test_batch_bounds_and_independent_message_order(self):
        module=self.bridge()
        conversations=[[{'role':'user','content':str(i)}] for i in range(2)]
        values,maximum=module.validate_batch({'conversations':conversations,'max_tokens':512})
        self.assertEqual(values,conversations);self.assertEqual(maximum,512)
        module.validate_batch({'conversations':conversations*8})
        for data in ({'conversations':[]},{'conversations':conversations*9},
                     {'conversations':[[{'role':'user','content':None}]]},
                     {'conversations':conversations,'max_tokens':0}):
            with self.assertRaises(ValueError):module.validate_batch(data)

    def test_long_batches_split_before_cuda_and_keep_answer_order(self):
        import contextlib
        import sys
        from types import SimpleNamespace
        module=self.bridge();calls=[]
        class Inputs(dict):
            def to(self,device):
                self.assert_safe=len(self['attention_mask'])*(5000+1024)<=32768
                if not self.assert_safe:raise AssertionError('Oversized batch reached CUDA')
                return self
        class Tokenizer:
            eos_token_id=99
            def apply_chat_template(self,conversations,**kwargs):
                return Inputs(input_ids=SimpleNamespace(shape=(len(conversations),5000),
                    values=[int(c[0]['content']) for c in conversations]),
                    attention_mask=[SimpleNamespace(sum=lambda:5000) for c in conversations])
            def decode(self,tokens,**kwargs):return str(tokens[0])
        class Model:
            generation_config=SimpleNamespace(eos_token_id=99)
            def disable_adapter(self):return contextlib.nullcontext()
            def generate(self,input_ids,**kwargs):
                calls.append(len(input_ids.values))
                return SimpleNamespace(tolist=lambda:[[0]*5000+[i,99] for i in input_ids.values])
        conversations=[[{'role':'user','content':str(i)}] for i in range(16)]
        with patch.dict(sys.modules,{'torch':SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            responses=module.generate(Model(),Tokenizer(),{'conversations':conversations},False)
        self.assertEqual(calls,[4,4,4,4])
        self.assertEqual([r['choices'][0]['message']['content'] for r in responses],list(map(str,range(16))))

    def test_completion_does_not_count_other_answers_padding(self):
        module=self.bridge()
        class Tokenizer:
            eos_token_id=9
            def decode(self,tokens,skip_special_tokens):return ','.join(str(t) for t in tokens if t!=9)
        response=module.completion(Tokenizer(),[1,2,9,9,9],7,5)
        self.assertEqual(response['choices'][0]['message']['content'],'1,2')
        self.assertEqual(response['usage'],{'prompt_tokens':7,'completion_tokens':3,'total_tokens':10})
        self.assertEqual(response['choices'][0]['finish_reason'],'stop')
        response=module.completion(Tokenizer(),[1,8,9,9],7,4,[8,9])
        self.assertEqual(response['usage']['completion_tokens'],2)

    def test_client_uses_one_request_and_validates_response_count(self):
        import evaluation
        self.assertTrue(hasattr(evaluation,'chat_batch'),'GPU batch client is missing')
        conversations=[[{'role':'user','content':str(i)}] for i in range(2)]
        with patch.object(evaluation,'request',return_value={'responses':[{'id':'first'},{'id':'second'}]}) as send:
            responses,seconds=evaluation.chat_batch('http://127.0.0.1:8090',conversations)
        self.assertEqual([r['id'] for r in responses],['first','second'])
        self.assertEqual(send.call_count,1);self.assertEqual(send.call_args.args[1],'/batch')
        self.assertEqual(send.call_args.args[2]['conversations'],conversations)
        with patch.object(evaluation,'request',return_value={'responses':[{}]}):
            with self.assertRaises(RuntimeError):evaluation.chat_batch('http://127.0.0.1:8090',conversations)

    def test_direct_checks_keep_task_identity_and_one_call_per_answer(self):
        import code_recipe
        self.assertTrue(hasattr(code_recipe,'solve_batch'),'Direct coding batches are missing')
        tasks=[{'id':str(i),'repository':'fixture-'+str(i),'split':'dev','editable':['solution.py'],
                'prompt':'Write a function.','files':{'solution.py':'def solve(): pass'}} for i in range(2)]
        responses=[{'choices':[{'message':{'content':f'def solve(): return {i}'}}],
                    'usage':{'prompt_tokens':10+i,'completion_tokens':5+i}} for i in range(2)]
        with patch('evaluation.chat_batch',return_value=(responses,2.0)) as call:
            rows=code_recipe.solve_batch(tasks,'http://127.0.0.1:8090',seconds=120)
        self.assertEqual([r['id'] for r in rows],['0','1']);self.assertEqual(call.call_count,1)
        self.assertTrue(all(r['calls']==1 and r['generation_batch_size']==2 for r in rows))
        self.assertEqual(rows[1]['patch'],{'solution.py':'def solve(): return 1'})
        self.assertEqual(rows[1]['input_tokens'],11)

    def test_completed_trial_keeps_its_archived_audit_and_new_trials_use_batches(self):
        from continuous_learning import Controller
        from train_adapter import atomic_json
        from training_data import sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);adapter=root/'adapter';adapter.mkdir();(adapter/'adapter_model.safetensors').write_text('weights')
            child=root/'child';atomic_json(child/'status.json',{'status':'completed','polyglot':True,
                'best_adapter':str(adapter),'confirmation':{'baseline_adapter':str(adapter)}})
            controller=Controller.create(root/'controller',child)
            atomic_json(controller.path/'confirmations'/child.name/'summary.json',{'completed':True,'regressions':[]})
            _,serial=controller.audit_command()
            archive=controller.path/'source-updates'/'old';(archive/'research').mkdir(parents=True)
            (archive/'research/polyglot_benchmark.py').write_text('original audit')
            sources={'research/polyglot_benchmark.py':sha256(archive/'research/polyglot_benchmark.py')}
            atomic_json(serial/'binding.json',{'adapter_sha256':sha256(adapter/'adapter_model.safetensors'),'sources':sources})
            controller.state.update(generation_batch_size=2,benchmark_output=str(serial),
                completed_source_generations={controller.state['child']:{'directory':str(archive),'benchmark_sources':sources}})
            command,output=controller.audit_command()
            self.assertEqual(output,serial,'An in-progress audit must not restart under changed settings')
            self.assertEqual(Path(command[1]),archive/'research/polyglot_benchmark.py')
            (archive/'research/polyglot_benchmark.py').write_text('changed')
            with self.assertRaises(ValueError):controller.audit_command()
            controller.state.pop('completed_source_generations')
            command,output=controller.audit_command()
            self.assertIn('--batch-size',command);self.assertEqual(command[-1],'2');self.assertNotEqual(output,serial)

    def test_throughput_counts_shared_generation_time_once(self):
        from research.polyglot_benchmark import summarize
        tasks={'python':[{'id':'a'},{'id':'b'}]}
        rows=[{'id':name,'passed':True,'elapsed_s':10,'output_tokens':20,
               'generation_batch_id':'one','generation_batch_seconds':10} for name in ('a','b')]
        result=summarize({'languages':['python'],'generation_batch_size':2},tasks,
                         {mode:{'python':rows} for mode in ('base','adapter')})
        measured=result['languages']['python']['models']['base']
        self.assertEqual(measured['generation_wall_seconds'],10)
        self.assertEqual(measured['output_tokens_per_generation_second'],4)

    def test_new_batch_size_rebaselines_without_changing_completed_work(self):
        from continuous_learning import Controller
        from train_adapter import atomic_json
        from training_data import sha256
        import learning_session as learning
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'completed';parent.mkdir();adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_model.safetensors').write_text('retained weights')
            previous={'tasks':[{'id':'old-pass','passed':True}]}
            atomic_json(parent/'status.json',{'status':'completed','active_seconds':123,'round':10,
                'recipe':'balanced-code-v1','polyglot_baseline_complete':True,'best_dev':previous,
                'best_adapter':str(adapter),'research_adapter':str(adapter),
                'sources':{n:sha256(learning.ROOT/n) for n in learning.SOURCE_FILES}})
            atomic_json(parent/'dev-base.json',previous);anchor=parent/'round-001';anchor.mkdir()
            atomic_json(anchor/'tasks.json',[{'id':'anchor','repository':'anchor','split':'train'}])
            tasks=[{'id':str(i),'repository':'problem-'+str(i),'split':'train' if i<20 else 'dev',
                    'language':('python','go','typescript','rust','dart')[i%5]} for i in range(40)]
            atomic_json(root/'next.json',tasks);completed=(parent/'status.json').read_bytes()
            controller=Controller.create(root/'controller',parent);controller.save(generation_batch_size=16)
            with patch.object(controller,'stash_adapter',side_effect=lambda path:str(path)):
                child=controller.prepare_child(root/'next.json')
            state=json.loads((child/'status.json').read_text())
            self.assertEqual(state['generation_batch_size'],16)
            self.assertFalse(state['polyglot_baseline_complete'])
            self.assertEqual(state['best_dev'],{'tasks':[]})
            self.assertEqual(json.loads((child/'previous-serial-baseline.json').read_text()),previous)
            self.assertFalse((child/'dev-base.json').exists())
            self.assertEqual(state['best_adapter'],str(adapter))
            self.assertEqual((parent/'status.json').read_bytes(),completed)
            self.assertEqual(controller.snapshot()['active_seconds'],123)


if __name__=='__main__':unittest.main()
