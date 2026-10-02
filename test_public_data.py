"""Check public task conversion without downloads or untrusted host execution."""
import json
from pathlib import Path

assert (Path(__file__).parent/'public_data.py').is_file(), 'Public coding data importer is missing'
from public_data import task_from_row

row={'id':'a'*32,'input':'Write add_one(x), returning x plus one.',
     'output':'```python\ndef add_one(x):\n    return x + 1\n```',
     'unit_tests':json.dumps(['\nassert add_one(2)==3\n','\\nassert add_one(-1)==0\\n']),
     'tests_execution_status':json.dumps(['pass','pass']),'average_test_score':'1'}
task=task_from_row(row)
assert task['files']['solution.py']!=task['reference_patch']['solution.py']
assert 'NotImplementedError' in task['files']['solution.py']
assert 'assert add_one(-1)==0' in task['checks']
assert 'VISIBLE_CHECKS_PASSED' in task['files']['test_visible.py']
assert task['editable']==['solution.py'] and 'test_visible.py' not in task['editable']
for field,value in [('average_test_score','0.9'),('tests_execution_status','["pass","fail"]'),
                    ('unit_tests','["print(123)"]'),('id','../escape'),
                    ('output','```python\nclass A: pass\n```'),('input','x'*5000)]:
    try: task_from_row({**row,field:value})
    except ValueError: pass
    else: raise AssertionError('Malformed or unsuitable public example accepted: '+field)
print('PASS: bounded Python examples, complete passing test metadata, escaped tests, and immutable tests')

import public_data
import tempfile
assert callable(getattr(public_data,'read_parquet',None)), 'Pinned parquet reader is missing'
with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'wrong.parquet'; path.write_bytes(b'untrusted bytes')
    try: public_data.read_parquet(path)
    except ValueError: pass
    else: raise AssertionError('Unpinned public dataset file was accepted')
    snapshot=Path(temporary)/'snapshot.json'; snapshot.write_text('[]')
    try: public_data.prepare(snapshot,Path(temporary)/'output')
    except ValueError: pass
    else: raise AssertionError('Malformed public snapshot was accepted')

import recursive_agent
import sys
from unittest.mock import patch
with tempfile.TemporaryDirectory() as temporary:
    registry=Path(temporary)/'tasks.json'; output=Path(temporary)/'report.json'
    registry.write_text(json.dumps([{**task,'split':'train'},{**task,'id':'dev-only','split':'dev'}]))
    result={'id':task['id'],'repository':task['repository'],'split':'train','passed':False,'calls':1,'elapsed_s':0}
    with patch.object(sys,'argv',['recursive_agent.py','--tasks',str(registry),'--split','train','--output',str(output)]), \
         patch.object(recursive_agent,'solve',return_value=result) as solve:
        recursive_agent.main()
    report=json.loads(output.read_text())
    assert report['tasks_sha256']==public_data.sha256(registry)
    assert [r['id'] for r in report['tasks']]==[task['id']]
    assert solve.call_count==1 and solve.call_args.args[0]['split']=='train'
print('PASS: custom public registry collection binds its digest and excludes development rows')
