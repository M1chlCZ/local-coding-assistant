"""Matched RLM development evaluation; no fixture text enters training."""
import hashlib,json,urllib.request
from pathlib import Path
from autoresearch import run
from recursive_agent import TASKS

out=Path('reports/adapter-local-round1')
tasks=[t for t in json.loads(TASKS.read_text()) if t['split']=='dev']
settings={'mode':'rlm','depth':2,'instruction':''}
limits={'calls':8,'output_tokens':4096,'seconds':120}
summary={'schema_version':1,'scope':'Same RLM harness, unchanged prompt, NF4 base, greedy decoding, and shared task budgets; adapter disabled for baseline. Development only; holdout unused.','settings':settings,'limits':limits,'results':{}}
for mode in ('base','adapter'):
    path=out/f'rlm-dev-{mode}.json'
    assert not path.exists()
    request=urllib.request.Request('http://127.0.0.1:8090/mode',data=json.dumps({'mode':mode}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=10) as response:
        assert json.load(response)['mode']==mode
    report=run(tasks,'http://127.0.0.1:8090',settings,limits,path,hashlib.sha256(TASKS.read_bytes()).hexdigest())
    summary['results'][mode]={'passed':sum(t['passed'] for t in report['tasks']),'total':len(report['tasks']),'calls':sum(t['calls'] for t in report['tasks'])}
    (out/'rlm-dev-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['results']),flush=True)
