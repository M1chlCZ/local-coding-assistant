"""Finite autoresearch-style search over RLM instructions and recursion depth."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation import chat, request
from recursive_agent import TASKS, UPSTREAM, grade, solve


def candidate(text, forbidden=()):
    value=json.loads(text)
    if (not isinstance(value,dict) or set(value)!={'depth','instruction'}
            or type(value['depth']) is not int or value['depth'] not in (1,2)
            or not isinstance(value['instruction'],str) or len(value['instruction'])>2000):
        raise ValueError('Proposal must contain only depth (1 or 2) and instruction (at most 2000 characters)')
    if any(name.casefold() in value['instruction'].casefold() for name in forbidden):
        raise ValueError('Proposal names a fixture; use a general instruction')
    return value


def score(report):
    rows=report['tasks']
    return (sum(r['passed'] for r in rows),-sum(r['output_tokens'] for r in rows),
            -sum(r['elapsed_s'] for r in rows))


def run(tasks,base,settings,limits,path,digest):
    report={'schema_version':1,'split':tasks[0]['split'],'tasks_sha256':digest,
            'upstream_revision':UPSTREAM,'settings':settings,'limits':limits,'tasks':[],
            'scope':'Synthetic mini-repositories; one patch attempt per task. Matched resource ceilings. Small development selection, not a general reliability benchmark.'}
    for task in tasks:
        if hashlib.sha256(TASKS.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Fixture file changed during the experiment')
        row=solve(task,base,**settings,**limits)
        if row.get('patch'):
            try:
                row.update(grade(task,row['patch']))
            except Exception as error:
                row.update(passed=False,feedback=str(error))
        report['tasks'].append(row)
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(f"{path.stem}/{task['id']}: {'PASS' if row['passed'] else 'FAIL'}; {row['calls']} calls, {row['elapsed_s']}s",flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8080')
    parser.add_argument('--trials',type=int,default=2,help='RLM trials, including the initial unchanged instruction')
    parser.add_argument('--calls',type=int,default=16)
    parser.add_argument('--output-tokens',type=int,default=8192)
    parser.add_argument('--seconds',type=int,default=300)
    parser.add_argument('--output',type=Path,required=True,help='New directory for reports and retained settings')
    parser.add_argument('--holdout',action='store_true',help='Evaluate retained settings on the final split once; no proposal sees these results')
    args=parser.parse_args()
    if not 1<=args.trials<=10 or args.output.exists():
        parser.error('Use 1-10 trials and a new output directory')
    tasks=json.loads(TASKS.read_text())
    digest=hashlib.sha256(TASKS.read_bytes()).hexdigest()
    dev=[t for t in tasks if t['split']=='dev']
    if not dev:
        parser.error('No development tasks')
    # Verify API before creating a partial experiment directory.
    properties=request(args.base,'/props')
    limits={'calls':args.calls,'output_tokens':args.output_tokens,'seconds':args.seconds}
    args.output.mkdir(parents=True)
    summary={'tasks_sha256':digest,'server_properties':properties,'selection':'Maximize passed development tasks, then minimize generated tokens, then elapsed seconds.',
             'limitations':'One run per setting on five synthetic tasks; selection can overfit development. No weight training occurs. Proposal generation is outside per-task ceilings.',
             'proposal_trace':[],'trials':[]}
    baseline=run(dev,args.base,{'mode':'baseline','depth':1,'instruction':''},limits,args.output/'baseline.json',digest)
    summary['baseline_score']=score(baseline)
    best=None
    settings={'mode':'rlm','depth':1,'instruction':''}
    for index in range(args.trials):
        if index:
            feedback=[{'task_index':i,'passed':r['passed'],'calls':r['calls'],
                       'output_tokens':r['output_tokens'],'error':r.get('error'),'feedback':r.get('feedback','')[:1500]}
                      for i,r in enumerate(best['tasks'])]
            messages=[{'role':'system','content':'Propose one small experiment for a recursive English coding agent. Return ONLY JSON with depth (1 or 2) and instruction (a general instruction under 2000 characters). Do not include task answers, source code, test-specific values, or change tests. The agent can inspect context files in Python and call llm_query or rlm_query. Optimize correct repairs, then fewer generated tokens. Instructions must generalize to unseen repositories.'},
                      {'role':'user','content':json.dumps({'retained_settings':settings,'development_results':feedback})}]
            try:
                result,elapsed=chat(args.base,messages,max_tokens=1024)
                text=result['choices'][0]['message'].get('content') or ''
                proposal=candidate(text, [t[key] for t in tasks for key in ('id','repository')])
                summary['proposal_trace'].append({'messages':messages,'response':text,'elapsed_s':elapsed,'usage':result.get('usage')})
                trial_settings={'mode':'rlm',**proposal}
            except Exception as error:
                summary['trials'].append({'index':index,'retained':False,'error':str(error)})
                (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                continue
        else:
            trial_settings=settings
        report=run(dev,args.base,trial_settings,limits,args.output/f'trial-{index}.json',digest)
        retained=best is None or score(report)>score(best)
        if retained:
            best,settings=report,trial_settings
            (args.output/'retained-settings.json').write_text(json.dumps(settings,indent=2)+'\n')
        summary['trials'].append({'index':index,'score':score(report),'retained':retained,'settings':trial_settings})
        summary['retained_score']=score(best)
        summary['improved_over_baseline']=score(best)[0]>score(baseline)[0]
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.holdout:
        held=[t for t in tasks if t['split']=='holdout']
        report=run(held,args.base,settings,limits,args.output/'holdout.json',digest)
        summary['holdout_score']=score(report)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Retained settings: '+str(args.output/'retained-settings.json'),flush=True)


if __name__=='__main__':
    main()
