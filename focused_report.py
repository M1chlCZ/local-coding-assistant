"""Report the finite comparison without changing the accepted model."""
import argparse
import json
from pathlib import Path
from train_adapter import atomic_json
from training_data import sha256


def compare(before, after):
    if set(before) != set(after):
        raise ValueError('Comparison languages differ')
    output = {}
    for language, rows in before.items():
        following = after[language]
        if ([r['id'] for r in rows] != [r['id'] for r in following]
                or any(type(r.get('passed')) is not bool or r.get('error') for r in rows+following)):
            raise ValueError('Comparison requires complete matched functional results')
        output[language] = {'before': sum(r['passed'] for r in rows),
            'after': sum(r['passed'] for r in following), 'total': len(rows),
            'gained': [a['id'] for a,b in zip(rows,following) if not a['passed'] and b['passed']],
            'lost': [a['id'] for a,b in zip(rows,following) if a['passed'] and not b['passed']]}
    return output


def report(session):
    loaded, inputs = {}, {}
    for model in ('old-student', 'qwen35-base', 'qwen35-trained'):
        loaded[model] = {}
        for suite in ('audit', 'repairs'):
            folder = session/(model+'-'+suite)
            summary_path = folder/'summary.json'
            summary = json.loads(summary_path.read_text())
            if summary.get('completed') is not True or summary['binding'].get('smoke_limit'):
                raise ValueError('Every scheduled audit must finish before the final report')
            rows = {language: json.loads((folder/(language+'.json')).read_text())
                    for language in summary['languages']}
            checked = compare(rows, rows)
            if any(checked[l]['after'] != summary['languages'][l]['passed'] or
                   checked[l]['total'] != summary['languages'][l]['total'] for l in rows):
                raise ValueError('Summary differs from its functional reports')
            passed=sum(v['after'] for v in checked.values())
            total=sum(v['total'] for v in checked.values())
            macro=sum(v['after']/v['total'] for v in checked.values())/len(checked)
            if (summary['passed'] != passed or summary['total'] != total
                    or abs(summary['macro_pass_at_1']-macro)>1e-12):
                raise ValueError('Aggregate summary differs from its functional reports')
            inputs[str(summary_path.relative_to(session))] = sha256(summary_path)
            inputs.update({str((folder/(l+'.json')).relative_to(session)):sha256(folder/(l+'.json')) for l in rows})
            loaded[model][suite] = (summary,rows)
    comparisons = {}
    for name,before,after in (
            ('training_effect','qwen35-base','qwen35-trained'),
            ('new_base_vs_previous_student','old-student','qwen35-base'),
            ('trained_vs_previous_student','old-student','qwen35-trained')):
        comparisons[name] = {}
        for suite in ('audit','repairs'):
            old,new=loaded[before][suite],loaded[after][suite]
            keys=('datasets','max_output_tokens','temperature','thinking','batch_size','prompt','container_image','sources')
            if name=='training_effect':
                keys+=('engine','model','revision','precision','runtime')
            for key in keys:
                if old[0]['binding'][key] != new[0]['binding'][key]:
                    raise ValueError('Comparison settings differ: '+key)
            comparisons[name][suite] = compare(old[1],new[1])
    result = {'completed':True,'automatically_promoted':False,'inputs':inputs,
        'models':{m:{s:{k:v[0][k] for k in ('passed','total','languages','macro_pass_at_1')}
                     for s,v in suites.items()} for m,suites in loaded.items()},
        'comparisons':comparisons,
        'decision':'Keep the existing model unchanged until these paired results are reviewed.',
        'limitations':['The repository pilot is small and uses supplied buggy code excerpts, not an autonomous agent.',
            'Public function benchmarks may occur in model pretraining and do not prove general coding quality.',
            'The previous student uses NF4 and the new model uses BF16; their comparison includes deployment differences.',
            'The unused final repair split remains reserved. This is one fixed training experiment, not repeated benchmark tuning.']}
    atomic_json(session/'comparison.json',result)
    print(json.dumps({m:{s:f"{v['passed']}/{v['total']}" for s,v in suites.items()}
                      for m,suites in result['models'].items()}),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True,type=Path)
    report(parser.parse_args().session)
