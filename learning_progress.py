"""Read-only progress for continuous training and its independent benchmarks."""
import argparse
import copy
import json
from pathlib import Path


def benchmark_activity(output, progress):
    """Read saved work without changing a benchmark's bound source or reports."""
    output = Path(output)
    binding = output/'binding.json'
    if not binding.exists():
        return {}
    languages = json.loads(binding.read_text()).get('languages', [])
    stages = [(mode, language) for mode in ('base', 'adapter') for language in languages]
    current = (progress.get('phase'), progress.get('language'))
    if current not in stages:
        return {}
    saved = {}
    for mode, language in stages:
        path = output/(mode+'-'+language+'.json')
        saved[mode, language] = json.loads(path.read_text()) if path.exists() else []
    value = {'stage': stages.index(current)+1, 'stages': len(stages),
             'overall_completed': sum(len(rows) for rows in saved.values())}
    # The base run finishes all languages before the adapter run begins.
    if current[0] == 'adapter':
        value['overall_total'] = 2*sum(len(saved['base', language]) for language in languages)
    if saved[current]:
        row = saved[current][-1]
        value['last_result'] = {'task': row['id'], 'passed': row.get('passed') is True}
    return value


def progress_view(snapshot, state, confirmation=None):
    value = copy.deepcopy(snapshot)
    if value['status'] in ('waiting', 'blocked'):
        value.update(phase='recovery', detail=state['detail'])
        return value
    if value.get('child_status') != 'completed':
        return value
    detail = state['detail']
    if state.get('auxiliary_active') and detail.startswith('Completing reserved'):
        progress = confirmation or {'phase': 'reference audit', 'mode': 'reference audit',
            'completed': 0, 'total': value.get('confirmation', {}).get('tasks', 20)}
        value.update(phase='confirmation', confirmation_progress=copy.deepcopy(progress),
            detail='Scheduled fresh reserved checks; training continues after the comparisons.')
    elif state.get('auxiliary_active') and detail.startswith('Scheduled HumanEval'):
        value.update(phase='benchmark', detail=detail)
    elif value['status'] == 'running':
        value.update(phase='prepare', detail=detail)
    return value


def main():
    from continuous_learning import Controller, DEFAULT, read
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        controller = Controller(args.session)
        snapshot = controller.snapshot()
        progress = snapshot.get('benchmark', {}).get('progress')
        if progress and controller.state.get('benchmark_output'):
            progress.update(benchmark_activity(controller.state['benchmark_output'], progress))
        confirmation = read(controller.path/'confirmations'/Path(snapshot['child']).name/'status.json')
        print(json.dumps(progress_view(snapshot, controller.state, confirmation)))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
