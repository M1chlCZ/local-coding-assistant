"""Read-only progress for continuous training and its independent benchmarks."""
import argparse
import copy
import json
from pathlib import Path


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
        confirmation = read(controller.path/'confirmations'/Path(snapshot['child']).name/'status.json')
        print(json.dumps(progress_view(snapshot, controller.state, confirmation)))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
