"""Continuous, step-limited Qwen3.5 trials using the existing durable worker."""
import json
from pathlib import Path

from focused_experiment import Experiment, ROOT, read
from qwen35_train import atomic_json, sha256

PHASES = ('prepare', 'collect', 'export', 'baseline', 'train', 'candidate', 'gate',
          'audit-parent', 'audit-candidate', 'finish')
SOURCES = ('coding_loop.py', 'coding_loop_stage.py', 'focused_experiment.py', 'focused_evaluation.py', 'focused_report.py',
           'qwen35_train.py', 'qwen35_server.py', 'polyglot_data.py', 'polyglot_runtime.py',
           'code_recipe.py', 'recursive_agent.py', 'evaluation.py', 'training_data.py', 'train_adapter.py',
           'research/polyglot_benchmark.py', 'research/adapter_eval_server.py',
           'research/codecontests-source.json', '.cache/polyglot-runtime.json')


def stages(path, cycle):
    folder = f'round-{cycle:06d}'
    return [dict(name=folder+'/'+phase,
        command=[str(ROOT/'.cache/qwen35-env/bin/python'), str(ROOT/'coding_loop_stage.py'),
                 '--session', str(path), '--round', str(cycle), '--phase', phase,
                 '--pause-file', str(path/'pause.requested')],
        result=folder+('/adapter/training.json' if phase == 'train' else '/'+phase+'.json'),
        progress=folder+('/adapter/progress.json' if phase == 'train' else
                         '/'+phase+'/progress.json' if phase.startswith('audit-') else '/progress.json'),
        gpu=phase in ('collect','baseline','train','candidate','audit-parent','audit-candidate'),
        train=phase == 'train') for phase in PHASES]


class CodingLoop(Experiment):
    def __init__(self, path):
        super().__init__(path)
        self.config['stages'] = stages(self.path, self.state.get('cycle', 1))

    @classmethod
    def prepare(cls, path):
        path = Path(path).resolve()
        if path.exists():
            raise ValueError('Choose a new loop name; existing results stay preserved')
        config = dict(schema=1, continuous=True, limit_seconds=None, steps_per_round=10,
            learning_rate=5e-6, fresh_tasks=40, validation_tasks=20, replay_per_language=64,
            purpose='Locally verified complete-code learning with replay and retained-model checks',
            inputs={str(ROOT/n): sha256(ROOT/n) for n in SOURCES}, stages=stages(path, 1))
        loop = cls.create(path, config)
        (path/'round-000001').mkdir()
        loop.save(cycle=1, continuous=True, retained_adapter=None, accepted=0,
                  detail='Continuous learning ready. Start collects verified complete code; no time cutoff.')
        return loop

    def next_cycle(self):
        cycle = self.state.get('cycle', 1)
        folder = self.path/f'round-{cycle:06d}'
        result = read(folder/'finish.json')
        if not result or not result.get('completed'):
            raise ValueError('The round has no complete decision')
        following = cycle+1
        (self.path/f'round-{following:06d}').mkdir(exist_ok=True)
        # One atomic state update commits the decision and the next round together.
        self.save(cycle=following, stage_index=0, round=following, phase=f'round-{following:06d}/prepare',
            retained_adapter=result['retained_adapter'], accepted=self.state.get('accepted',0)+int(result['accepted']),
            last_decision=result['decision'], last_report=str((folder/'finish.json').relative_to(self.path)),
            last_report_sha256=sha256(folder/'finish.json'),
            detail='Next round: fresh tasks, verified replay, short training, matched evaluation')
        self.config['stages'] = stages(self.path, following)
        return True

    def snapshot(self):
        result = super().snapshot()
        result.update(continuous=True, cycle=self.state.get('cycle',1), limit_seconds=None)
        return result


def promotion(before, after):
    """Never promote partial results, a tie, or a gain that loses another task."""
    if not before or [r['id'] for r in before] != [r['id'] for r in after]:
        raise ValueError('Promotion needs complete matched task identities')
    if any(type(r.get('passed')) is not bool or r.get('error') for r in before+after):
        raise ValueError('Promotion needs functional results, not infrastructure errors')
    gained = [a['id'] for a,b in zip(before,after) if not a['passed'] and b['passed']]
    lost = [a['id'] for a,b in zip(before,after) if a['passed'] and not b['passed']]
    return dict(gained=gained, lost=lost, promising=bool(gained) and not lost)
