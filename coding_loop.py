"""Continuous, step-limited Qwen3.5 trials using the existing durable worker."""
import json
from pathlib import Path

from focused_experiment import Experiment, ROOT, read
from qwen35_train import atomic_json, sha256

PHASES = ('prepare', 'collect', 'export', 'baseline', 'train', 'candidate', 'gate',
          'audit-parent', 'audit-candidate', 'finish')
SOURCES = ('coding_loop.py', 'coding_loop_stage.py', 'focused_experiment.py', 'focused_evaluation.py', 'focused_report.py',
           'qwen35_train.py', 'qwen35_server.py', 'structured_output.py', 'polyglot_data.py', 'polyglot_runtime.py',
           'code_recipe.py', 'recursive_agent.py', 'evaluation.py', 'training_data.py', 'train_adapter.py',
           'teacher_data.py', 'focused_data.py', 'repl_learning.py', 'rlm_worker.py',
           'research/polyglot_benchmark.py', 'research/adapter_eval_server.py',
           'research/codecontests-source.json', '.cache/polyglot-runtime.json')


def stages(path, cycle, method=None):
    folder = f'round-{cycle:06d}'
    phases = list(PHASES)
    if method == 'execution_repair':
        phases.insert(phases.index('train'), 'repl-baseline')
        phases.insert(phases.index('gate'), 'repl-candidate')
    return [dict(name=folder+'/'+phase,
        command=[str(ROOT/'.cache/qwen35-env/bin/python'), str(ROOT/'coding_loop_stage.py'),
                 '--session', str(path), '--round', str(cycle), '--phase', phase,
                 '--pause-file', str(path/'pause.requested')],
        result=folder+('/adapter/training.json' if phase == 'train' else '/'+phase+'.json'),
        progress=folder+('/adapter/progress.json' if phase == 'train' else
                         '/'+phase+'/progress.json' if phase.startswith('audit-') else '/progress.json'),
        gpu=phase in ('collect','export','baseline','train','candidate','audit-parent','audit-candidate','repl-baseline','repl-candidate'),
        train=phase == 'train') for phase in phases]


class CodingLoop(Experiment):
    def __init__(self, path):
        super().__init__(path)
        self.config['stages'] = stages(self.path, self.state.get('cycle', 1), self.config.get('learning_method'))

    @classmethod
    def prepare(cls, path, teacher=True):
        path = Path(path).resolve()
        if path.exists():
            raise ValueError('Choose a new loop name; existing results stay preserved')
        config = dict(schema=1, continuous=True, limit_seconds=None, steps_per_round=128,
            learning_method='execution_repair', repair_turns=3, repl_tasks=10, repl_lessons_per_language=2,
            max_epochs=1, learning_rate=5e-6, min_examples=100, min_per_language=8,
            audit_every_updates=20, fresh_tasks=40, validation_tasks=20, replay_per_language=128,
            teacher_examples=128 if teacher else 0, teacher_validation=20, teacher_learning_rate=5e-6,
            purpose='Execution feedback, verified REPL repair lessons and matched retention checks',
            inputs={str(ROOT/n): sha256(ROOT/n) for n in SOURCES}, stages=stages(path, 1, 'execution_repair'))
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
        self.config['stages'] = stages(self.path, following, self.config.get('learning_method'))
        return True

    def snapshot(self):
        result = super().snapshot()
        result.update(continuous=True, cycle=self.state.get('cycle',1), limit_seconds=None)
        folder = self.path/f"round-{result['cycle']:06d}"
        exported = read(folder/'export.json',{})
        collected = read(folder/'collect-rows.json', {}).get('rows', [])
        result.update(teaching_method=self.config.get('learning_method') or ('teacher_mix' if (folder/'teacher.json').exists() else 'self_training'),
                      teacher_examples=exported.get('teacher_rows',0), training_examples=exported.get('rows',0),
                      collected_examples=len(collected), first_passes=sum(r.get('first_passed',False) for r in collected),
                      repaired_examples=sum(bool(r.get('repair_messages')) for r in collected),
                      repl_examples=exported.get('repl_rows',0), correction_examples=exported.get('correction_rows',0),
                      training_languages=exported.get('languages',{}),
                      readiness=exported.get('reason','Collecting checked examples'),
                      accepted_updates=self.state.get('accepted',0))
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
