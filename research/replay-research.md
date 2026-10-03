# Replay research

The completed run retained an 18/25 development adapter after 309 fresh task attempts.
Both the earlier 16/25 adapter and the later 18/25 adapter passed the same 9/10 reserved confirmation tasks.
See the [confirmation comparison](confirmation.md) for complete results and the empty-batch recovery fix.

## Earlier result

The accepted adapter passed 16/25 development repairs.
Two later checkpoints passed 17/25, but each lost one previously passing repair.
The worker retained the accepted adapter and stopped after three rejected updates.
The [aggregate results](../reports/replay-research-2026-10-03/previous-session.json) include every checkpoint and its lost tasks.

The teacher passed 94/96 fresh training repairs. The session used 2 hours and 47 minutes of active work.
These results show that training works on the local GPU. They do not establish general coding reliability.
An extra pass on this small, repeated development set remains a lead for further research.

## Next experiment

The research branch starts from a 17/25 checkpoint. The accepted 16/25 adapter remains available.
Each update mixes the original successful public training examples with a fresh public training batch.
The original 123 examples became a minority in the earlier 401, 541, and 674-row datasets.
Fixed replay tests whether retaining those examples reduces forgetting.
Replay is an established research approach; this mixture remains an experiment. See [the replay paper](https://arxiv.org/abs/2603.09892).

The worker tries short updates of 5, 10, and 20 steps at `5e-6`.
The data changes between updates. These runs are not controlled comparisons of step counts.
The worker can retain a higher-scoring research branch with at most one lost task.
It promotes a candidate only after a strict score increase with no lost previously passing tasks.
Research continues after three rejected updates, within the 12-hour active limit.
Pause, resume, checkpoint checks, exclusive GPU ownership, and storage limits remain active.
Duplicate-only data and exhausted fresh data can end the session before its time limit.

Fresh tasks come from rows 1000 through 2999 of the pinned NVIDIA OpenCodeInstruct shard.
The importer reruns reference solutions and broken stubs in isolated containers.
Training excludes development tasks, previously used task identities, private history, and the original holdout.
Fresh development tasks remain reserved for a separate confirmation check.
Kimi reviewed the experiment and identified repeated selection on a small development set as a material limitation.

All prompts, datasets, and model files stay in ignored local folders.
The live Windows console shows collection, training, evaluation, retained scores, and research scores.
See [Windows controls](learning-session.md) and [public source attribution](public-data.md).
