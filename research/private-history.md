# Private Codex history

The exporter reads local Codex session files with Git metadata.
It keeps visible user and assistant messages.
It excludes system instructions, hidden reasoning, images, and tool output.
Duplicate sessions are excluded. Each message has a 32,000-character limit.

The exporter redacts common tokens, passwords, personal home paths, email addresses, and private IPv4 addresses.
This process cannot detect every secret. The complete export remains private.

On the computer with Codex history, run:

```sh
python3 history_data.py export --output private-data/history/latest
python3 history_data.py inspect --output private-data/history/latest
```

Copy the complete output directory to the PC through your private connection.
The inspection checks the export hash before it reads the conversations.

If the GPU is free, start the local model server.
Then run this bounded review on the PC:

```sh
python3 history_data.py review --output private-data/history/latest --limit 8
```

The review connects only to the loopback model server.
It selects recent coding excerpts from up to eight projects.
It saves English task ideas in `reviews.jsonl`.
It treats quoted conversations as untrusted data and does not execute model output.
The learning session and this server must not use the GPU at the same time.

Raw conversations and model suggestions are not verified training examples.
Each useful idea needs a standalone repair task and executable tests before adapter training.
The existing trainer accepts only checked repair trajectories.

`private-data/` is excluded by `.gitignore`.
The publication check rejects files in that directory and records with the private history marker.
This check also catches a marked export after a rename.
Keep private datasets, reports, and adapters in the same excluded directory.

Before each push, enable the hook:

```sh
git config core.hooksPath .githooks
python3 check_public_data.py --history
```

Git exclusions do not replace a review of files selected for publication.
