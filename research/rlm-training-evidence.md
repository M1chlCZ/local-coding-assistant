# RLM training evidence

Sources reviewed on October 2, 2026. These external results do not establish coding quality on this project's GPU.

## Published methods

The [RLM paper](https://arxiv.org/html/2512.24601v3#A1) trained an 8B model on 1,000 filtered teacher trajectories.
The examples taught REPL use and decisions about recursive calls.
The authors report gains on long-context tasks. Their fine-tuning used 48 H100 hours.

The [alphaXiv experiment](https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms) used short supervised training before reinforcement learning.
Each executed turn supplied a separate training example.
Parent and child calls used one policy. Children inherited the parent's final reward.
The reported multi-paper run used eight H200 GPUs. Its task extracted evidence from documents, rather than repaired code.

## Available models and data

| Source | What it supplies | Limit for this project |
| --- | --- | --- |
| [MIT RLM-Qwen3-8B](https://huggingface.co/mit-oasys/rlm-qwen3-8b-v0.1) | Post-trained 8B weights, MIT license | Requires the original prompt and scaffold. Local coding quality remains unmeasured. |
| [alphaXiv 4B RLM](https://huggingface.co/alphaXiv/evidence-multi-rlm-4b-grpo-step100) | Evidence-extraction weights, Apache-2.0 license, updated August 11 | Includes an unused vision tower. Its document scores do not measure code repair. |
| [alphaXiv supervised traces](https://huggingface.co/datasets/alphaXiv/sft-traces-arxivqa-multi) | 384 parent and child examples | No dataset license or card appeared in the reviewed repository. These examples were not imported. |
| [HotCopy seed](https://huggingface.co/datasets/HotCopyAI/rlm-trajectories-seed) | 12 synthetic examples, Apache-2.0 license | The card identifies illustrative outputs and metrics. They are not executed repair evidence. |

## Local training and history

The current exporter runs inspect, failing-test, repair-and-retest, and submission steps for each accepted teacher patch.
These steps teach tested repairs. They contain no recursive calls.
The current collector also produced no child calls in its first revised round.
Training loss alone does not establish coding improvement or recursion.

Coding chats can supply realistic bug descriptions and test cases.
Our assessment favors independently executable tasks derived from selected chats over direct training on entire conversations.
This approach preserves outcome checks and a consistent REPL interface.
Raw chat replies and tool logs do not automatically prove that a repair works.
Chat storage can also contain duplicate events, interrupted work, private paths, and unrelated topics.

History-derived data requires a separate private dataset and review of its contents.
Repository-level separation prevents one project's repairs from entering both training and evaluation.
Personal datasets and adapters receive no automatic upload.
No chat history entered the current training session.
