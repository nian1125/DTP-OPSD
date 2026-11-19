<div align="center">

# DTP-OPSD

### Reinforcement Learning with Distractor-Aware Temporal-Privilege On-Policy Self-Distillation for Temporal Video Grounding

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57-FFD21E?logo=huggingface&logoColor=black)
![TRL](https://img.shields.io/badge/TRL-0.19-blue)
![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO--2-0A66C2)

</div>

---

## Overview

Temporal Video Grounding (TVG) localizes the start and end times of an event described by a language query. It becomes particularly hard when the **same video contains repeated, similar, or opposite events**: every response sampled by GRPO may select a wrong occurrence and receive **zero tIoU**. The group-relative advantages of such an *all-zero group* are then all zero, and the hardest examples provide no policy gradient at all.

**DTP-OPSD** augments GRPO with *distractor-aware temporal-privilege on-policy self-distillation*:

- **Temporal-privilege view.** A shared-weight teacher watches a *privileged* copy of the video in which the target window is marked with **red** full-frame borders and verified same-video distractors with **blue** borders. The student only ever sees the clean video, and both use the same prompt.
- **Same-prefix distillation.** The teacher evaluates the student's own on-policy prefixes and provides dense token-level **JSD** supervision, which stays informative even when GRPO has no learning signal.
- **Adaptive controller.** Distillation is **suppressed** when the teacher does not outperform the student and **strengthened** for all-zero groups where the teacher is better.

With only **300** post-training examples, DTP-OPSD achieves the highest mIoU among the compared open-source 8B models on Charades-TimeLens, ActivityNet-TimeLens and QVHighlights-TimeLens.

<div align="center">

| Method | Charades R1@0.7 | Charades mIoU | ActivityNet R1@0.7 | ActivityNet mIoU | QVHighlights R1@0.7 | QVHighlights mIoU |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| TimeLens-SFT (init.) | 31.12 | 51.85 | 36.61 | 49.26 | 45.38 | 57.34 |
| TimeLens-8B | 35.26 | 55.28 | 40.68 | 53.23 | 55.57 | 65.59 |
| Pure GRPO | 35.15 | 55.43 | 41.85 | 53.62 | 55.61 | 65.95 |
| **DTP-OPSD** | **36.43** | **55.93** | **43.47** | **54.77** | **57.39** | **66.71** |

<sub>All models are evaluated with the corrected TimeLens annotations and the TimeLens evaluation protocol. Pure GRPO and DTP-OPSD share the same initialization and training data.</sub>

</div>

## Method at a Glance

For a rollout group $g$ with $G$ student responses sampled from the clean video:

**Same-prefix JSD.** At every position of a student response, the student (clean video) and the teacher (privileged video, stop-gradient) predict the next token on the *same student prefix*:

$$\ell^{\mathrm{JSD}}_g=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_{g,i}}\sum_{t=1}^{T_{g,i}}\mathrm{JSD}\big(p^t_{S,g,i},\ \mathrm{sg}[p^t_{T,g,i}]\big)$$

**Adaptive controller.** With $\bar r^S_g$ the mean student tIoU and $\bar r^T_g$ the mean tIoU of $G$ independent teacher rollouts on the privileged video,

$$\lambda^{\mathrm{adapt}}_g=\begin{cases}0, & \bar r^T_g\le\bar r^S_g\\ 2, & \bar r^T_g>\bar r^S_g \text{ and } r^S_{g,i}=0\ \forall i\\ 1, & \text{otherwise}\end{cases}$$

**Objective.**

$$\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{GRPO}}+\lambda_{\mathrm{JSD}}\frac{1}{|\mathcal{B}|}\sum_{g\in\mathcal{B}}\lambda^{\mathrm{adapt}}_g\,\ell^{\mathrm{JSD}}_g$$

The only reward is tIoU, advantages are mean-centered without std scaling, and the teacher rollouts are used **only** by the controller — they never enter the JSD loss.

## Repository Structure

```
DTP-OPSD/
├── dtp_opsd/
│   ├── data/dataset.py              # clean / privileged views, red-blue rendering, caching
│   ├── trainer/dtp_opsd_trainer.py  # GRPO + same-prefix JSD + adaptive controller
│   ├── rewards.py                   # tIoU reward and answer parsing
│   └── train.py                     # training entry point
├── configs/zero2_offload.json       # DeepSpeed ZeRO-2 configuration
├── scripts/train.sh                 # full training configuration (8 GPUs)
└── third_party/TimeLens/            # minimal subset of the TimeLens GRPO trainer
```

## Installation

```bash
git clone https://github.com/nian1125/DTP-OPSD.git
cd DTP-OPSD

conda create -n dtp-opsd python=3.12 -y && conda activate dtp-opsd
# install a CUDA-matched PyTorch build first, then:
pip install -r requirements.txt
```

`ffmpeg` is required to render privileged videos; `imageio-ffmpeg` ships a static binary that is used automatically.

## Data Preparation

Training annotations are stored as JSONL, one video–query pair per line. Videos come from the [TimeLens-100K](https://github.com/TencentARC/TimeLens) training split.

```json
{"video_path": "didemo/xxxx.mp4", "query": "The man in the white shirt stands up.", "span": [[21.0, 22.0]], "duration": 23.0, "distractors": [[3.0, 5.5], [14.0, 16.0]]}
```

| Field | Description |
|:--|:--|
| `video_path` | absolute path, or relative to `--video_dir` |
| `query` | natural-language query |
| `span` | ground-truth interval `[[start, end]]` in seconds |
| `duration` | video duration in seconds |
| `distractors` | verified same-video distractor intervals (blue borders) |

Privileged videos are rendered on the fly the first time an example is seen and cached under `--teacher_cache_dir`, so later epochs and runs reuse them.

## Training

```bash
MODEL_PATH=/path/to/TimeLens-SFT \
ANNO=data/hi_tvg_300.jsonl \
VIDEO_DIR=data/videos \
bash scripts/train.sh
```

The script reproduces the configuration of the paper: Qwen3-VL-8B initialized from TimeLens-SFT, $G=8$ rollouts, a global batch of 8 prompt groups (64 completions) per update, one update per rollout batch, learning rate $1\times10^{-6}$, $\lambda_{\mathrm{JSD}}=1$, $\beta=0$, videos at 2 FPS with at most 448 frames and a 14,336-token visual budget, one epoch, DeepSpeed ZeRO-2 on 8 GPUs. Set `SEED` to run multiple seeds.

### Key Arguments

| Argument | Default | Description |
|:--|:--:|:--|
| `--privileged_view` | `target_distractor` | `target_distractor` · `target_only` · `target_random_distractor` |
| `--distractor_source` | `annotation` | `annotation` (verified `distractors`) · `other_queries` (intervals of other queries on the same video) |
| `--target_expand` | `1.5` | expansion ratio of the red target window (distractors are not expanded) |
| `--lambda_jsd` | `1.0` | global JSD coefficient $\lambda_{\mathrm{JSD}}$ (`0` = Pure GRPO) |
| `--grpo_weight` | `1.0` | GRPO coefficient (`0` = JSD only) |
| `--jsd_topk` | `100` | teacher top-$k$ tokens; the remaining mass forms one tail category |
| `--adaptive_controller` | `True` | `False` = fixed JSD ($\lambda_g = 1$) |
| `--lambda_no_edge` / `--lambda_dead` | `0.0` / `2.0` | $\lambda_g$ for "teacher not better" / "teacher better & all-zero group" |
| `--teacher_rollout_n` | `0` | teacher rollouts per group for the controller (`0` = $G$) |
| `--teacher_rollout_batch` | `0` | rollouts per `generate` call, reduce to save memory (`0` = all at once) |

### Ablation Recipes

Append the following overrides to `scripts/train.sh`:

| Variant | Overrides |
|:--|:--|
| Pure GRPO | `--lambda_jsd 0` |
| Fixed JSD | `--adaptive_controller False` |
| Rescue only | `--lambda_no_edge 1 --lambda_dead 2` |
| Suppression only | `--lambda_no_edge 0 --lambda_dead 1` |
| Target only | `--privileged_view target_only` |
| Target + random distractors | `--privileged_view target_random_distractor` |
| Target expansion | `--target_expand 1.0` / `2.0` |

## Evaluation

We follow the TimeLens evaluation protocol on the corrected Charades-TimeLens, ActivityNet-TimeLens and QVHighlights-TimeLens benchmarks. Please refer to the [TimeLens repository](https://github.com/TencentARC/TimeLens) for benchmark data and evaluation scripts; the trained checkpoint in `OUTPUT_DIR` can be evaluated directly with the same prompt:

```
Please find the visual event described by the sentence '{query}', determining its starting and ending times. The format should be: 'The event happens in <start time> - <end time> seconds'.
```

## Implementation Notes

- **Memory-efficient JSD.** Both distributions are coarse-grained onto the teacher's top-100 tokens plus one tail category ($101$ classes), a lower bound of the full-vocabulary JSD that keeps the tail mass.
- **Teacher forward.** The teacher is the current policy itself (shared weights); its forward on student prefixes runs under `torch.no_grad()` in BF16, one sample at a time, and only the completion positions are projected by `lm_head`.
- **Cross-rank consistency.** Teacher competence is merged across processes per rollout group, and the decision to skip the teacher forward when every $\lambda_g=0$ is synchronized, so all ranks issue the same collective calls.

## Acknowledgements

This project builds on [TimeLens](https://github.com/TencentARC/TimeLens), [TRL](https://github.com/huggingface/trl) and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). `third_party/TimeLens` contains a subset of the TimeLens training code, with minor modifications (chunked rollout generation, Qwen3-VL `mm_token_type_ids` handling, and group-level reward statistics), and is distributed under its original license (see `third_party/TimeLens/LICENSE` and `NOTICE`).

## License

This project is released under the [Apache 2.0 License](LICENSE). Code under `third_party/TimeLens` remains under its original license.

## Citation

If you find this work useful, please consider citing our paper. The BibTeX entry will be added upon publication.
