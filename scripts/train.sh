#!/usr/bin/env bash
# DTP-OPSD training on 8 GPUs.
#
# Batch layout: 2 completions/GPU x 8 GPUs x 4 accumulation steps = 64
# completions = 8 prompt groups x G=8 per optimizer update. With
# steps_per_generation = gradient_accumulation_steps, each rollout batch is
# used for exactly one update. One epoch over 300 examples = 37 updates.
#
# Usage:
#   MODEL_PATH=/path/to/TimeLens-SFT ANNO=data/hi_tvg_300.jsonl VIDEO_DIR=data/videos \
#       bash scripts/train.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the TimeLens-SFT checkpoint}"
ANNO="${ANNO:?set ANNO to the training annotation file (JSONL)}"
VIDEO_DIR="${VIDEO_DIR:-}"
NUM_GPUS="${NUM_GPUS:-8}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/dtp_opsd_seed${SEED}}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/cache/privileged_videos}"
MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "${OUTPUT_DIR}"

deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT}" \
  "${REPO_ROOT}/dtp_opsd/train.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --annotation_path "${ANNO}" \
  ${VIDEO_DIR:+--video_dir "${VIDEO_DIR}"} \
  --teacher_cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed "${REPO_ROOT}/configs/zero2_offload.json" \
  --bf16 True --tf32 True --gradient_checkpointing True \
  --disable_flash_attn2 False \
  --freeze_vision_tower True --freeze_merger False --freeze_llm False \
  `# video sampling` \
  --fps 2 --fps_max_frames 448 --min_tokens 64 --total_tokens 14336 \
  --min_video_len 5 --max_video_len 500 \
  `# privileged view` \
  --privileged_view target_distractor --distractor_source annotation --target_expand 1.5 \
  `# GRPO` \
  --reward_funcs tiou --num_generations 8 --temperature 1.0 --top_p 1.0 \
  --max_completion_length 512 --scale_rewards False --loss_type grpo --beta 0.0 \
  --num_iterations 1 \
  `# distillation + adaptive controller` \
  --lambda_jsd 1.0 --grpo_weight 1.0 --jsd_alpha 0.5 --jsd_temperature 1.0 --jsd_topk 100 \
  --adaptive_controller True --lambda_no_edge 0.0 --lambda_dead 2.0 \
  --teacher_competence rollout \
  `# optimization` \
  --learning_rate 1e-6 --lr_scheduler_type constant --warmup_steps 0 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --steps_per_generation 4 \
  --seed "${SEED}" \
  `# misc` \
  --logging_steps 1 --save_strategy no --report_to none \
  --dataloader_num_workers 4 --remove_unused_columns False \
  --use_liger False --use_liger_loss False --ddp_timeout 7200 \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
