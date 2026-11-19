"""DTP-OPSD training entry point.

Launch through ``scripts/train.sh``, which contains the full configuration:

    MODEL_PATH=/path/to/TimeLens-SFT ANNO=data/hi_tvg_300.jsonl VIDEO_DIR=data/videos \
        bash scripts/train.sh
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, HfArgumentParser

import dtp_opsd  # noqa: F401  (puts the vendored TimeLens package on sys.path)
from dtp_opsd.data import DISTRACTOR_SOURCES, PRIVILEGED_VIEWS, DTPOPSDDataset
from dtp_opsd.rewards import load_reward_funcs
from dtp_opsd.trainer import DTPOPSDTrainer
from training.params import DataArguments, GRPOArguments, ModelArguments
from training.train.train_utils import print_trainable_parameters, safe_save_model_for_hf_trainer

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.WARNING)
logging.getLogger("dtp_opsd").setLevel(logging.INFO)
logger = logging.getLogger("dtp_opsd.train")


@dataclass
class DTPOPSDArguments:
    # data
    annotation_path: str = field(metadata={"help": "Training annotations (JSONL)."})
    video_dir: Optional[str] = field(default=None, metadata={"help": "Prefix for relative video paths."})
    teacher_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache directory for rendered privileged videos."})

    # privileged view (Sec. 3.2)
    privileged_view: str = field(default="target_distractor", metadata={"help": f"One of {PRIVILEGED_VIEWS}."})
    distractor_source: str = field(default="annotation", metadata={"help": f"One of {DISTRACTOR_SOURCES}."})
    target_expand: float = field(default=1.5, metadata={"help": "Expansion ratio of the red target window."})

    # distillation (Eq. 3 / Appendix A.3)
    lambda_jsd: float = field(default=1.0, metadata={"help": "Global JSD coefficient (0 = pure GRPO)."})
    grpo_weight: float = field(default=1.0, metadata={"help": "GRPO coefficient."})
    jsd_temperature: float = field(default=1.0)
    jsd_alpha: float = field(default=0.5, metadata={"help": "0.5 = symmetric JSD."})
    jsd_topk: int = field(default=100, metadata={"help": "Teacher top-k tokens (+1 tail category)."})

    # adaptive controller (Eq. 4)
    adaptive_controller: bool = field(default=True, metadata={"help": "False = fixed JSD (lambda_g = 1)."})
    lambda_no_edge: float = field(default=0.0, metadata={"help": "lambda_g when r_T <= r_S."})
    lambda_dead: float = field(default=2.0, metadata={"help": "lambda_g for all-zero groups with r_T > r_S."})
    skip_zero_lambda: bool = field(default=True, metadata={"help": "Skip the teacher forward if all lambda_g = 0."})

    # teacher competence (Sec. 3.3)
    teacher_competence: str = field(default="rollout", metadata={"help": "Teacher-competence estimator."})
    teacher_rollout_n: int = field(default=0, metadata={"help": "Teacher rollouts per group (0 = G)."})
    teacher_rollout_batch: int = field(default=0, metadata={"help": "Rollouts per generate call (0 = all)."})
    teacher_max_new_tokens: int = field(default=48)


def _find_vision_tower(model):
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        for attr in ("visual", "vision_model", "vision_tower"):
            if hasattr(owner, attr):
                return getattr(owner, attr)
    raise AttributeError(f"Cannot find the vision tower of {type(model).__name__}.")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments, DTPOPSDArguments))
    model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()
    is_main = training_args.local_rank in (0, -1)

    dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    visual = _find_vision_tower(model)
    for p in visual.parameters():
        p.requires_grad = not training_args.freeze_vision_tower
    for p in visual.merger.parameters():
        p.requires_grad = not training_args.freeze_merger
    for p in list(model.model.parameters()) + list(model.lm_head.parameters()):
        p.requires_grad = not training_args.freeze_llm
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    if is_main:
        print_trainable_parameters(model, training_args=training_args)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=True)
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.bos_token_id = processor.tokenizer.bos_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.pad_token = processor.tokenizer.pad_token

    dataset = DTPOPSDDataset(
        processor=processor,
        data_args=data_args,
        annotation_path=args.annotation_path,
        video_dir=args.video_dir,
        teacher_cache_dir=args.teacher_cache_dir,
        privileged_view=args.privileged_view,
        distractor_source=args.distractor_source,
        target_expand=args.target_expand,
    )

    trainer = DTPOPSDTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset,
        reward_funcs=load_reward_funcs([n.strip() for n in training_args.reward_funcs.split(",")]),
        args=training_args,
        lambda_jsd=args.lambda_jsd,
        grpo_weight=args.grpo_weight,
        jsd_temperature=args.jsd_temperature,
        jsd_alpha=args.jsd_alpha,
        jsd_topk=args.jsd_topk,
        adaptive_controller=args.adaptive_controller,
        lambda_no_edge=args.lambda_no_edge,
        lambda_dead=args.lambda_dead,
        skip_zero_lambda=args.skip_zero_lambda,
        teacher_competence=args.teacher_competence,
        teacher_rollout_n=args.teacher_rollout_n,
        teacher_rollout_batch=args.teacher_rollout_batch,
        teacher_max_new_tokens=args.teacher_max_new_tokens,
    )
    if is_main:
        logger.info(f"DTP-OPSD config: {args}")

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
