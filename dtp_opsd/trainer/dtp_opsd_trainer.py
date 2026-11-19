"""DTP-OPSD trainer.

GRPO on the clean video plus group-adaptive, same-prefix JSD distillation
from a shared-weight teacher that observes the privileged video:

    L_total = L_GRPO + lambda_JSD * (1/|B|) * sum_g lambda_g^adapt * l_g^JSD      (Eq. 5)

with the group-level multiplier (Eq. 4)

    lambda_g^adapt = 0   if  r_T(g) <= r_S(g)
                   = 2   if  r_T(g) >  r_S(g)  and  r_S(g, i) = 0 for all i
                   = 1   otherwise,

where r_S(g) is the mean student tIoU of rollout group g and r_T(g) is the
teacher-competence estimate obtained from independent teacher rollouts on the
privileged view (Sec. 3.3). Teacher rollouts are only used by the controller;
the JSD is always evaluated on student-generated prefixes (Eq. 3).
"""

import logging
from typing import Any, Union

import torch
import torch.nn.functional as F
from trl.trainer.utils import selective_log_softmax

import dtp_opsd  # noqa: F401  (puts the vendored TimeLens package on sys.path)
from dtp_opsd.rewards import response_tiou
from training.trainer.grpo_trainer_qwenvl import QWEN_2_5_VL_KEY_WORDS, QwenvlGRPOTrainer

logger = logging.getLogger(__name__)

TEACHER_VISION_KEYS = tuple(f"teacher_{k}" for k in QWEN_2_5_VL_KEY_WORDS)
TEACHER_TENSOR_KEYS = tuple(
    f"teacher_{k}" for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
) + ("teacher_num_visuals",)
TEACHER_LIST_KEYS = ("teacher_second_per_grid_ts",)

TEACHER_COMPETENCE_ESTIMATORS = ("rollout",)


# ---------------------------------------------------------------------------
# Memory-efficient JSD (Appendix A.3)
# ---------------------------------------------------------------------------

def topk_jsd(student_logits, teacher_logits, completion_mask, temperature=1.0, alpha=0.5, topk=100):
    """Per-sample JSD between student and (detached) teacher distributions.

    Both distributions are coarse-grained onto the teacher's top-``topk`` tokens
    plus one tail category holding all remaining probability mass, giving a
    ``topk + 1``-way distribution. With ``alpha = 0.5`` this is the symmetric
    JSD with equal mixture weights. Returns a tensor of shape ``(B,)`` holding
    the masked token-average JSD of each sample.
    """
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)

    if topk is not None and topk < s_logp.size(-1):
        idx = t_logp.topk(topk, dim=-1).indices

        def _with_tail(logp):
            logp = logp.gather(-1, idx)
            log_mass = torch.logsumexp(logp, dim=-1, keepdim=True).clamp(max=-1e-7)
            return torch.cat([logp, torch.log(-torch.expm1(log_mass))], dim=-1)

        s_logp, t_logp = _with_tail(s_logp), _with_tail(t_logp)

    a = torch.tensor(alpha, dtype=s_logp.dtype, device=s_logp.device)
    m_logp = torch.logsumexp(torch.stack([s_logp + torch.log(1 - a), t_logp + torch.log(a)]), dim=0)
    kl_t = F.kl_div(m_logp, t_logp, reduction="none", log_target=True)  # KL(p_T || m)
    kl_s = F.kl_div(m_logp, s_logp, reduction="none", log_target=True)  # KL(p_S || m)
    per_token = torch.lerp(kl_s, kl_t, a).sum(-1)                        # (B, L)
    mask = completion_mask.to(per_token.dtype)
    return (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)


def _split_visual_kwargs(kwargs, start, end):
    """Slice packed Qwen-VL visual kwargs to samples ``[start, end)``."""
    out = {}
    for key, val in kwargs.items():
        if key in ("video_grid_thw", "image_grid_thw"):
            out[key] = val[start:end]
        elif key in ("pixel_values_videos", "pixel_values"):
            grid_key = "video_grid_thw" if key == "pixel_values_videos" else "image_grid_thw"
            if grid_key in kwargs:
                cum = kwargs[grid_key].prod(dim=-1).cumsum(0)
                s = 0 if start == 0 else int(cum[start - 1].item())
                out[key] = val[s:int(cum[end - 1].item())]
            else:
                out[key] = val[start:end]
        elif isinstance(val, (list, torch.Tensor)):
            out[key] = val[start:end]
        else:
            out[key] = val
    return out


def _pad_mm_token_type_ids(mm, length):
    """Right-pad (or crop) Qwen3-VL ``mm_token_type_ids`` to ``length`` (completion = text)."""
    if mm.dim() == 1:
        mm = mm.unsqueeze(0)
    pad = length - mm.size(1)
    if pad > 0:
        mm = torch.cat([mm, torch.zeros(mm.size(0), pad, dtype=mm.dtype, device=mm.device)], dim=1)
    elif pad < 0:
        mm = mm[:, :length]
    return mm


class DTPOPSDTrainer(QwenvlGRPOTrainer):
    """GRPO + adaptive temporal-privilege on-policy self-distillation."""

    _METRIC_KEYS = (
        "grpo_loss", "jsd_raw", "jsd_weighted", "teacher_student_top1_agree",
        "teacher_rollout_tiou", "ctrl_lambda_mean", "ctrl_lambda_zero_frac",
        "ctrl_lambda_dead_frac", "ctrl_skip_teacher", "adv_dead_frac",
    )

    def __init__(
        self,
        *args,
        lambda_jsd: float = 1.0,
        grpo_weight: float = 1.0,
        jsd_temperature: float = 1.0,
        jsd_alpha: float = 0.5,
        jsd_topk: int = 100,
        adaptive_controller: bool = True,
        lambda_no_edge: float = 0.0,
        lambda_dead: float = 2.0,
        skip_zero_lambda: bool = True,
        teacher_competence: str = "rollout",
        teacher_rollout_n: int = 0,
        teacher_rollout_batch: int = 0,
        teacher_max_new_tokens: int = 48,
        **kwargs,
    ):
        # The TimeLens base trainer loads a reference model with
        # AutoModelForCausalLM when beta > 0; Qwen-VL needs ImageTextToText.
        import training.trainer.grpo_trainer_qwenvl as _base
        from transformers import AutoModelForImageTextToText

        _orig = getattr(_base, "AutoModelForCausalLM", None)
        _base.AutoModelForCausalLM = AutoModelForImageTextToText
        try:
            super().__init__(*args, **kwargs)
        finally:
            if _orig is not None:
                _base.AutoModelForCausalLM = _orig

        if teacher_competence not in TEACHER_COMPETENCE_ESTIMATORS:
            self._teacher_competence_placeholder(None, None)   # fail fast

        self.lambda_jsd = float(lambda_jsd)
        self.grpo_weight = float(grpo_weight)
        self.jsd_temperature = float(jsd_temperature)
        self.jsd_alpha = float(jsd_alpha)
        self.jsd_topk = int(jsd_topk)

        # Eq. 4. The four controller variants of Table 2 are obtained with
        # (lambda_no_edge, lambda_dead) = (1, 1) fixed JSD, (1, 2) rescue only,
        # (0, 1) suppression only and (0, 2) full controller.
        self.adaptive_controller = bool(adaptive_controller)
        self.lambda_no_edge = float(lambda_no_edge)
        self.lambda_dead = float(lambda_dead)
        self.skip_zero_lambda = bool(skip_zero_lambda)

        self.teacher_competence = teacher_competence
        self.teacher_rollout_n = int(teacher_rollout_n)        # 0 -> G (num_generations)
        self.teacher_rollout_batch = int(teacher_rollout_batch)  # 0 -> all rollouts at once
        self.teacher_max_new_tokens = int(teacher_max_new_tokens)

        self._warned = set()
        self._nonfinite_steps = 0
        for mode in ("train", "eval"):
            for key in self._METRIC_KEYS:
                self._metrics[mode][key] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _tokenizer(self):
        return getattr(self.processing_class, "tokenizer", self.processing_class)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _group_key(example):
        """Rollout-group identifier: (video, query, ground-truth span)."""
        anno = example.get("anno", {}) or {}
        span = anno.get("span", [[0.0, 0.0]])
        span = span[0] if isinstance(span[0], (list, tuple)) else span
        return (anno.get("video_path", ""), anno.get("query", ""),
                round(float(span[0]), 3), round(float(span[1]), 3))

    @staticmethod
    def _repeat_visual_kwargs(kw, n):
        """Turn single-sample Qwen-VL visual kwargs into an ``n``-row batch.

        ``generate(num_return_sequences=n)`` only expands kwargs whose first
        dimension equals the batch size, which leaves ``pixel_values_videos``
        un-expanded. Rows are therefore replicated explicitly, mirroring the
        collator.
        """
        if n == 1:
            return kw
        out = {}
        for k, v in kw.items():
            if k in ("video_grid_thw", "image_grid_thw", "mm_token_type_ids"):
                out[k] = v.repeat(n, 1)
            elif k in ("pixel_values_videos", "pixel_values"):
                out[k] = v.repeat(n, *([1] * (v.dim() - 1)))
            elif k == "second_per_grid_ts":
                out[k] = list(v) * n if isinstance(v, (list, tuple)) else v
            else:
                out[k] = v
        return out

    def _warn_once(self, key, msg):
        if key not in self._warned and self.accelerator.is_main_process:
            self._warned.add(key)
            logger.warning(msg)

    # ------------------------------------------------------------------
    # Teacher competence r_T(g) (Sec. 3.3)
    # ------------------------------------------------------------------
    def _estimate_teacher_competence(self, inputs, result):
        """Return a per-example list of r_T(g), or ``None`` on failure."""
        if self.teacher_competence == "rollout":
            return self._teacher_rollout_competence(inputs)
        return self._teacher_competence_placeholder(inputs, result)

    def _teacher_competence_placeholder(self, inputs, result):
        # [PLACEHOLDER] Additional teacher-competence estimators are not part
        # of this release.
        raise NotImplementedError("This teacher-competence estimator is not included in the release.")

    def _teacher_rollout_competence(self, inputs):
        """Mean tIoU of G independent teacher rollouts on the privileged view.

        The teacher uses the current shared weights, the same prompt, sampling
        temperature and number of rollouts as the student; only the video input
        differs. Invalid responses receive zero reward. Per-group sums are
        merged across processes so that every rank sees the same r_T(g) and
        hence the same lambda_g.
        """
        device = self.accelerator.device
        tok = self._tokenizer()
        mod = self._unwrap(self.model)
        n_roll = self.teacher_rollout_n or self.num_generations

        gen_kwargs = dict(
            do_sample=True,
            temperature=self.temperature,
            top_p=getattr(self.args, "top_p", 1.0) or 1.0,
            top_k=getattr(self.args, "top_k", None),
        )
        try:
            import inspect

            if "use_model_defaults" in inspect.signature(mod.generate).parameters:
                gen_kwargs["use_model_defaults"] = False
        except Exception:
            pass

        was_training, prev_cache = mod.training, getattr(mod.config, "use_cache", False)
        mod.eval()
        mod.config.use_cache = True
        local, ok = {}, True
        try:
            with torch.no_grad():
                seen = set()
                for d in inputs:
                    gkey = self._group_key(d)
                    if gkey in seen:
                        continue
                    seen.add(gkey)
                    ids = d.get("teacher_input_ids")
                    if ids is None:
                        local[gkey] = [0.0, 1]
                        continue
                    ids = ids.unsqueeze(0).to(device)
                    mask = d.get("teacher_attention_mask")
                    mask = mask.unsqueeze(0).to(device) if mask is not None else torch.ones_like(ids)
                    kw = {}
                    for k in QWEN_2_5_VL_KEY_WORDS:
                        v = d.get(f"teacher_{k}")
                        if v is None:
                            continue
                        if k == "mm_token_type_ids" and isinstance(v, torch.Tensor) and v.dim() == 1:
                            v = v.unsqueeze(0)
                        kw[k] = v.to(device) if isinstance(v, torch.Tensor) else v

                    gt, total, count = (gkey[2], gkey[3]), 0.0, 0
                    remaining, chunk = n_roll, (self.teacher_rollout_batch or n_roll)
                    while remaining > 0:
                        nb = min(chunk, remaining)
                        out = mod.generate(
                            input_ids=ids.repeat(nb, 1), attention_mask=mask.repeat(nb, 1),
                            max_new_tokens=self.teacher_max_new_tokens,
                            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                            **self._repeat_visual_kwargs(kw, nb), **gen_kwargs,
                        )
                        for r in range(out.size(0)):
                            text = tok.decode(out[r, ids.size(1):], skip_special_tokens=True)
                            total += response_tiou(text, gt)
                            count += 1
                        del out
                        remaining -= nb
                    local[gkey] = [total, count]
        except Exception as e:
            self._warn_once("teacher_rollout", f"Teacher rollout failed ({type(e).__name__}: {e}); "
                                               f"falling back to lambda = 1 for this batch.")
            ok = False
        finally:
            mod.config.use_cache = prev_cache
            if was_training:
                mod.train()
            torch.cuda.empty_cache()

        # Every rank joins the gather, including one whose rollout failed.
        try:
            from accelerate.utils import gather_object

            shards = gather_object([local if ok else {}])
        except Exception:
            shards = [local]
        if not ok:
            return None
        merged = {}
        for shard in shards:
            for k, (s, c) in shard.items():
                acc = merged.setdefault(k, [0.0, 0])
                acc[0] += s
                acc[1] += c
        group_mean = {k: (s / c if c else 0.0) for k, (s, c) in merged.items()}
        return [group_mean.get(self._group_key(d), 0.0) for d in inputs]

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def log(self, logs, *args, **kwargs):
        mode = "train" if self.model.training else "eval"
        removed = {k: self._metrics[mode].pop(k) for k in self._METRIC_KEYS
                   if k in self._metrics[mode] and len(self._metrics[mode][k]) == 0}
        super().log(logs, *args, **kwargs)
        for k in self._METRIC_KEYS:
            self._metrics[mode].setdefault(k, removed.get(k, []))

    def _record(self, key, value):
        self._metrics["train" if self.model.training else "eval"][key].append(float(value))

    # ------------------------------------------------------------------
    # forward helpers
    # ------------------------------------------------------------------
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep,
                             batch_size=None, **forward_kwargs):
        """Backbone forward + lm_head on completion positions only (saves memory)."""
        inner = self._unwrap(model)
        if "mm_token_type_ids" in forward_kwargs:
            forward_kwargs["mm_token_type_ids"] = _pad_mm_token_type_ids(
                forward_kwargs["mm_token_type_ids"], input_ids.size(1))
        batch_size = batch_size or input_ids.size(0)
        logps = []
        for i in range(0, input_ids.size(0), batch_size):
            end = min(i + batch_size, input_ids.size(0))
            kw = forward_kwargs if (i == 0 and end == input_ids.size(0)) \
                else _split_visual_kwargs(forward_kwargs, i, end)
            hidden = inner.model(input_ids=input_ids[i:end], attention_mask=attention_mask[i:end], **kw)[0]
            logits = inner.lm_head(hidden[:, -(logits_to_keep + 1):-1, :]) / self.temperature
            del hidden
            logps.append(selective_log_softmax(logits, input_ids[i:end, -logits_to_keep:]))
        return torch.cat(logps, dim=0)

    # ------------------------------------------------------------------
    # rollout + controller
    # ------------------------------------------------------------------
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        teacher_collected = {}
        for key in (*TEACHER_VISION_KEYS, "teacher_input_ids", "teacher_attention_mask",
                    "teacher_num_visuals", "teacher_mm_token_type_ids"):
            if key in inputs[0]:
                teacher_collected[key] = [d[key] for d in inputs]

        result = super()._generate_and_score_completions(inputs)
        result.update(teacher_collected)

        if not self.adaptive_controller or self.lambda_jsd <= 0:
            return result

        # r_T(g) is computed before the rollout batch is shuffled and split, so
        # that complete groups are available; it is returned as a regular
        # tensor and aligned with the completions by the base trainer.
        r_t = self._estimate_teacher_competence(inputs, result)
        r_s = result.get("group_reward_mean")   # mean student tIoU of each group
        all_zero = result.get("group_all_zero")  # r_S(g, i) = 0 for all i
        if r_t is None or r_s is None:
            self._warn_once("ctrl", "Adaptive controller is missing statistics; using lambda = 1.")
            return result

        rep = max(1, result["advantages"].shape[0] // len(inputs))
        r_t = torch.tensor(r_t, dtype=torch.float32).repeat_interleave(rep, dim=0)
        r_s = r_s.detach().float().cpu()

        lam = torch.ones_like(r_s)                                   # Eq. 4
        no_edge = r_t <= r_s
        lam[no_edge] = self.lambda_no_edge
        if all_zero is not None:
            lam[(~no_edge) & all_zero.detach().bool().cpu()] = self.lambda_dead

        result["teacher_rollout_tiou"] = r_t
        result["lambda_adapt"] = lam
        if self.accelerator.is_main_process:
            logger.info(
                f"[controller] step={self.state.global_step} r_T={r_t.mean():.4f} r_S={r_s.mean():.4f} "
                f"lambda=0:{int((lam == self.lambda_no_edge).sum())}/{lam.numel()} "
                f"lambda=dead:{int((lam == self.lambda_dead).sum())}/{lam.numel()}"
            )
        return result

    def _prepare_inputs(self, generation_batch: dict[str, Union[torch.Tensor, Any]]):
        inputs = super()._prepare_inputs(generation_batch)
        device = inputs["prompt_ids"].device

        for key in ("teacher_rollout_tiou", "lambda_adapt", "group_all_zero"):
            if isinstance(inputs.get(key), torch.Tensor):
                inputs[key] = inputs[key].to(device)
        for key in TEACHER_TENSOR_KEYS:
            if isinstance(inputs.get(key), list):
                inputs[key] = torch.cat(inputs[key]).to(device)
        for key in TEACHER_LIST_KEYS:
            if isinstance(inputs.get(key), list) and inputs[key] and isinstance(inputs[key][0], list):
                inputs[key] = [t for ts in inputs[key] for t in ts]

        # Left-pad the teacher prompts (their lengths differ from the student's).
        if isinstance(inputs.get("teacher_input_ids"), list):
            ids_list = inputs.pop("teacher_input_ids")
            mask_list = inputs.pop("teacher_attention_mask", None) or [torch.ones_like(t) for t in ids_list]
            pad_id = getattr(self.processing_class, "pad_token_id", 0) or 0
            bsz, max_len = len(ids_list), max(t.size(0) for t in ids_list)
            ids = torch.full((bsz, max_len), pad_id, dtype=ids_list[0].dtype, device=device)
            mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
            for i, (t_ids, t_mask) in enumerate(zip(ids_list, mask_list)):
                ids[i, max_len - t_ids.size(0):] = t_ids
                mask[i, max_len - t_ids.size(0):] = t_mask
            inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"] = ids, mask

            if isinstance(inputs.get("teacher_mm_token_type_ids"), list):
                mm = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
                for i, t_mm in enumerate(inputs["teacher_mm_token_type_ids"]):
                    mm[i, max_len - t_mm.size(0):] = t_mm
                inputs["teacher_mm_token_type_ids"] = mm
        return inputs

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def _teacher_logits(self, model, inputs, completion_ids, completion_mask, logits_to_keep):
        """Teacher next-token logits on student prefixes (stop-gradient, per sample)."""
        t_kwargs = {k: inputs[f"teacher_{k}"] for k in QWEN_2_5_VL_KEY_WORDS if f"teacher_{k}" in inputs}
        t_ids = torch.cat([inputs["teacher_prompt_ids"], completion_ids], dim=1)
        t_mask = torch.cat([inputs["teacher_prompt_mask"], completion_mask], dim=1)
        if "mm_token_type_ids" in t_kwargs:
            t_kwargs["mm_token_type_ids"] = _pad_mm_token_type_ids(t_kwargs["mm_token_type_ids"], t_ids.size(1))

        bsz = t_ids.size(0)
        n_vis = inputs.get("teacher_num_visuals")
        if n_vis is not None:
            vis_end = torch.cumsum(n_vis.to(torch.long), 0).tolist()
            vis_start = [0] + vis_end[:-1]
        else:
            vis_start, vis_end = list(range(bsz)), list(range(1, bsz + 1))

        inner = self._unwrap(model)
        logits = []
        for i in range(bsz):
            torch.cuda.empty_cache()
            kw = _split_visual_kwargs(t_kwargs, vis_start[i], vis_end[i])
            if "mm_token_type_ids" in t_kwargs:
                kw["mm_token_type_ids"] = t_kwargs["mm_token_type_ids"][i:i + 1]
            with torch.no_grad():
                hidden = inner.model(input_ids=t_ids[i:i + 1], attention_mask=t_mask[i:i + 1], **kw)[0]
                logits.append(inner.lm_head(hidden[:, -(logits_to_keep + 1):-1, :]))
                del hidden
        return logits

    def _skip_teacher(self, lam, device):
        """Skip the teacher forward only if *every* rank has lambda == 0.

        The decision must be identical on all ranks; otherwise ranks that skip
        would miss collective calls issued by the others.
        """
        # Gate on global configuration only, so that every rank takes the same
        # path even if lambda is missing locally (e.g. a failed teacher rollout).
        if not (self.adaptive_controller and self.skip_zero_lambda):
            return False
        local_skip = lam is not None and float(lam.abs().sum()) == 0.0
        if self.accelerator.num_processes > 1:
            try:
                flag = torch.tensor([0.0 if local_skip else 1.0], device=device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                return flag.item() == 0.0
            except Exception:
                return False
        return local_skip

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # --- student forward on the clean video ---
        s_kwargs = {k: inputs[k] for k in inputs if k in QWEN_2_5_VL_KEY_WORDS}
        if "mm_token_type_ids" in s_kwargs:
            s_kwargs["mm_token_type_ids"] = _pad_mm_token_type_ids(s_kwargs["mm_token_type_ids"], input_ids.size(1))
        inner = self._unwrap(model)
        hidden = inner.model(input_ids=input_ids, attention_mask=attention_mask, **s_kwargs)[0]
        student_logits = inner.lm_head(hidden[:, -(logits_to_keep + 1):-1, :])
        del hidden
        per_token_logps = selective_log_softmax(student_logits / self.temperature, input_ids[:, -logits_to_keep:])
        torch.cuda.empty_cache()

        # --- privileged-view JSD on student prefixes (Eq. 3) ---
        jsd_loss = torch.zeros((), device=input_ids.device)
        teacher_top1 = None
        lam = inputs.get("lambda_adapt") if self.adaptive_controller else None
        has_teacher = "teacher_prompt_ids" in inputs
        skip = self._skip_teacher(lam, input_ids.device)
        if skip:
            self._record("ctrl_skip_teacher", 1.0)

        if has_teacher and self.lambda_jsd > 0 and not skip:
            try:
                t_logits = self._teacher_logits(model, inputs, completion_ids, completion_mask, logits_to_keep)
                per_sample = torch.cat([
                    topk_jsd(student_logits[i:i + 1], t_logits[i].detach(), completion_mask[i:i + 1],
                             temperature=self.jsd_temperature, alpha=self.jsd_alpha, topk=self.jsd_topk)
                    for i in range(len(t_logits))
                ])
                teacher_top1 = torch.cat([t.argmax(dim=-1) for t in t_logits], dim=0)
                del t_logits
                self._record("jsd_raw", per_sample.mean().item())

                if lam is not None:  # Eq. 5: (1/|B|) sum_g lambda_g * l_g^JSD
                    lam = lam.to(per_sample.device).float().reshape(-1)[: per_sample.numel()]
                    jsd_loss = (per_sample[: lam.numel()] * lam).mean()
                    self._record("ctrl_lambda_mean", lam.mean().item())
                    self._record("ctrl_lambda_zero_frac", (lam == self.lambda_no_edge).float().mean().item())
                    self._record("ctrl_lambda_dead_frac", (lam == self.lambda_dead).float().mean().item())
                else:                # fixed JSD (lambda_g = 1)
                    jsd_loss = per_sample.mean()
                self._record("jsd_weighted", jsd_loss.item())
                if isinstance(inputs.get("teacher_rollout_tiou"), torch.Tensor):
                    self._record("teacher_rollout_tiou", inputs["teacher_rollout_tiou"].float().mean().item())
            except ValueError as e:
                if "do not match" not in str(e):
                    raise
                self._warn_once("token_mismatch", f"Teacher visual token mismatch, skipping JSD: {e}")

        # --- GRPO loss (Eq. 8) ---
        advantages = inputs["advantages"]
        old_logps = inputs["old_per_token_logps"]
        old_logps = per_token_logps.detach() if old_logps is None else old_logps
        coef_1 = torch.exp(per_token_logps - old_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        adv = advantages.unsqueeze(1)
        per_token_loss = -torch.min(coef_1 * adv, coef_2 * adv)
        if self.beta != 0.0:
            ref = inputs["ref_per_token_logps"]
            per_token_loss = per_token_loss + self.beta * (
                torch.exp(ref - per_token_logps) - (ref - per_token_logps) - 1)

        if self.loss_type == "grpo":
            grpo_loss = ((per_token_loss * completion_mask).sum(-1)
                         / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            grpo_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            grpo_loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = self.grpo_weight * grpo_loss + self.lambda_jsd * jsd_loss   # Eq. 5

        # --- logging ---
        gather = lambda x: self.accelerator.gather(x.detach()).nanmean().item()  # noqa: E731
        self._record("grpo_loss", gather(grpo_loss))
        self._record("adv_dead_frac", gather((advantages.detach().abs() < 1e-6).float().mean()))
        # Collective calls must be issued identically on every rank.
        if teacher_top1 is not None:
            agree = ((student_logits.argmax(-1) == teacher_top1).float() * completion_mask).sum() \
                / completion_mask.sum().clamp(min=1.0)
        else:
            agree = torch.ones((), device=input_ids.device)
        self._record("teacher_student_top1_agree", gather(agree))

        mode = "train" if self.model.training else "eval"
        clipped = ((coef_1 < 1 - self.epsilon_low) & (adv < 0)) | ((coef_1 > 1 + self.epsilon_high) & (adv > 0))
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gather((clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)))

        # Skip (zero-gradient) a step with a non-finite loss instead of
        # propagating NaNs into the weights; stop if it keeps happening.
        if not torch.isfinite(loss):
            self._nonfinite_steps += 1
            self._warn_once("nonfinite", "Non-finite loss encountered; the step is skipped.")
            if self._nonfinite_steps >= 30:
                raise RuntimeError("Loss has been non-finite for 30 steps; aborting training.")
            return loss.nan_to_num(0.0) * 0.0
        return loss
