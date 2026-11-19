"""tIoU reward and answer parsing for temporal video grounding.

Following the thinking-free RLVR setting of TimeLens, the only reward is the
temporal Intersection-over-Union between the predicted and ground-truth
intervals (Eq. 1). No format reward or temporal-distance reward is used.
"""

import re

_VISION_TOKENS = re.compile(r"<\|(video_pad|image_pad|vision_start|vision_end)\|>")


def extract_answer(text: str) -> str:
    """Return the content of the last ``<answer>`` tag, or the full text."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1].strip() if matches else text.strip()


def extract_time(paragraph: str):
    """Parse ``(start, end)`` pairs (in seconds) from a model response."""
    paragraph = paragraph.lower()
    timestamps = []

    # hh:mm:ss / mm:ss style timestamps
    time_regex = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b")
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]
    if time_matches:
        converted = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:
                converted.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
            else:
                converted.append(int(parts[0]) * 60 + float(parts[1]))
        timestamps = [(converted[i], converted[i + 1]) for i in range(0, len(converted), 2)]

    # "<start> - <end>" / "<start> to <end>"
    if not timestamps:
        for pattern in (r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)"):
            matches = re.findall(pattern, paragraph)
            if matches:
                timestamps = [(float(s), float(e)) for s, e in matches]
                break

    # fallback: consecutive numbers
    if not timestamps:
        nums = re.findall(r"\b(\d+\.\d+|\d+)\b", paragraph)
        nums = nums[: len(nums) // 2 * 2]
        timestamps = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]

    return timestamps


def iou(a, b) -> float:
    """Temporal IoU between two intervals ``a`` and ``b``."""
    inter = max(min(a[1], b[1]) - max(a[0], b[0]), 0.0)
    union = max(max(a[1], b[1]) - min(a[0], b[0]), 1e-8)
    return inter / union


def interval_tiou(pred, gt) -> float:
    """tIoU with the validity rules of the paper (Appendix A.4).

    An unparseable interval, a negative timestamp, or an end time earlier than
    the start time receives zero reward.
    """
    if pred is None:
        return 0.0
    s, e = float(pred[0]), float(pred[1])
    if s < 0 or e < 0 or e < s:
        return 0.0
    return iou((s, e), gt)


def response_tiou(text: str, gt) -> float:
    """Parse a response and score its last predicted interval against ``gt``."""
    ts = extract_time(extract_answer(text))
    return interval_tiou(ts[-1] if ts else None, gt)


def tiou_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """GRPO reward function (TimeLens reward interface)."""
    contents = [c[0]["content"] for c in completions]
    rewards = []
    for i, text in enumerate(contents):
        gt = anno[i]["span"]
        if isinstance(gt[0], (list, tuple)):
            gt = gt[0]
        rewards.append(response_tiou(text, gt))
    return rewards


REWARD_REGISTRY = {"tiou": tiou_reward}


def load_reward_funcs(names):
    funcs = []
    for name in names:
        if name not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward '{name}'. Available: {list(REWARD_REGISTRY)}")
        funcs.append(REWARD_REGISTRY[name])
    return funcs
