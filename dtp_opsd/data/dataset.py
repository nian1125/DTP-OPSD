"""Dataset with aligned clean and privileged video views (Sec. 3.2).

For every example the dataset returns
  * the student input: the clean video and the grounding prompt, and
  * the teacher input: the privileged video and the *same* prompt.

The privileged video has identical content and duration. Frames inside the
target window (expanded to ``target_expand`` x its duration around its center)
receive red full-frame borders; frames inside same-video distractor windows
receive blue full-frame borders. Red takes priority on overlap, distractor
windows are not expanded, and the prompt never mentions the border colors.
Privileged videos are rendered once and cached on disk.
"""

import copy
import hashlib
import json
import os
import random
import tempfile
import traceback

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

# Shared by the clean-video student and the privileged-video teacher (Appendix A.4).
GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in "
    "<start time> - <end time> seconds'."
)

PRIVILEGED_VIEWS = (
    "target_distractor",         # red target + blue verified distractors (main setting)
    "target_only",               # red target only
    "target_random_distractor",  # red target + blue random windows (control)
)
DISTRACTOR_SOURCES = (
    "annotation",     # verified distractors stored in the `distractors` field
    "other_queries",  # target intervals of other queries on the same video (control)
)


# ---------------------------------------------------------------------------
# Vision inputs
# ---------------------------------------------------------------------------

def build_video_content(video_path, data_args):
    downsample_rate = 32
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": int(data_args.min_tokens * downsample_rate * downsample_rate),
        "total_pixels": int(data_args.total_tokens * downsample_rate * downsample_rate),
        "fps": float(data_args.fps),
    }
    if getattr(data_args, "fps_max_frames", None) is not None:
        content["max_frames"] = int(data_args.fps_max_frames)
    return content


def process_vision_inputs(messages, processor):
    """Tokenize a chat message with video through the Qwen-VL pipeline."""
    text = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]

    patch_size = 16  # Qwen3-VL default; read from the processor when available
    for attr in ("image_processor", "video_processor"):
        sub = getattr(processor, attr, None)
        value = getattr(sub, "patch_size", None) if sub is not None else None
        if value:
            patch_size = int(value)
            break

    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=patch_size,
        return_video_kwargs=True, return_video_metadata=True,
    )

    videos_list, video_metadatas, is_timelens = None, None, False
    if videos is not None:
        # The TimeLens processor interleaves timestamps and expects an even
        # number of frames; trim the tensor and its frame indices together.
        def _trim_even(tensor, meta):
            n = int(tensor.shape[0]) if hasattr(tensor, "shape") else None
            if n is not None and n % 2 == 1:
                tensor = tensor[:-1]
                if isinstance(meta, dict) and len(meta.get("frames_indices", [])) == n:
                    meta = {**meta, "frames_indices": meta["frames_indices"][:-1]}
            return tensor, meta

        trimmed = [_trim_even(v[0], v[1]) for v in videos]
        videos_list = [t for t, _ in trimmed]
        video_metadatas = [m for _, m in trimmed]
        is_timelens = "timelens" in (type(processor).__name__ + type(processor).__module__).lower()

    if is_timelens:
        inputs = processor(
            text=text, images=images, videos=list(zip(videos_list, video_metadatas)),
            return_tensors="pt", do_resize=False, **video_kwargs,
        )
    else:
        inputs = processor(
            text=text, images=images, videos=videos_list, video_metadata=video_metadatas,
            return_tensors="pt", do_resize=False, **video_kwargs,
        )
    return inputs, text[0]


# ---------------------------------------------------------------------------
# Privileged-view rendering
# ---------------------------------------------------------------------------

def _add_border(frame: Image.Image, color: str, width: int = 8) -> Image.Image:
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    for i in range(width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    return frame


def _read_video_frames(video_path):
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    return [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))], vr.get_avg_fps()


def _write_video(frames, fps, output_path):
    # macro_block_size=1 keeps the original resolution; yuv420p needs even sizes.
    def _to_even(arr):
        h, w = arr.shape[:2]
        return arr[: h - (h % 2), : w - (w % 2)]

    writer = imageio.get_writer(
        output_path, format="FFMPEG", fps=float(fps), codec="libx264", macro_block_size=1,
        output_params=["-preset", "ultrafast", "-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        writer.append_data(_to_even(np.asarray(frame)))
    writer.close()


def _clip_window(lo, hi, duration):
    lo = max(0.0, lo)
    if duration and duration > 0:
        hi = min(float(duration), hi)
    return lo, hi


def expand_window(start, end, ratio, duration=None):
    """Expand ``[start, end]`` to ``ratio`` x its duration around its center."""
    center, half = (start + end) / 2.0, (end - start) / 2.0 * ratio
    return _clip_window(center - half, center + half, duration)


def _random_windows(video_path, target_window, distractors, duration, n_default=1):
    """Deterministic random windows that do not overlap the target window."""
    seed = int(hashlib.md5(f"{video_path}_{target_window}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    count = len(distractors) if distractors else n_default
    if distractors:
        win = sum(e - s for s, e in distractors) / len(distractors)
    else:
        win = target_window[1] - target_window[0]
    win = max(win, 2.0)
    windows = []
    for _ in range(count):
        for _attempt in range(20):
            max_start = max(0.0, duration - win)
            if max_start <= 0:
                break
            s = rng.uniform(0.0, max_start)
            if s + win < target_window[0] or s > target_window[1]:
                windows.append((s, s + win))
                break
    return windows


def render_privileged_video(video_path, target, distractors, cache_dir,
                            view="target_distractor", target_expand=1.5, duration=None):
    """Render (or load from cache) the privileged view of ``video_path``."""
    if view not in PRIVILEGED_VIEWS:
        raise ValueError(f"Unknown privileged view '{view}'. Choose from {PRIVILEGED_VIEWS}.")
    distractors = sorted([list(map(float, d)) for d in (distractors or [])])
    key = hashlib.md5(
        f"{video_path}_{target[0]:.2f}_{target[1]:.2f}_{view}_{distractors}_{target_expand:.2f}".encode()
    ).hexdigest()[:16]
    cached = os.path.join(cache_dir, f"priv_{key}.mp4")
    if os.path.exists(cached):
        return cached

    frames, fps = _read_video_frames(video_path)
    dur = float(duration) if duration else (len(frames) / fps if fps > 0 else 0.0)

    red = expand_window(target[0], target[1], target_expand, dur)
    if view == "target_distractor":
        blue = [_clip_window(s, e, dur) for s, e in distractors]
    elif view == "target_random_distractor":
        blue = _random_windows(video_path, red, distractors, dur)
    else:
        blue = []

    marked = []
    for i, frame in enumerate(frames):
        ts = i / fps if fps > 0 else 0.0
        if red[0] <= ts <= red[1]:                      # red has priority
            marked.append(_add_border(frame, "red"))
        elif any(lo <= ts <= hi for lo, hi in blue):
            marked.append(_add_border(frame, "blue"))
        else:
            marked.append(frame)

    os.makedirs(cache_dir, exist_ok=True)
    tmp = cached.replace(".mp4", f".tmp{os.getpid()}.mp4")
    _write_video(marked, fps, tmp)
    os.replace(tmp, cached)
    return cached


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def normalize_spans(span):
    if isinstance(span, tuple):
        return [list(span)]
    if isinstance(span, list) and span and isinstance(span[0], (list, tuple)):
        return [list(s) for s in span]
    if isinstance(span, list) and len(span) == 2 and isinstance(span[0], (int, float)):
        return [span]
    raise ValueError(f"Unsupported span format: {span}")


def load_annotations(path):
    """Load a JSONL file (one example per line) or a JSON list."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DTPOPSDDataset(Dataset):
    """Returns student (clean) and teacher (privileged) inputs for each example.

    Annotation fields: ``video_path``, ``query``, ``span`` ([[start, end]]),
    ``duration`` and, for the main setting, ``distractors`` ([[start, end], ...]).
    """

    def __init__(self, processor, data_args, annotation_path, video_dir=None,
                 teacher_cache_dir=None, privileged_view="target_distractor",
                 distractor_source="annotation", target_expand=1.5):
        super().__init__()
        if distractor_source not in DISTRACTOR_SOURCES:
            raise ValueError(f"Unknown distractor source '{distractor_source}'. "
                             f"Choose from {DISTRACTOR_SOURCES}.")
        self.processor = processor
        self.data_args = data_args
        self.video_dir = video_dir
        self.privileged_view = privileged_view
        self.distractor_source = distractor_source
        self.target_expand = float(target_expand)
        self.cache_dir = teacher_cache_dir or tempfile.mkdtemp(prefix="dtp_opsd_priv_")
        self._num_teacher_fallback = 0

        self.annos = self._filter(load_annotations(annotation_path))

        # Intervals of all queries per video (used by distractor_source="other_queries").
        self._video_spans = {}
        for a in self.annos:
            for s, e in a["span"]:
                self._video_spans.setdefault(a["video_path"], set()).add((round(s, 2), round(e, 2)))

        with_distractors = sum(1 for a in self.annos if self._distractors(a))
        print(f"[DTPOPSDDataset] {len(self.annos)} examples | view={privileged_view} | "
              f"distractor_source={distractor_source} | target_expand={self.target_expand} | "
              f"examples with distractors: {with_distractors}/{len(self.annos)}")

    def _filter(self, annos):
        kept, n_dur, n_span, n_path = [], 0, 0, 0
        min_len = getattr(self.data_args, "min_video_len", -1)
        max_len = getattr(self.data_args, "max_video_len", -1)
        for a in annos:
            a = dict(a)
            duration = float(a.get("duration", float("inf")))
            if (min_len >= 0 and duration < min_len) or (max_len >= 0 and duration > max_len):
                n_dur += 1
                continue
            try:
                a["span"] = normalize_spans(a["span"])
            except (ValueError, KeyError):
                n_span += 1
                continue
            vp = a.get("video_path", "")
            if self.video_dir and vp and not os.path.isabs(vp):
                vp = os.path.join(self.video_dir, vp)
            a["video_path"] = vp
            if not os.path.exists(vp):
                n_path += 1
                continue
            kept.append(a)
        print(f"[DTPOPSDDataset] filtered: duration={n_dur}, span={n_span}, "
              f"missing_video={n_path}, kept={len(kept)}")
        return kept

    def _distractors(self, anno):
        if self.distractor_source == "annotation":
            return [list(map(float, d)) for d in anno.get("distractors", []) or []]
        current = tuple(round(x, 2) for x in anno["span"][0])
        spans = self._video_spans.get(anno["video_path"], set())
        return sorted([list(s) for s in spans if s != current])

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annos[idx])
        gt_start, gt_end = anno["span"][0]
        prompt = {"type": "text", "text": GROUNDING_PROMPT.format(anno["query"])}

        # --- student: clean video ---
        student_messages = [{
            "role": "user",
            "content": [build_video_content(anno["video_path"], self.data_args), prompt],
        }]
        student_inputs, prompt_text = process_vision_inputs(student_messages, self.processor)
        student_inputs["input_ids"] = student_inputs["input_ids"][0]
        if "mm_token_type_ids" in student_inputs and student_inputs["mm_token_type_ids"].dim() > 1:
            student_inputs["mm_token_type_ids"] = student_inputs["mm_token_type_ids"][0]

        # --- teacher: privileged video, same prompt ---
        teacher_failed = False
        try:
            priv_path = render_privileged_video(
                anno["video_path"], (gt_start, gt_end), self._distractors(anno), self.cache_dir,
                view=self.privileged_view, target_expand=self.target_expand,
                duration=anno.get("duration"),
            )
            teacher_messages = [{
                "role": "user",
                "content": [build_video_content(priv_path, self.data_args), prompt],
            }]
            teacher_inputs, _ = process_vision_inputs(teacher_messages, self.processor)
        except Exception as e:  # fall back to the clean view (JSD becomes ~0 for this example)
            self._num_teacher_fallback += 1
            if self._num_teacher_fallback == 1:
                print(f"[DTPOPSDDataset] WARNING: failed to build the privileged view for "
                      f"{anno['video_path']}; falling back to the clean view. Reason: {e}")
                traceback.print_exc()
            teacher_inputs, teacher_failed = student_inputs, True

        result = dict(student_inputs)
        result["prompt"] = student_messages
        result["prompt_text"] = prompt_text
        result["anno"] = anno
        result["teacher_failed"] = teacher_failed
        for key in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
                    "pixel_values", "image_grid_thw"):
            if key in teacher_inputs:
                result[f"teacher_{key}"] = teacher_inputs[key]
        if "mm_token_type_ids" in teacher_inputs:
            tmm = teacher_inputs["mm_token_type_ids"]
            result["teacher_mm_token_type_ids"] = tmm[0] if tmm.dim() > 1 else tmm

        n_visual = 0
        if "video_grid_thw" in teacher_inputs:
            n_visual = int(teacher_inputs["video_grid_thw"].shape[0])
        elif "image_grid_thw" in teacher_inputs:
            n_visual = int(teacher_inputs["image_grid_thw"].shape[0])
        result["teacher_num_visuals"] = torch.tensor([max(n_visual, 1)], dtype=torch.long)

        # The teacher prompt length can differ from the student's (different
        # number of visual tokens after re-encoding), so keep its own ids.
        result["teacher_input_ids"] = teacher_inputs["input_ids"][0]
        result["teacher_attention_mask"] = teacher_inputs["attention_mask"][0]
        return result
