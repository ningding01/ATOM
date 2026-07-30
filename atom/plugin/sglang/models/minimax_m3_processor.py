# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""HF processor classes live in sglang.srt.configs.minimax_vl_processor to avoid circular imports with model classes."""

import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import torch
import torchvision
from torchvision.transforms import InterpolationMode

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import round_up


class MiniMaxM3SparseForCausalLM:
    pass


class MiniMaxM3SparseForConditionalGeneration:
    pass


def get_hw_multiple_of(
    image_size: Tuple[int, int],
    multiple: int,
    max_size: Union[None, int, Tuple[int, int]] = None,
) -> Tuple[int, int]:
    w, h = image_size

    if isinstance(max_size, int):
        ratio = 1.0
        max_dim = max(w, h)
        if max_dim > max_size:
            ratio = max_size / max_dim
        new_w = round_up(round(w * ratio), multiple)
        new_h = round_up(round(h * ratio), multiple)
        return new_w, new_h

    new_w = round_up(w, multiple)
    new_h = round_up(h, multiple)

    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0

        if new_w > max_w or new_h > max_h:
            new_w_ = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h_ = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)
            new_w = new_w_
            new_h = new_h_

            new_w = (
                new_w
                if new_w % multiple == 0
                else new_w + (multiple - new_w % multiple)
            )
            new_h = (
                new_h
                if new_h % multiple == 0
                else new_h + (multiple - new_h % multiple)
            )

        assert new_w % multiple == 0 and new_h % multiple == 0
        assert new_w <= max_w and new_h <= max_h

    return new_w, new_h


def _compute_sampled_frame_indices(
    total_frames: int,
    video_fps: float,
    fps: float,
    max_frames: Optional[int] = None,
) -> List[int]:
    """Frame indices must match SFT extract_frame.py constant-mode sampling (>=1/fps apart, always keep last) or eval diverges from training."""
    if total_frames <= 0 or video_fps <= 0 or fps <= 0:
        return [0] if total_frames > 0 else []

    read_time_interval = 1.0 / fps
    eps = 1e-4

    indices: List[int] = []
    prev_kept_ts = -float("inf")
    while True:
        if not indices:
            target_frame = 0
        else:
            target_ts = prev_kept_ts + read_time_interval - eps
            target_frame = math.ceil(target_ts * video_fps)
            target_frame = max(target_frame, indices[-1] + 1)
        if target_frame >= total_frames:
            break
        indices.append(target_frame)
        prev_kept_ts = target_frame / video_fps

    last_frame_idx = total_frames - 1
    last_ts = last_frame_idx / video_fps
    if indices and indices[-1] != last_frame_idx and last_ts - prev_kept_ts > eps:
        indices.append(last_frame_idx)

    if not indices:
        indices = [0]
    if max_frames is not None and len(indices) > max_frames > 0:
        last = indices[-1]
        if max_frames == 1:
            # max_frames == 1 would divide by (max_frames - 1) == 0 below; keep only the last frame.
            indices = [last]
        else:
            step = len(indices) / (max_frames - 1)
            indices = [indices[int(i * step)] for i in range(max_frames - 1)]
            indices.append(last)
    return indices


def _normalize_video_tensor(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4 or video.shape[0] == 0:
        raise ValueError(f"expected a non-empty 4D video tensor, got {tuple(video.shape)}")

    channels_first = video.shape[1] in {1, 3, 4}
    channels_last = video.shape[-1] in {1, 3, 4}
    if channels_first and channels_last:
        raise ValueError(f"ambiguous video tensor layout: {tuple(video.shape)}")
    if channels_last:
        video = video.permute(0, 3, 1, 2)
    elif not channels_first:
        raise ValueError(f"unsupported video tensor layout: {tuple(video.shape)}")

    if video.shape[1] == 4:
        video = video[:, :3]
    return video.float()


async def get_video_tensor(
    vr,
    image_factor: int,
    max_size: Tuple[int, int],
    fps: Optional[float] = None,
    frame_max_size: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    if fps is None:
        fps = 1.0
    if frame_max_size is None:
        frame_max_size = max_size[0]
    if fps <= 0:
        raise ValueError(f"video fps must be > 0, got {fps}")

    if isinstance(vr, torch.Tensor):
        video_tchw = _normalize_video_tensor(vr)
        _, _, height, width = video_tchw.shape
        resized_width, resized_height = get_hw_multiple_of(
            (width, height), image_factor, frame_max_size
        )
        resized = torchvision.transforms.functional.resize(
            video_tchw,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
        )
        return resized, {
            "total_num_frames": resized.shape[0],
            "fps": None,
            "frames_indices": None,
        }

    total_frames = len(vr)
    video_fps = vr.avg_fps
    if video_fps <= 0 or total_frames <= 0:
        raise ValueError(
            f"Invalid video metadata: fps={video_fps}, frames={total_frames}"
        )
    indices = _compute_sampled_frame_indices(total_frames, video_fps, fps, max_frames)
    video_tchw = _normalize_video_tensor(vr.get_frames_as_tensor(indices))

    _, _, height, width = video_tchw.shape
    resized_width, resized_height = get_hw_multiple_of(
        (width, height), image_factor, frame_max_size
    )
    resized = torchvision.transforms.functional.resize(
        video_tchw,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
    )
    return resized, {
        "total_num_frames": total_frames,
        "fps": video_fps,
        "frames_indices": indices,
    }


class MiniMaxM3VLProcessor(BaseMultimodalProcessor):
    models = [MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration]

    gpu_image_decode = False

    # M3's tokenizer has no pad_token.
    tokenizer_padding = False

    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    IMAGE_START_TOKEN = "]<]start of image[>["
    IMAGE_END_TOKEN = "]<]end of image[>["

    def _patch_default_resample(self):
        for processor_name in ("image_processor", "video_processor"):
            processor = getattr(self._processor, processor_name, None)
            default_resample = getattr(processor, "resample", None)
            preprocess = getattr(processor, "_preprocess", None)
            if (
                processor is None
                or default_resample is None
                or preprocess is None
                or getattr(processor, "_atom_resample_patched", False)
            ):
                continue

            def _preprocess_with_resample(
                *args,
                _preprocess=preprocess,
                _default_resample=default_resample,
                **kwargs,
            ):
                kwargs.setdefault("resample", _default_resample)
                return _preprocess(*args, **kwargs)

            processor._preprocess = _preprocess_with_resample
            processor._atom_resample_patched = True

    @staticmethod
    def _token_id(tokenizer, token):
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None, f"token id for {token!r} not found"
        return token_id

    @property
    def spatial_merge_size(self):
        return self._processor.image_processor.merge_size

    def _video_resize_config(self):
        video_processor = self._processor.video_processor
        image_factor = video_processor.patch_size * video_processor.merge_size
        # Newer M3 video processors expose max_pixels (area) instead of max_size; derive an equivalent (max_w, max_h) cap.
        max_size = getattr(video_processor, "max_size", None)
        if max_size is None:
            max_pixels = getattr(video_processor, "max_pixels", None)
            if max_pixels is not None:
                side = int(math.isqrt(int(max_pixels)))
                side -= side % image_factor
                max_size = (side, side)
            else:
                max_size = video_processor._max_size_from_size(video_processor.size)
        assert max_size is not None, "video processor max_size is required"
        return image_factor, max_size

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self._patch_default_resample()

        tokenizer = _processor.tokenizer
        assert tokenizer is not None, "tokenizer is required"

        self.IM_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_TOKEN)
        self.VIDEO_TOKEN_ID = self._token_id(tokenizer, self.VIDEO_TOKEN)
        self.IM_START_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_START_TOKEN)
        self.IM_END_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_END_TOKEN)
        video_config = dict(self.video_config)
        self.video_fps = video_config.pop("fps", None)
        self.video_frame_max_size = video_config.pop(
            "frame_max_size", video_config.pop("max_long_side_pixel", None)
        )
        self.video_max_frames = video_config.pop("max_frames", None)
        video_config.pop("detail", None)
        self.video_config = video_config

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=re.compile(
                r"<image>|<\|image\|>|<\|image_pad\|>|\]\<\]image\[\>\["
            ),
            video_token=self.VIDEO_TOKEN,
            video_token_id=self.VIDEO_TOKEN_ID,
            video_token_regex=re.compile(r"<video>|<\|video\|>|\]\<\]video\[\>\["),
        ).build(_processor)

    @staticmethod
    def _validate_video_size(video_item) -> None:
        max_bytes = 50 * 1024 * 1024
        url = getattr(video_item, "url", video_item)
        size = None
        if isinstance(url, bytes):
            size = len(url)
        elif isinstance(url, str) and url.startswith("data:"):
            _, encoded = url.split(",", 1)
            size = (len(encoded.rstrip("=")) * 3) // 4
        elif isinstance(url, str) and url.startswith("file://"):
            size = os.path.getsize(unquote(urlparse(url).path))
        elif isinstance(url, str) and os.path.isfile(unquote(urlparse(url).path)):
            size = os.path.getsize(unquote(urlparse(url).path))
        elif isinstance(url, str) and url.startswith(("http://", "https://")):
            try:
                from sglang.srt.utils.common import get_mm_http_session

                response = get_mm_http_session().head(
                    url,
                    allow_redirects=True,
                    timeout=int(os.getenv("REQUEST_TIMEOUT", "10")),
                )
                if response.ok and response.headers.get("Content-Length"):
                    size = int(response.headers["Content-Length"])
            except Exception:
                pass
        if size is not None and size > max_bytes:
            raise ValueError(f"video exceeds 50MB limit: {size} bytes")

    @staticmethod
    def _video_item_config(video_item) -> dict:
        config = dict(getattr(video_item, "preprocess_kwargs", None) or {})
        fps = config.get("fps")
        if fps is not None and not 0.2 <= float(fps) <= 5.0:
            raise ValueError(f"video fps must be in [0.2, 5.0], got {fps}")

        max_long_side = config.get("max_long_side_pixel")
        if max_long_side is not None:
            max_long_side = int(max_long_side)
            if max_long_side <= 0 or max_long_side % 28 != 0:
                raise ValueError(
                    "max_long_side_pixel must be a positive multiple of 28, "
                    f"got {max_long_side}"
                )
            config["frame_max_size"] = max_long_side

        if "frame_max_size" not in config:
            detail = config.get("detail")
            detail_sizes = {"low": 504, "default": 1008, "high": 2016}
            if detail in detail_sizes:
                config["frame_max_size"] = detail_sizes[detail]
        return config

    async def process_mm_data_async(
        self,
        image_data: Optional[List],
        audio_data: Optional[List],
        input_text: str,
        request_obj,
        **kwargs,
    ) -> Dict:
        video_data = request_obj.video_data or []
        if len(video_data) > 20:
            raise ValueError(f"at most 20 videos are supported, got {len(video_data)}")
        for item in video_data:
            self._validate_video_size(item)
        video_configs = [self._video_item_config(item) for item in video_data]

        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )

        video_metadata = None
        if base_output.videos:
            image_factor, max_size = self._video_resize_config()
            videos_processed = []
            for video, config in zip(base_output.videos, video_configs):
                videos_processed.append(
                    await get_video_tensor(
                        video,
                        image_factor=image_factor,
                        max_size=max_size,
                        fps=config.get("fps", self.video_fps),
                        frame_max_size=config.get(
                            "frame_max_size", self.video_frame_max_size
                        ),
                        max_frames=config.get("max_frames", self.video_max_frames),
                    )
                )
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output=base_output,
            mm_tokens=self.mm_tokens,
            video_metadata=video_metadata,
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        )


def register_minimax_m3_processor() -> None:
    try:
        from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING
    except Exception:
        return

    for model_cls in MiniMaxM3VLProcessor.models:
        PROCESSOR_MAPPING[model_cls] = MiniMaxM3VLProcessor


# Backward-compatible entrypoint used by atom.plugin.sglang.models.__init__.
def register_minimax_m3_text_only_processor() -> None:
    register_minimax_m3_processor()
