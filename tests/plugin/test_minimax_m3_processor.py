import asyncio
from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang.models import minimax_m3_processor as m3


def test_sampled_frames_keep_last_and_respect_limit():
    indices = m3._compute_sampled_frame_indices(31, 30.0, 5.0, max_frames=4)
    assert indices[0] == 0
    assert indices[-1] == 30
    assert len(indices) == 4


def test_normalize_video_tensor_supports_tchw_and_thwc():
    tchw = torch.zeros(2, 3, 28, 56)
    thwc = torch.zeros(2, 28, 56, 3)
    assert m3._normalize_video_tensor(tchw).shape == (2, 3, 28, 56)
    assert m3._normalize_video_tensor(thwc).shape == (2, 3, 28, 56)


def test_normalize_video_tensor_rejects_invalid_or_ambiguous_layout():
    with pytest.raises(ValueError, match="4D"):
        m3._normalize_video_tensor(torch.zeros(3, 28, 56))
    with pytest.raises(ValueError, match="ambiguous"):
        m3._normalize_video_tensor(torch.zeros(2, 3, 28, 3))


def test_get_video_tensor_decodes_reader_to_aligned_tchw():
    class Reader:
        avg_fps = 30.0

        def __len__(self):
            return 31

        def get_frames_as_tensor(self, indices):
            assert indices[-1] == 30
            return torch.zeros(len(indices), 45, 79, 3, dtype=torch.uint8)

    video, metadata = asyncio.run(
        m3.get_video_tensor(
            Reader(), image_factor=28, max_size=(112, 112), fps=5.0
        )
    )
    assert video.ndim == 4
    assert video.shape[1] == 3
    assert video.shape[2] % 28 == 0
    assert video.shape[3] % 28 == 0
    assert metadata["frames_indices"][-1] == 30


def test_video_item_config_validates_and_maps_parameters():
    item = SimpleNamespace(
        preprocess_kwargs={"fps": 2, "detail": "high", "max_long_side_pixel": 1008}
    )
    config = m3.MiniMaxM3VLProcessor._video_item_config(item)
    assert config["fps"] == 2
    assert config["frame_max_size"] == 1008

    with pytest.raises(ValueError, match="fps"):
        m3.MiniMaxM3VLProcessor._video_item_config(
            SimpleNamespace(preprocess_kwargs={"fps": 100})
        )
    with pytest.raises(ValueError, match="multiple of 28"):
        m3.MiniMaxM3VLProcessor._video_item_config(
            SimpleNamespace(preprocess_kwargs={"max_long_side_pixel": 1000})
        )


def test_video_size_limit_uses_decoded_data_uri_size():
    under = "data:video/mp4;base64," + "A" * ((50 * 1024 * 1024 * 4) // 3)
    m3.MiniMaxM3VLProcessor._validate_video_size(under)

    over = "data:video/mp4;base64," + "A" * (((50 * 1024 * 1024 + 4) * 4) // 3)
    with pytest.raises(ValueError, match="50MB"):
        m3.MiniMaxM3VLProcessor._validate_video_size(over)


def test_registration_replaces_generic_processor(monkeypatch):
    mapping = {}
    module = SimpleNamespace(PROCESSOR_MAPPING=mapping)
    monkeypatch.setitem(
        __import__("sys").modules,
        "sglang.srt.managers.multimodal_processor",
        module,
    )
    m3.register_minimax_m3_processor()
    assert mapping[m3.MiniMaxM3SparseForCausalLM] is m3.MiniMaxM3VLProcessor
    assert (
        mapping[m3.MiniMaxM3SparseForConditionalGeneration]
        is m3.MiniMaxM3VLProcessor
    )
