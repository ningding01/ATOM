from types import SimpleNamespace

import torch

from atom.plugin.sglang.attention_backend import minimax_m3_sparse as sparse


class _DecodeMode:
    def is_decode_or_idle(self):
        return True

    def is_target_verify(self):
        return False


class _ExtendMode:
    def is_decode_or_idle(self):
        return False

    def is_target_verify(self):
        return False


def _batch(mode, seq_len):
    return SimpleNamespace(
        batch_size=1,
        forward_mode=mode,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
    )


def test_decode_capture_uses_static_block_table_capacity(monkeypatch):
    monkeypatch.setattr(sparse, "_is_stream_capturing", lambda: True)
    block_table = torch.zeros((1, 8192), dtype=torch.int32)

    metadata = sparse.build_minimax_m3_forward_metadata(
        _batch(_DecodeMode(), 1), block_table, sparse.SPARSE_BLOCK_SIZE
    )

    assert metadata.max_seq_len == 8192 * sparse.SPARSE_BLOCK_SIZE


def test_decode_eager_uses_live_sequence_length(monkeypatch):
    monkeypatch.setattr(sparse, "_is_stream_capturing", lambda: False)
    block_table = torch.zeros((1, 8192), dtype=torch.int32)

    metadata = sparse.build_minimax_m3_forward_metadata(
        _batch(_DecodeMode(), 4567), block_table, sparse.SPARSE_BLOCK_SIZE
    )

    assert metadata.max_seq_len == 4567
