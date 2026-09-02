"""Regression tests for latent-action codebook usage accounting."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "latent_action_model"))

from genie.metrics import codebook_usage  # noqa: E402


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return torch.device(request.param)


@pytest.mark.parametrize("num_codes", [16, 32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_codebook_usage_uses_actual_size_and_preserves_device(device, num_codes, index_dtype):
    indices = torch.tensor([0, num_codes - 1, 0], dtype=index_dtype, device=device)

    usage = codebook_usage(indices, num_codes)

    assert usage.device == device
    assert usage.dtype == torch.float32
    assert usage.item() == pytest.approx(2.0 / num_codes)


def test_codebook_usage_empty_batch_is_zero():
    usage = codebook_usage(torch.empty(0, dtype=torch.int64), 16)
    assert usage.item() == 0.0
