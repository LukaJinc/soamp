from unittest.mock import patch

import pytest
import torch

from soamp.utils.device import DeviceError, resolve_device


def test_auto_returns_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == torch.device("cuda")


def test_auto_falls_back_to_cpu_without_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("auto") == torch.device("cpu")


def test_default_spec_is_auto():
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device() == torch.device("cpu")


def test_explicit_cpu_is_honoured_even_with_cuda_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_cuda_index_is_passed_through():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("cuda:1") == torch.device("cuda", 1)


def test_explicit_cuda_without_cuda_raises_rather_than_silently_using_cpu():
    """A run asked for a GPU should fail loudly, not quietly become a CPU run
    whose wall-clock is the only hint anything went wrong."""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(DeviceError, match="is_available"):
            resolve_device("cuda")


def test_unknown_spec_raises():
    with pytest.raises(DeviceError, match="unrecognized device spec"):
        resolve_device("gpu")
