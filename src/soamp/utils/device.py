"""Device resolution, so torch.cuda.is_available() lives in exactly one place.

Every caller that needs a device (pipeline/train.py's Trainer, the PeptideCLM
featurizer's frozen transformer) goes through resolve_device() with a plain
config-driven string, rather than each branching on cuda availability itself --
the same "one source of truth" reasoning the featurizer/model registries use.
"""
import torch


class DeviceError(ValueError):
    """Raised when a device spec is neither 'auto' nor something
    torch.device() accepts, or when a CUDA device is explicitly requested
    on a machine with no CUDA available -- failing loud rather than
    silently falling back to CPU, which would make a 'GPU run' quietly
    not one."""


def resolve_device(spec: str = "auto") -> torch.device:
    """'auto' -> cuda if available else cpu. Anything else is passed to
    torch.device() verbatim ('cpu', 'cuda', 'cuda:1', ...)."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(spec)
    except (RuntimeError, TypeError) as e:
        raise DeviceError(f"unrecognized device spec: {spec!r}") from e
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceError(
            f"device {spec!r} requested but torch.cuda.is_available() is False"
        )
    return device
