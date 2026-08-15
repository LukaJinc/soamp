"""Pure MIC -> active/inactive/uncertain/unlabeled label derivation.

No I/O here -- see pipeline/labeling/02_binarize_mic_labels.py for the
orchestration that resolves a threshold via src/soamp/common/thresholds.py
and looks up recovered censoring bounds via src/soamp/labeling/censoring.py
before calling `derive_label`.
"""
from typing import Literal

Label = Literal["active", "inactive", "uncertain", "unlabeled"]

_DIRECT_MIC_TYPES = ("exact", "averaged")


def derive_label(
    mic_value_uM: float,
    mic_type: str,
    active_threshold_uM: float | None,
    inactive_threshold_uM: float | None,
    recovered_min_uM: float | None = None,
    recovered_max_uM: float | None = None,
) -> tuple[Label, str]:
    """Derive an activity label for one (peptide, organism) row against a
    dual active/inactive breakpoint (see src/soamp/common/thresholds.py) --
    a gap between the two is an intentional gray zone.

    Every mic_type reduces to a plausible-value interval [lo, hi]:
    - exact/averaged: a real point value, lo == hi == mic_value_uM.
    - censored: the true value only known to lie within
      [recovered_min_uM, recovered_max_uM] (see
      src/soamp/labeling/censoring.py).

    Label 'active' only when the entire interval is at or below
    active_threshold_uM, 'inactive' only when the entire interval is at or
    above inactive_threshold_uM -- otherwise 'uncertain' (the interval
    straddles a breakpoint, or falls in the gray zone between them), rather
    than guessing.
    """
    if active_threshold_uM is None or inactive_threshold_uM is None:
        return "unlabeled", "no_threshold_available"

    if mic_type in _DIRECT_MIC_TYPES:
        lo = hi = mic_value_uM
    elif mic_type == "censored":
        if recovered_min_uM is None or recovered_max_uM is None:
            raise ValueError(
                "mic_type='censored' requires recovered_min_uM/recovered_max_uM "
                "(see pipeline/labeling/01_recover_censor_direction.py)"
            )
        lo, hi = recovered_min_uM, recovered_max_uM
    else:
        raise ValueError(f"Unknown mic_type {mic_type!r}, expected one of "
                          f"{_DIRECT_MIC_TYPES + ('censored',)}")

    if hi <= active_threshold_uM:
        return "active", "value_at_or_below_active_threshold"
    if lo >= inactive_threshold_uM:
        return "inactive", "value_at_or_above_inactive_threshold"
    return "uncertain", "value_in_gray_zone_or_straddles_threshold"
