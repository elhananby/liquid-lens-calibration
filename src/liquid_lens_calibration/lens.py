"""Thin wrapper over the optotune-lens package."""

from optotune_lens import Lens, LensError


def open_lens(port: str = "/dev/optotune_ld") -> tuple[Lens, tuple[float, float]]:
    """Open the Optotune lens and switch to focal-power mode.

    Args:
        port: Serial port device path.

    Returns:
        ``(lens_instance, (min_diopter, max_diopter))``.

    Raises:
        LensError: If connection or mode switch fails.
    """
    lens = Lens(port)
    diopter_range = lens.to_focal_power_mode()
    return lens, diopter_range
