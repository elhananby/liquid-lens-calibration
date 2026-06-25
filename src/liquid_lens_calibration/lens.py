"""Thin wrapper over the optotune-lens package."""

from optotune_lens import Lens, LensError
from optotune_lens.lens import OperatingMode


def open_lens(port: str = "/dev/optotune_ld") -> tuple[Lens, tuple[float, float]]:
    """Open the Optotune lens and ensure it is in focal-power mode.

    If the lens is already in focal-power mode (e.g. not power-cycled between
    runs), the mode-switch command is skipped to avoid the spurious error 72
    ("already in this mode") returned by the firmware.

    Args:
        port: Serial port device path.

    Returns:
        ``(lens_instance, (min_diopter, max_diopter))``.

    Raises:
        LensError: If connection or mode switch fails.
    """
    lens = Lens(port)

    if lens.mode == OperatingMode.FOCAL_POWER:
        # Already in the right mode; min/max diopter were set by
        # set_temperature_limits during __init__.
        assert lens.min_diopter is not None and lens.max_diopter is not None
        diopter_range = (lens.min_diopter, lens.max_diopter)
    else:
        diopter_range = lens.to_focal_power_mode()
        if lens.mode != OperatingMode.FOCAL_POWER:
            raise LensError(
                f"Failed to switch lens to focal power mode; current mode: {lens.mode}"
            )

    return lens, diopter_range
