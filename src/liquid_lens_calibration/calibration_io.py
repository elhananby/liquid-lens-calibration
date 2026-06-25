"""Parse braid-format multi-camera calibration XML into per-camera data."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class CameraCalibration:
    """Calibration data for one camera from the braid XML."""

    cam_id: str
    calibration_matrix: npt.NDArray[np.float64]  # 3x4 world→pixel projection
    resolution: tuple[int, int]  # width, height in pixels
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy
    distortion_coeffs: tuple[float, float, float, float, float]  # k1, k2, p1, p2, k3 (OpenCV order)
    alpha_c: float = 0.0


def _parse_non_linear(el: ET.Element) -> dict:
    """Extract distortion parameters from a <non_linear_parameters> block."""
    keys = ["fc1", "fc2", "cc1", "cc2", "k1", "k2", "p1", "p2", "k3", "alpha_c"]
    vals = {}
    for k in keys:
        child = el.find(k)
        if child is not None and child.text:
            vals[k] = float(child.text)
        else:
            vals[k] = 0.0
    return vals


def _parse_matrix(text: str) -> npt.NDArray[np.float64]:
    """Parse a 3x4 matrix string (semicolon-separated rows, space-separated values)."""
    rows = text.strip().split(";")
    return np.array(
        [[float(v) for v in row.strip().split()] for row in rows],
        dtype=np.float64,
    )


def parse_calibration_xml(path: str) -> dict[str, CameraCalibration]:
    """Parse braid multi-camera calibration XML.

    Args:
        path: Path to the XML file.

    Returns:
        Dict mapping ``cam_id`` (e.g. ``"Basler-40080153"``) to
        :class:`CameraCalibration`.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    calibrations: dict[str, CameraCalibration] = {}
    for single in root.findall("single_camera_calibration"):
        cam_id_el = single.find("cam_id")
        mat_el = single.find("calibration_matrix")
        res_el = single.find("resolution")
        nl_el = single.find("non_linear_parameters")

        if cam_id_el is None or mat_el is None:
            continue

        cam_id = cam_id_el.text or ""
        matrix = _parse_matrix(mat_el.text or "")

        resolution = (1920, 1200)
        if res_el is not None and res_el.text:
            parts = res_el.text.strip().split()
            if len(parts) == 2:
                resolution = (int(parts[0]), int(parts[1]))

        if nl_el is not None:
            nl = _parse_non_linear(nl_el)
            intrinsics = (nl["fc1"], nl["fc2"], nl["cc1"], nl["cc2"])
            distortion_coeffs = (nl["k1"], nl["k2"], nl["p1"], nl["p2"], nl["k3"])
            alpha_c = nl["alpha_c"]
        else:
            intrinsics = (0.0, 0.0, 0.0, 0.0)
            distortion_coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)
            alpha_c = 0.0

        calibrations[cam_id] = CameraCalibration(
            cam_id=cam_id,
            calibration_matrix=matrix,
            resolution=resolution,
            intrinsics=intrinsics,
            distortion_coeffs=distortion_coeffs,
            alpha_c=alpha_c,
        )

    return calibrations
