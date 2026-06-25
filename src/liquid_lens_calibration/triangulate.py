"""Undistortion, AprilTag detection, and DLT triangulation."""

import numpy as np
import numpy.typing as npt
import cv2

from liquid_lens_calibration.calibration_io import CameraCalibration

_detector_cache: dict[tuple[int, bool], cv2.aruco.ArucoDetector] = {}


def _get_detector(
    dictionary_type: int,
    robust: bool = False,
) -> cv2.aruco.ArucoDetector:
    key = (dictionary_type, robust)
    if key not in _detector_cache:
        d = cv2.aruco.getPredefinedDictionary(dictionary_type)
        p = cv2.aruco.DetectorParameters()
        if robust:
            # Larger adaptive-threshold windows tolerate blur and defocus.
            # Finer step means more threshold candidates are tried.
            # Higher error-correction rate accepts partially corrupted bits.
            # Lower min-perimeter rate allows detection even when the tag is
            # small or occupies only part of the frame.
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 53
            p.adaptiveThreshWinSizeStep = 4
            p.adaptiveThreshConstant = 7
            p.errorCorrectionRate = 0.9
            p.minMarkerPerimeterRate = 0.01
            p.perspectiveRemovePixelPerCell = 8
        _detector_cache[key] = cv2.aruco.ArucoDetector(d, p)
    return _detector_cache[key]


TAG_FAMILIES: dict[str, int] = {
    "36h11": cv2.aruco.DICT_APRILTAG_36H11,
    "36h10": cv2.aruco.DICT_APRILTAG_36H10,
    "25h9": cv2.aruco.DICT_APRILTAG_25H9,
    "16h5": cv2.aruco.DICT_APRILTAG_16H5,
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}


def undistort_pixel(
    pt: tuple[float, float],
    intrinsics: tuple[float, float, float, float],
    dist_coeffs: tuple[float, float, float, float, float],
) -> npt.NDArray[np.float64]:
    """Undistort a single pixel coordinate using OpenCV.

    Args:
        pt: ``(x, y)`` pixel coordinate.
        intrinsics: ``(fx, fy, cx, cy)``.
        dist_coeffs: ``(k1, k2, p1, p2, k3)``.

    Returns:
        Undistorted ``(x, y)`` in *pixel* coordinates (braid convention —
        calibration_matrix maps world→pixel, so the undistorted point stays
        in pixel space for DLT triangulation with the raw 3x4 matrix).
    """
    fx, fy, cx, cy = intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)

    pts = np.array([pt], dtype=np.float64).reshape(1, 1, 2)
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    return undistorted.reshape(2)


def dlt_triangulate(
    points_2d: list[npt.NDArray[np.float64]],
    projection_matrices: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Linear DLT triangulation from 2D-3D correspondences.

    Args:
        points_2d: List of undistorted ``(x, y)`` pixel coordinates, one per camera.
        projection_matrices: List of 3x4 world→pixel projection matrices, one per camera.

    Returns:
        Dehomogenized ``[x, y, z, 1.0]`` (4-element array).

    Raises:
        ValueError: Fewer than 2 correspondences provided.
    """
    n = len(points_2d)
    if n < 2:
        raise ValueError(f"Need at least 2 views, got {n}")

    A = np.zeros((2 * n, 4), dtype=np.float64)
    for i, (pt, P) in enumerate(zip(points_2d, projection_matrices)):
        x, y = pt
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X / X[3]


def detect_apriltags(
    gray: npt.NDArray[np.uint8],
    dictionary_type: int = cv2.aruco.DICT_APRILTAG_36H11,
    robust: bool = False,
) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.int32] | None]:
    """Detect markers in a grayscale image.

    Args:
        gray: Grayscale image.
        dictionary_type: OpenCV aruco dictionary constant.
        robust: Use tuned parameters for blurry / defocused images (XIMEA
            coarse sweep). Slightly slower, more permissive.

    Returns:
        ``(corners_list, ids)`` where each corner array has shape ``(4, 2)``.
        ``ids`` is ``None`` when no markers are detected.
    """
    corners, ids, _ = _get_detector(dictionary_type, robust=robust).detectMarkers(gray)
    return corners, ids


def detect_and_triangulate(
    frames: dict[str, npt.NDArray[np.uint8]],
    calibrations: dict[str, CameraCalibration],
    tag_family: str = "36h11",
) -> dict[int, tuple[float, float, float, int]]:
    """Detect all markers in each camera frame and triangulate each one.

    Every visible tag ID is triangulated independently using all cameras that
    detected it. Tags seen by fewer than 2 calibrated cameras are skipped.

    Args:
        frames: ``{cam_id: grayscale_frame}``.
        calibrations: ``{cam_id: CameraCalibration}`` from the XML.
        tag_family: Key into :data:`TAG_FAMILIES` (default ``"36h11"``).

    Returns:
        ``{tag_id: (x, y, z, n_cameras)}`` — one entry per successfully
        triangulated tag.

    Raises:
        RuntimeError: If no tag is seen by at least 2 calibrated cameras.
    """
    dictionary_type = TAG_FAMILIES.get(tag_family, cv2.aruco.DICT_APRILTAG_36H11)

    # Collect per-tag observations across all cameras
    per_tag: dict[int, list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]] = {}

    for cam_id, gray in frames.items():
        cal = calibrations.get(cam_id)
        if cal is None:
            continue

        corners_list, ids = detect_apriltags(gray, dictionary_type)
        if ids is None:
            continue

        for i, tag_id in enumerate(ids.flatten()):
            corners_4 = corners_list[i].reshape(4, 2)
            center = corners_4.mean(axis=0)
            pt_undistorted = undistort_pixel(
                (float(center[0]), float(center[1])),
                cal.intrinsics,
                cal.distortion_coeffs,
            )
            per_tag.setdefault(int(tag_id), []).append(
                (pt_undistorted, cal.calibration_matrix)
            )

    results: dict[int, tuple[float, float, float, int]] = {}
    for tag_id, obs in per_tag.items():
        if len(obs) < 2:
            continue
        xyz = dlt_triangulate([o[0] for o in obs], [o[1] for o in obs])
        results[tag_id] = (float(xyz[0]), float(xyz[1]), float(xyz[2]), len(obs))

    if not results:
        raise RuntimeError(
            f"No tag seen by ≥2 calibrated cameras (family={tag_family!r})"
        )

    return results
