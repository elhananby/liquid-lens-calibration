"""Undistortion, AprilTag detection, and DLT triangulation."""

import numpy as np
import numpy.typing as npt
import cv2

from liquid_lens_calibration.calibration_io import CameraCalibration


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
        Homogeneous 4-vector or dehomogenized ``(x, y, z)``.

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
) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.int32]]:
    """Detect AprilTag markers (tag36h11) in a grayscale image.

    Args:
        gray: Grayscale image.

    Returns:
        ``(corners_list, ids)`` where each corner array has shape ``(4, 2)``.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def detect_and_triangulate(
    frames: dict[str, npt.NDArray[np.uint8]],
    calibrations: dict[str, CameraCalibration],
) -> tuple[float, float, float, int]:
    """Detect an AprilTag in each camera frame and triangulate its 3D position.

    Args:
        frames: ``{cam_id: grayscale_frame}``.
        calibrations: ``{cam_id: CameraCalibration}`` from the XML.

    Returns:
        ``(x, y, z, n_cameras)`` — triangulated world position and
        number of cameras that successfully detected the tag.

    Raises:
        RuntimeError: If no camera detects exactly one tag.
    """
    pts_2d: list[npt.NDArray[np.float64]] = []
    proj_mats: list[npt.NDArray[np.float64]] = []

    for cam_id, gray in frames.items():
        cal = calibrations.get(cam_id)
        if cal is None:
            continue

        corners, ids = detect_apriltags(gray)
        if ids is None or len(ids) != 1:
            continue

        # Tag center as pixel coordinate
        corners_4 = corners[0].reshape(4, 2)
        center = corners_4.mean(axis=0)

        # Undistort (braid convention: undistort to pixel coords)
        pt_undistorted = undistort_pixel(
            (float(center[0]), float(center[1])),
            cal.intrinsics,
            cal.distortion_coeffs,
        )

        pts_2d.append(pt_undistorted)
        proj_mats.append(cal.calibration_matrix)

    n_cameras = len(pts_2d)
    if n_cameras < 2:
        raise RuntimeError(
            f"Tag detected in only {n_cameras} camera(s); need at least 2"
        )

    xyz = dlt_triangulate(pts_2d, proj_mats)
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return x, y, z, n_cameras
