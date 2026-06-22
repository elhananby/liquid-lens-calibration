"""Focus metric computation and best-focus parabola fit."""

import time

import numpy as np
import numpy.typing as npt
import cv2


def focus_metric(patch: npt.NDArray[np.uint8]) -> float:
    """Compute a focus metric on an image patch.

    Uses variance of Laplacian — a simple, well-behaved contrast
    measure. Higher values indicate sharper focus.

    Args:
        patch: Grayscale image patch.

    Returns:
        Scalar focus metric.
    """
    laplacian = cv2.Laplacian(patch, cv2.CV_64F)
    return float(laplacian.var())


def fit_parabola(
    xs: npt.NDArray[np.float64], ys: npt.NDArray[np.float64]
) -> tuple[float, float, float]:
    """Fit a parabola ``y = a*(x - x0)^2 + y0`` by least squares.

    Args:
        xs: Independent variable values (e.g. diopter steps).
        ys: Dependent variable values (e.g. focus metrics).

    Returns:
        ``(x0, y0, a)`` — vertex ``(x0, y0)`` and curvature ``a``.
    """
    A = np.column_stack([xs * xs, xs, np.ones_like(xs)])
    coeffs, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a2, b, c = coeffs
    x0 = -b / (2 * a2)
    y0 = a2 * x0 * x0 + b * x0 + c
    return x0, y0, a2


def sweep_and_find_peak(
    lens,
    focus_cam,
    roi: tuple[int, int, int, int],
    diopter_range: tuple[float, float],
    n_steps: int = 30,
    settle_s: float = 0.05,
) -> tuple[float, float, float, list[float], list[float]]:
    """Sweep the lens across its diopter range and find best focus.

    For each step: set diopter, wait a brief settle, grab a focus frame,
    crop to ROI, compute the focus metric. After the sweep, fit a parabola
    to the data near the metric peak and return the vertex diopter.

    Args:
        lens: Optotune lens instance (must be in focal-power mode).
        focus_cam: :class:`XimeaFocusCamera` instance.
        roi: ``(x, y, w, h)`` ROI for the focus metric.
        diopter_range: ``(min_diopter, max_diopter)``.
        n_steps: Number of diopter steps in the sweep.
        settle_s: Seconds to wait after setting diopter before grabbing frame.

    Returns:
        ``(best_diopter, peak_metric, curvature, diopters_list, metrics_list)``.
    """
    d_min, d_max = diopter_range
    diopters = np.linspace(d_min, d_max, n_steps, dtype=np.float64)
    metrics: list[float] = []

    for d in diopters:
        lens.set_diopter(float(d))
        time.sleep(settle_s)
        patch = focus_cam.grab_roi_frame(roi)
        m = focus_metric(patch)
        metrics.append(m)

    # Find the argmax region and fit parabola around it
    xs = np.array(diopters, dtype=np.float64)
    ys = np.array(metrics, dtype=np.float64)
    peak_idx = int(np.argmax(ys))

    # Take a window around the peak (at least 3 points for a 3-param fit)
    half_window = max(2, n_steps // 6)
    left = max(0, peak_idx - half_window)
    right = min(len(xs), peak_idx + half_window + 1)

    if right - left < 3:
        # Fall back to argmax if window is too small
        return float(xs[peak_idx]), float(ys[peak_idx]), 0.0, diopters.tolist(), metrics

    x0, y0, a = fit_parabola(xs[left:right], ys[left:right])
    return x0, y0, a, diopters.tolist(), metrics
