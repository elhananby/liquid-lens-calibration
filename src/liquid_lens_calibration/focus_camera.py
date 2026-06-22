"""XIMEA focus camera control via ximea-py (xiAPI)."""

from ximea import xiapi

import numpy as np
import numpy.typing as npt
import cv2


class XimeaFocusCamera:
    """Wrapper around the XIMEA CB160CG-LX-X8G3 focus camera.

    Gain is hardcoded to 0.0. Exposure is user-settable.

    Usage::

        with XimeaFocusCamera(exposure_us=10000) as cam:
            roi = cam.select_roi()
            frame = cam.grab_roi_frame(roi)
    """

    def __init__(self, exposure_us: int = 10000) -> None:
        self._exposure_us = exposure_us
        self._cam: xiapi.Camera | None = None

    def __enter__(self) -> "XimeaFocusCamera":
        self._cam = xiapi.Camera()
        self._cam.open_device()
        self._cam.set_gain(0.0)
        self._cam.set_exposure(self._exposure_us)
        # Minimise buffer queue for newest-frame reads
        try:
            self._cam.set_acq_buffer_size(1)
        except Exception:
            pass  # not all xiAPI versions expose this; fallback to throwaway
        self._cam.start_acquisition()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def flush(self, n: int = 3) -> None:
        """Grab and discard *n* frames to flush stale buffer contents."""
        if self._cam is None:
            raise RuntimeError("Camera not opened")
        img = xiapi.Image()
        for _ in range(n):
            self._cam.get_image(img)

    def close(self) -> None:
        """Stop acquisition and close the device."""
        if self._cam is not None:
            try:
                self._cam.stop_acquisition()
            except Exception:
                pass
            try:
                self._cam.close_device()
            except Exception:
                pass
            self._cam = None

    def _grab_newest_frame(self) -> npt.NDArray[np.uint8]:
        """Grab the newest available frame.

        If buffer-size=1 was set, this reads the single buffered frame.
        Otherwise we grab-and-discard a few frames to flush, then grab.
        """
        if self._cam is None:
            raise RuntimeError("Camera not opened")
        self.flush(n=3)
        img = xiapi.Image()
        self._cam.get_image(img)
        return np.asarray(img.get_image_data_numpy(), dtype=np.uint8)

    def select_roi(self) -> tuple[int, int, int, int]:
        """Grab a full preview frame and let the user draw the focus ROI.

        Uses ``cv2.selectROI``. The ROI is defined once at startup and
        never changed afterward.

        Returns:
            ``(x, y, w, h)`` — OpenCV-style bounding box.
        """
        frame = self._grab_newest_frame()
        if frame.ndim == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()
        roi = cv2.selectROI("Select focus region (press SPACE/ENTER when done)", display)
        cv2.destroyWindow("Select focus region (press SPACE/ENTER when done)")
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        if w == 0 or h == 0:
            # No ROI drawn — use full frame
            h, w = frame.shape[:2]
            return (0, 0, w, h)
        return (x, y, w, h)

    def grab_roi_frame(self, roi: tuple[int, int, int, int]) -> npt.NDArray[np.uint8]:
        """Grab the newest frame and crop to the ROI.

        Args:
            roi: ``(x, y, w, h)``.

        Returns:
            Grayscale ROI crop.
        """
        frame = self._grab_newest_frame()
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        x, y, w, h = roi
        return np.asarray(gray[y : y + h, x : x + w], dtype=np.uint8)
