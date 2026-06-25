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
        """Show a live preview (~5 Hz) and let the user draw the focus ROI.

        The preview displays at half resolution to fit the screen, but
        returned coordinates are in full-resolution pixels.

        Click-drag to draw a rectangle, then press SPACE/ENTER to confirm
        or ESC to use the full frame.

        Returns:
            ``(x, y, w, h)`` — OpenCV-style bounding box, full-resolution.
        """
        scale = 0.5
        inv_scale = 2

        # Shared state for the mouse callback
        state: dict[str, int | bool] = {
            "x1": -1, "y1": -1, "x2": -1, "y2": -1,
            "drawing": False, "done": False, "confirmed": False,
        }

        def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drawing"] = True
                state["x1"] = x
                state["y1"] = y
                state["x2"] = x
                state["y2"] = y
            elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
                state["x2"] = x
                state["y2"] = y
            elif event == cv2.EVENT_LBUTTONUP:
                state["drawing"] = False
                state["x2"] = x
                state["y2"] = y

        # Grab first frame: get dimensions and show it to create the window
        # (Qt backend needs an imshow before setMouseCallback).
        frame = self._grab_newest_frame()
        h_full, w_full = frame.shape[:2]
        small_size = (int(w_full * scale), int(h_full * scale))

        if frame.ndim == 2:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()
        small = cv2.resize(display, small_size, interpolation=cv2.INTER_AREA)

        win_name = "Select focus region - drag box, SPACE/ENTER to confirm, ESC full-frame"
        cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(win_name, small)
        cv2.waitKey(1)

        cv2.setMouseCallback(win_name, mouse_callback)

        while not state["done"]:
            frame = self._grab_newest_frame()
            if frame.ndim == 2:
                display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                display = frame.copy()
            small = cv2.resize(display, small_size, interpolation=cv2.INTER_AREA)

            # Draw the current selection rectangle
            if state["x1"] >= 0 and state["y1"] >= 0:
                cv2.rectangle(small,
                              (state["x1"], state["y1"]),  # type: ignore[arg-type]
                              (state["x2"], state["y2"]),  # type: ignore[arg-type]
                              (0, 255, 0), 2)

            cv2.imshow(win_name, small)
            key = cv2.waitKey(200)  # ~5 Hz
            if key in (13, 32):  # ENTER or SPACE
                state["done"] = True
                state["confirmed"] = True
            elif key == 27:  # ESC
                state["done"] = True

        cv2.destroyWindow(win_name)

        if (not state["confirmed"]
                or state["x1"] < 0
                or state["y1"] < 0
                or state["x2"] < 0
                or state["y2"] < 0):
            return (0, 0, w_full, h_full)

        x = min(state["x1"], state["x2"]) * inv_scale  # type: ignore[operator]
        y = min(state["y1"], state["y2"]) * inv_scale  # type: ignore[operator]
        w = abs(state["x2"] - state["x1"]) * inv_scale  # type: ignore[operator]
        h = abs(state["y2"] - state["y1"]) * inv_scale  # type: ignore[operator]

        if w == 0 or h == 0:
            return (0, 0, w_full, h_full)

        return (x, y, w, h)

    def grab_full_frame(self) -> npt.NDArray[np.uint8]:
        """Return the newest full XIMEA frame without any crop."""
        return self._grab_newest_frame()

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
