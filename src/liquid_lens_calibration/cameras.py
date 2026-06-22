"""Basler camera control via pypylon."""

import cv2
from pypylon import pylon

import numpy as np
import numpy.typing as npt


def discover_basler_cameras(
    cam_ids: set[str],
) -> dict[str, pylon.InstantCamera]:
    """Discover Basler cameras and match them to calibration cam_ids.

    Matches by serial number — each ``cam_id`` has the form
    ``"Basler-<serial>"``.

    Args:
        cam_ids: Set of expected ``cam_id`` strings from the calibration XML.

    Returns:
        ``{cam_id: pylon.InstantCamera}`` for each matched camera.

    Raises:
        RuntimeError: If any expected camera is not found.
    """
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    serial_to_cam_id = {}
    for cam_id in cam_ids:
        if cam_id.startswith("Basler-"):
            serial_to_cam_id[cam_id.removeprefix("Basler-")] = cam_id
        else:
            serial_to_cam_id[cam_id] = cam_id

    cameras: dict[str, pylon.InstantCamera] = {}
    for dev in devices:
        serial = dev.GetSerialNumber()
        if serial in serial_to_cam_id:
            cam_id = serial_to_cam_id[serial]
            camera = pylon.InstantCamera(tl_factory.CreateDevice(dev))
            camera.Open()
            cameras[cam_id] = camera

    missing = set(cam_ids) - set(cameras)
    if missing:
        raise RuntimeError(
            f"Could not find {len(missing)} Basler camera(s): {missing}"
        )

    return cameras


def grab_frame(camera: pylon.InstantCamera) -> npt.NDArray[np.uint8]:
    """Grab a single frame from a Basler camera and return it as grayscale.

    Args:
        camera: Opened pylon camera.

    Returns:
        2D grayscale numpy array.
    """
    result = camera.GrabOne(5000)
    img = result.Array
    if img.ndim == 3:
        img = img if img.shape[2] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.asarray(img, dtype=np.uint8)


def flush_buffers(camera: pylon.InstantCamera, n: int = 3) -> None:
    """Grab and discard *n* frames to flush stale buffer contents.

    Args:
        camera: Opened pylon camera.
        n: Number of frames to discard.
    """
    for _ in range(n):
        camera.GrabOne(5000)
