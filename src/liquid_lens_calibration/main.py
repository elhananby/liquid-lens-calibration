"""CLI entry point for liquid-lens focus calibration.

Usage::

    uv run python main.py [--exposure 10000] [--calibration <path>]
                          [--port /dev/optotune_ld] [--steps 30]
"""

import argparse
import csv
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

from liquid_lens_calibration.calibration_io import parse_calibration_xml
from liquid_lens_calibration.cameras import discover_basler_cameras, grab_frame, flush_buffers as flush_basler
from liquid_lens_calibration.focus_camera import XimeaFocusCamera
from liquid_lens_calibration.triangulate import detect_and_triangulate
from liquid_lens_calibration.focus import sweep_and_find_peak
from liquid_lens_calibration.lens import open_lens


def vergence_model(z: np.ndarray, a: float, z0: float, b: float) -> np.ndarray:
    """Required diopter as a function of object distance (vergence model)."""
    return a / (z - z0) + b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a z → diopter lookup table for an Optotune liquid lens."
    )
    parser.add_argument(
        "--calibration",
        default="/home/nfc/braid-configs/calibration_charuco.xml",
        help="Path to braid multi-camera calibration XML (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        default="/dev/optotune_ld",
        help="Serial port for the Optotune lens (default: %(default)s)",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=10000,
        help="XIMEA exposure in microseconds (default: %(default)s)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diopter steps per sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=50,
        help="Milliseconds to wait after setting diopter (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    calibrations = parse_calibration_xml(args.calibration)
    print(f"Loaded {len(calibrations)} camera calibration(s) from {args.calibration}")

    basler_cameras = discover_basler_cameras(set(calibrations.keys()))
    print(f"Matched {len(basler_cameras)} Basler camera(s)")

    lens, (d_min, d_max) = open_lens(args.port)
    settle_s = args.settle_ms / 1000.0
    dataset: list[dict[str, float | int | str]] = []

    try:
        with XimeaFocusCamera(exposure_us=args.exposure) as focus_cam, lens:
            print("Select focus ROI in the preview window (press SPACE / ENTER when done)")
            roi = focus_cam.select_roi()
            print(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            print(f"Optotune lens diopter range: {d_min:.2f} to {d_max:.2f} D")

            print("\n" + "=" * 60)
            print("Calibration loop")
            print("  Place the AprilTag target, then press Enter to measure.")
            print("  Type 'q' then Enter to quit.")
            print("=" * 60)

            while True:
                cmd = input("\nTarget in position? [Enter=measure, q=quit] ").strip()
                if cmd.lower() in ("q", "quit", "exit"):
                    break

                # Flush all camera buffers
                for cam in basler_cameras.values():
                    flush_basler(cam, n=3)
                focus_cam.flush(n=3)

                # Grab Basler frames
                frames: dict[str, np.ndarray] = {}
                for cam_id, cam in basler_cameras.items():
                    frames[cam_id] = grab_frame(cam)

                # Detect AprilTag & triangulate
                try:
                    x, y, z, n_cam = detect_and_triangulate(frames, calibrations)
                except RuntimeError as e:
                    print(f"  Triangulation failed: {e}")
                    continue

                print(f"  Tag: x={x:.1f}, y={y:.1f}, z={z:.1f}  (seen by {n_cam} cameras)")

                # Sweep lens & find best-focus diopter
                best_diopter, peak_metric, curvature, *_ = sweep_and_find_peak(
                    lens,
                    focus_cam,
                    roi,
                    (d_min, d_max),
                    n_steps=args.steps,
                    settle_s=settle_s,
                )

                print(
                    f"  Best focus: {best_diopter:.3f} D  "
                    f"(metric peak={peak_metric:.1f}, curvature={curvature:.4f})"
                )

                dataset.append({
                    "z": z,
                    "diopter": best_diopter,
                    "x": x,
                    "y": y,
                    "n_cameras": n_cam,
                    "focus_metric_peak": peak_metric,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                print(f"  Collected {len(dataset)} data point(s)")

    finally:
        for cam in basler_cameras.values():
            try:
                cam.Close()
            except Exception:
                pass

    if len(dataset) < 3:
        print(f"Only {len(dataset)} data point(s) — need at least 3 for a fit.")
        return

    z_vals = np.array([d["z"] for d in dataset], dtype=np.float64)
    d_vals = np.array([d["diopter"] for d in dataset], dtype=np.float64)

    print(f"\nFitting vergence model: D = a/(z - z0) + b")
    print(f"  {len(dataset)} data points")

    p0 = (1.0, float(z_vals.min()) - 10.0, float(d_vals.mean()))
    try:
        (a_fit, z0_fit, b_fit), _ = curve_fit(
            vergence_model, z_vals, d_vals, p0=p0, maxfev=10000
        )
        residuals = d_vals - vergence_model(z_vals, a_fit, z0_fit, b_fit)
        rms = float(np.sqrt(np.mean(residuals**2)))

        print(f"  a  = {a_fit:.4f}")
        print(f"  z0 = {z0_fit:.4f}")
        print(f"  b  = {b_fit:.4f}")
        print(f"  Residual RMS = {rms:.4f} D")
    except Exception as e:
        print(f"  Fit failed: {e}")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"lens_calib_{timestamp_str}.csv"
    fieldnames = ["z", "diopter", "x", "y", "n_cameras", "focus_metric_peak", "timestamp"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"\nData saved to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
