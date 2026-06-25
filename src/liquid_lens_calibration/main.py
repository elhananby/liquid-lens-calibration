"""CLI entry point for liquid-lens focus calibration.

Usage::

    uv run python main.py [--exposure 10000] [--calibration <path>]
                          [--port /dev/optotune_ld]
                          [--coarse-steps 20] [--fine-steps 20]
                          [--z-thresh 0.02]
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from liquid_lens_calibration.calibration_io import parse_calibration_xml
from liquid_lens_calibration.cameras import discover_basler_cameras, grab_frame, flush_buffers as flush_basler
from liquid_lens_calibration.focus_camera import XimeaFocusCamera
from liquid_lens_calibration.triangulate import detect_and_triangulate, TAG_FAMILIES
from liquid_lens_calibration.focus import sweep_all_tags
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
        "--coarse-steps",
        type=int,
        default=20,
        help="Diopter steps in the coarse sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--fine-steps",
        type=int,
        default=20,
        help="Diopter steps in the fine sweep per tag (default: %(default)s)",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=50,
        help="Milliseconds to wait after setting diopter (default: %(default)s)",
    )
    parser.add_argument(
        "--z-thresh",
        type=float,
        default=0.02,
        help=(
            "Max z-spread (metres) to treat all visible tags as coplanar "
            "and fuse their z values (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tag-family",
        default="36h11",
        choices=list(TAG_FAMILIES),
        help="Marker dictionary to use for detection (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save a focus-curve plot and ROI crop for each tag after every sweep. "
            "Files are written to ./debug/ as debug_tag<id>_<time>_curve.png "
            "and debug_tag<id>_<time>_roi.jpg."
        ),
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
            print(f"Optotune lens diopter range: {d_min:.2f} to {d_max:.2f} D")
            print(f"Coarse steps: {args.coarse_steps}  Fine steps: {args.fine_steps}")
            print(f"Tags within {args.z_thresh * 1000:.0f} mm z-spread → coplanar (z fused)")
            if args.debug:
                print(f"Debug mode ON — plots and ROI crops saved to ./debug/")

            print("\n" + "=" * 60)
            print("Calibration loop")
            print("  Place AprilTag(s), then press Enter to measure.")
            print("  Multiple tags at different heights → one data point each.")
            print("  Multiple tags at similar height → fused z, one data point.")
            print("  Type 'q' then Enter to quit.")
            print("=" * 60)

            while True:
                cmd = input("\nTarget(s) in position? [Enter=measure, q=quit] ").strip()
                if cmd.lower() in ("q", "quit", "exit"):
                    break

                # Flush all camera buffers
                for cam in basler_cameras.values():
                    flush_basler(cam, n=3)
                focus_cam.flush(n=3)

                # Grab Basler frames and triangulate all visible tags
                frames: dict[str, np.ndarray] = {}
                for cam_id, cam in basler_cameras.items():
                    frames[cam_id] = grab_frame(cam)

                try:
                    tag_results = detect_and_triangulate(
                        frames, calibrations, tag_family=args.tag_family
                    )
                except RuntimeError as e:
                    print(f"  Triangulation failed: {e}")
                    continue

                print(f"  Basler detected {len(tag_results)} tag(s):")
                for tid, (x, y, z, n_cam) in sorted(tag_results.items()):
                    print(f"    tag {tid}: x={x:.4f}, y={y:.4f}, z={z:.4f} m  ({n_cam} cameras)")

                # Determine mode: coplanar (fuse z) or multi-height (per tag)
                zs = [v[2] for v in tag_results.values()]
                z_spread = max(zs) - min(zs)
                single_height = z_spread < args.z_thresh

                if single_height and len(tag_results) > 1:
                    print(f"  z-spread={z_spread * 1000:.1f} mm < {args.z_thresh * 1000:.0f} mm → coplanar, fusing z")

                # Sweep lens — per-tag ROIs auto-detected from XIMEA
                print(
                    f"  Sweeping: {args.coarse_steps} coarse + "
                    f"{args.fine_steps} fine steps per tag …"
                )
                sweep_results = sweep_all_tags(
                    lens,
                    focus_cam,
                    args.tag_family,
                    (d_min, d_max),
                    n_coarse=args.coarse_steps,
                    n_fine=args.fine_steps,
                    settle_s=settle_s,
                    debug=args.debug,
                    debug_dir=Path("debug"),
                )

                if not sweep_results:
                    print("  No tags detected in XIMEA during sweep — skipping.")
                    continue

                timestamp = datetime.now().isoformat(timespec="seconds")

                if single_height:
                    # Fuse z values weighted by number of cameras per tag
                    weights = np.array(
                        [tag_results[tid][3] for tid in tag_results], dtype=np.float64
                    )
                    all_x = [tag_results[tid][0] for tid in tag_results]
                    all_y = [tag_results[tid][1] for tid in tag_results]
                    all_z = [tag_results[tid][2] for tid in tag_results]
                    z_fused = float(np.average(all_z, weights=weights))
                    x_fused = float(np.average(all_x, weights=weights))
                    y_fused = float(np.average(all_y, weights=weights))
                    n_views = int(weights.sum())

                    # Pick the sweep result with the best (highest) metric peak
                    best_tid = max(sweep_results, key=lambda t: sweep_results[t][1])
                    best_d, peak_m, _curv, *_ = sweep_results[best_tid]

                    print(
                        f"  Best focus: {best_d:.3f} D  "
                        f"(z={z_fused:.4f} m, metric peak={peak_m:.1f}, "
                        f"{len(tag_results)} tag(s), {n_views} views)"
                    )
                    dataset.append({
                        "z": z_fused,
                        "diopter": best_d,
                        "x": x_fused,
                        "y": y_fused,
                        "n_cameras": n_views,
                        "n_tags": len(tag_results),
                        "focus_metric_peak": peak_m,
                        "timestamp": timestamp,
                    })

                else:
                    # Multi-height: one data point per tag (XIMEA ∩ Basler)
                    added = 0
                    for tid, (best_d, peak_m, _curv, *_) in sorted(sweep_results.items()):
                        if tid not in tag_results:
                            print(f"  Tag {tid} seen in XIMEA but not triangulated — skipping.")
                            continue
                        x, y, z, n_cam = tag_results[tid]
                        print(
                            f"  Tag {tid}: z={z:.4f} m → {best_d:.3f} D  "
                            f"(metric peak={peak_m:.1f}, {n_cam} cameras)"
                        )
                        dataset.append({
                            "z": z,
                            "diopter": best_d,
                            "x": x,
                            "y": y,
                            "n_cameras": n_cam,
                            "n_tags": 1,
                            "focus_metric_peak": peak_m,
                            "timestamp": timestamp,
                        })
                        added += 1
                    if added == 0:
                        print("  No matching tags between XIMEA and Basler — skipping.")
                        continue

                print(f"  Total data points collected: {len(dataset)}")

    finally:
        for cam in basler_cameras.values():
            try:
                cam.Close()
            except Exception:
                pass

    if len(dataset) < 3:
        print(f"Only {len(dataset)} data point(s) — need at least 3 for a fit.")
        _write_csv(dataset)
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
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        print(f"  a  = {a_fit:.4f}")
        print(f"  z0 = {z0_fit:.4f}")
        print(f"  b  = {b_fit:.4f}")
        print(f"  Residual RMS = {rms:.4f} D")
    except Exception as e:
        print(f"  Fit failed: {e}")

    _write_csv(dataset)


def _write_csv(dataset: list[dict]) -> None:
    if not dataset:
        return
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"lens_calib_{timestamp_str}.csv"
    fieldnames = [
        "z", "diopter", "x", "y", "n_cameras", "n_tags", "focus_metric_peak", "timestamp"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"\nData saved to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
