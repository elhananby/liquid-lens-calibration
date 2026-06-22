# Liquid-Lens Focus Calibration Tool

Builds a `z → diopter` lookup table for an Optotune liquid lens, using a
multi-camera Basler rig to measure the true `z` of a static target.

## Usage

```bash
uv run lens-calibrate [options]
# or: uv run python -m liquid_lens_calibration.main [options]
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--exposure` | 10000 | XIMEA exposure in µs |
| `--calibration` | `/home/nfc/braid-configs/calibration_charuco.xml` | Braid multi-camera calibration XML |
| `--port` | `/dev/optotune_ld` | Optotune lens serial port |
| `--steps` | 30 | Diopter sweep steps per measurement |
| `--settle-ms` | 50 | Settle time after setting diopter |

## Procedure

1. Place an AprilTag (tag36h11) target below the cameras.
2. Run the tool — it opens all cameras, loads calibration, asks you to draw
   a focus ROI on the XIMEA preview.
3. For each measurement: the tool flushes buffers, grabs Basler frames,
   detects the tag, triangulates its position, sweeps the lens, and finds the
   best-focus diopter by parabola fit.
4. Move the target ~10–15 times, sampling uniformly in vergence (1/z).
5. On quit, the tool fits `D = a/(z - z0) + b` and writes a timestamped CSV.

## Dependencies

- Python ≥3.12
- Basler Pylon SDK runtime (for `pypylon`)
- XIMEA xiAPI runtime (for `ximea-py`)
- Optotune lens driver (`optotune-lens` package at `../optotune-lens`)
