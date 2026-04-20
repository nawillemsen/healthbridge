"""
Extract 60-second PPG clips from the BIDMC PPG and Respiration dataset.

Input:  data/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv/
Output: data/demo_cases/*.npy  (float32, 100 Hz, 6000 samples each)

Run from the repository root:
    python scripts/extract_bidmc_clips.py

The BIDMC dataset (125 Hz) is resampled to 100 Hz via a polyphase filter
(up=4, down=5, exact integer ratio). The raw dataset directory is gitignored;
only the extracted clips are committed.

Dataset source:
    Pimentel et al., "Toward a Robust Estimation of Respiratory Rate from
    Pulse Oximeters", IEEE TBME 2017.
    https://physionet.org/content/bidmc/1.0.0/
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

DATASET_DIR = Path("data/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv")
OUTPUT_DIR  = Path("data/demo_cases")
FS_IN       = 125   # Hz — native BIDMC sampling rate
FS_OUT      = 100   # Hz — target (125 * 4/5)
DURATION_S  = 60    # seconds per clip
N_IN        = DURATION_S * FS_IN   # 7500 samples read from CSV
N_OUT       = DURATION_S * FS_OUT  # 6000 samples after resample

# (subject_id, window_start_s, output_filename_stem)
# Windows chosen to maximise SQI for each physiological scenario.
CLIPS: list[tuple[int, int, str]] = [
    (23, 180, "bidmc23_resting_63bpm"),          # HR≈63, SQI≈0.62
    ( 5, 180, "bidmc05_elevated_99bpm"),          # HR≈99, SQI≈0.66
    (32, 120, "bidmc32_low_hrv_81bpm"),           # HR≈81, SQI≈0.65, SDNN≈27 ms
    (26, 360, "bidmc26_low_quality_67bpm"),       # HR≈67, SQI≈0.32 — below gate
]


def load_clip(subject: int, start_s: int) -> np.ndarray:
    """Read PLETH column, slice the requested window, resample to FS_OUT."""
    signals_csv = DATASET_DIR / f"bidmc_{subject:02d}_Signals.csv"
    if not signals_csv.exists():
        raise FileNotFoundError(f"Not found: {signals_csv}")

    start_idx = start_s * FS_IN
    end_idx   = start_idx + N_IN

    with signals_csv.open() as f:
        reader = csv.DictReader(f)
        pleth = [float(row[" PLETH"]) for row in reader]

    if end_idx > len(pleth):
        raise ValueError(
            f"Subject {subject:02d}: requested window [{start_s}s, {start_s+DURATION_S}s] "
            f"exceeds recording length {len(pleth)/FS_IN:.0f}s."
        )

    window = np.array(pleth[start_idx:end_idx], dtype=np.float64)
    resampled = resample_poly(window, up=4, down=5)  # 125 → 100 Hz
    return resampled[:N_OUT].astype(np.float32)


def main() -> None:
    if not DATASET_DIR.exists():
        print(
            f"ERROR: Dataset directory not found:\n  {DATASET_DIR}\n"
            "Download from https://physionet.org/content/bidmc/1.0.0/ "
            "and place in data/bidmc-ppg-and-respiration-dataset-1.0.0/",
            file=sys.stderr,
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subject, start_s, stem in CLIPS:
        out_path = OUTPUT_DIR / f"{stem}.npy"
        print(f"  Subject {subject:02d}  t={start_s}s  →  {out_path} ...", end=" ", flush=True)
        clip = load_clip(subject, start_s)
        np.save(out_path, clip)
        size_kb = out_path.stat().st_size / 1024
        print(f"saved ({clip.shape[0]} samples, {size_kb:.1f} KB)")

    print(f"\nDone. {len(CLIPS)} clips written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
