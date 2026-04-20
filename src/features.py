"""Feature extraction from preprocessed PPG: HR, HRV (SDNN, RMSSD, pNN50), and SQI."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Heart Rate
# ---------------------------------------------------------------------------

def heart_rate(rr_ms: np.ndarray) -> float:
    """Mean heart rate in bpm from R-R intervals (ms)."""
    if len(rr_ms) == 0:
        return float("nan")
    return 60_000.0 / float(np.mean(rr_ms))


# ---------------------------------------------------------------------------
# HRV time-domain metrics
# ---------------------------------------------------------------------------

def sdnn(rr_ms: np.ndarray) -> float:
    """Standard deviation of R-R intervals (ms). Global HRV measure."""
    if len(rr_ms) < 2:
        return float("nan")
    return float(np.std(rr_ms, ddof=1))


def rmssd(rr_ms: np.ndarray) -> float:
    """Root mean square of successive differences (ms). Vagal/parasympathetic proxy."""
    if len(rr_ms) < 2:
        return float("nan")
    successive_diffs = np.diff(rr_ms)
    return float(np.sqrt(np.mean(successive_diffs ** 2)))


def pnn50(rr_ms: np.ndarray) -> float:
    """Percentage of successive R-R differences > 50 ms. Autonomic balance marker."""
    if len(rr_ms) < 2:
        return float("nan")
    successive_diffs = np.abs(np.diff(rr_ms))
    return float(np.mean(successive_diffs > 50.0) * 100.0)


# ---------------------------------------------------------------------------
# Signal Quality Index
# ---------------------------------------------------------------------------

def signal_quality_index(
    signal_raw: np.ndarray,
    signal_filtered: np.ndarray,
    peaks: np.ndarray,
    rr_ms: np.ndarray,
    fs: float,
) -> float:
    """Composite SQI in [0, 1] combining three sub-scores.

    Sub-scores (equal weight):
      1. SNR score  — ratio of signal power in 0.5–4 Hz band to total power.
      2. Peak regularity — 1 minus coefficient of variation of R-R intervals.
      3. Beat count plausibility — fraction of expected beats actually detected,
         capped at 1. Expected beats derived from recording length and mean HR.

    Returns NaN if the signal is too short or no peaks were found.
    """
    if len(peaks) < 2 or len(rr_ms) == 0:
        return float("nan")

    # 1. SNR score via power ratio of filtered vs. raw signal
    power_raw = float(np.mean(signal_raw ** 2))
    power_filtered = float(np.mean(signal_filtered ** 2))
    if power_raw == 0:
        snr_score = 0.0
    else:
        snr_score = min(power_filtered / power_raw, 1.0)

    # 2. R-R regularity: low CV → high quality
    mean_rr = float(np.mean(rr_ms))
    cv = float(np.std(rr_ms, ddof=1)) / mean_rr if mean_rr > 0 else 1.0
    # CV of 0 → score 1; CV ≥ 0.5 (extreme irregularity) → score 0
    regularity_score = float(np.clip(1.0 - cv / 0.5, 0.0, 1.0))

    # 3. Beat count plausibility
    duration_s = len(signal_raw) / fs
    expected_beats = (60_000.0 / mean_rr) * (duration_s / 60.0)
    detected_beats = float(len(peaks))
    beat_score = float(np.clip(detected_beats / expected_beats, 0.0, 1.0)) if expected_beats > 0 else 0.0

    return float(np.mean([snr_score, regularity_score, beat_score]))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def extract_features(preproc: dict, signal_raw: np.ndarray, fs: float) -> dict:
    """Compute all features from a preprocessing result dict.

    Parameters
    ----------
    preproc : output of src.preprocessing.preprocess()
    signal_raw : original (unfiltered) signal array
    fs : sampling frequency in Hz

    Returns
    -------
    dict with keys:
        hr_bpm, sdnn_ms, rmssd_ms, pnn50_pct, sqi,
        n_beats, recording_duration_s, rr_ms
    """
    rr = preproc["rr_ms_filtered"]
    peaks = preproc["peaks"]
    filtered = preproc["filtered"]

    return {
        "hr_bpm": heart_rate(rr),
        "sdnn_ms": sdnn(rr),
        "rmssd_ms": rmssd(rr),
        "pnn50_pct": pnn50(rr),
        "sqi": signal_quality_index(signal_raw, filtered, peaks, rr, fs),
        "n_beats": len(peaks),
        "recording_duration_s": len(signal_raw) / fs,
        "rr_ms": rr,
    }
