"""PPG signal preprocessing: filtering, peak detection, R-R interval extraction."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 4.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter covering typical HR range (30–240 bpm)."""
    nyq = fs / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    if low <= 0 or high >= 1:
        raise ValueError(f"Filter cutoffs out of range for fs={fs} Hz: [{low_hz}, {high_hz}] Hz")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detect_peaks(signal: np.ndarray, fs: float, min_hr_bpm: float = 30.0, max_hr_bpm: float = 240.0) -> np.ndarray:
    """Detect systolic peaks in a filtered PPG signal.

    Returns array of peak indices. Distance constraint derived from max_hr_bpm
    so closely spaced noise spikes are rejected.
    """
    min_distance_samples = int(fs * 60.0 / max_hr_bpm)
    # Prominence threshold: 10% of peak-to-peak amplitude keeps low-amplitude noise out
    ptp = np.ptp(signal)
    prominence = 0.10 * ptp if ptp > 0 else 0.0
    peaks, _ = find_peaks(signal, distance=min_distance_samples, prominence=prominence)
    return peaks


def compute_rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Convert peak sample indices to R-R (peak-to-peak) intervals in milliseconds."""
    if len(peaks) < 2:
        return np.array([], dtype=float)
    return np.diff(peaks.astype(float)) / fs * 1000.0


def filter_rr_intervals(rr_ms: np.ndarray, min_ms: float = 300.0, max_ms: float = 2000.0) -> np.ndarray:
    """Remove physiologically implausible R-R intervals (< 300 ms or > 2000 ms)."""
    return rr_ms[(rr_ms >= min_ms) & (rr_ms <= max_ms)]


def preprocess(signal: np.ndarray, fs: float) -> dict:
    """Full preprocessing pipeline: filter → peaks → R-R intervals.

    Returns a dict with keys: filtered, peaks, rr_ms, rr_ms_filtered.
    """
    if len(signal) < int(fs * 2):
        raise ValueError("Signal too short: need at least 2 seconds of data.")

    filtered = bandpass_filter(signal, fs)
    peaks = detect_peaks(filtered, fs)
    rr_ms = compute_rr_intervals(peaks, fs)
    rr_ms_filtered = filter_rr_intervals(rr_ms)

    return {
        "filtered": filtered,
        "peaks": peaks,
        "rr_ms": rr_ms,
        "rr_ms_filtered": rr_ms_filtered,
    }
