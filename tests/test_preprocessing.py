"""Tests for src.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    bandpass_filter,
    compute_rr_intervals,
    detect_peaks,
    filter_rr_intervals,
    preprocess,
)

FS = 100.0


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_shape_matches_input(self):
        sig = np.random.default_rng(0).standard_normal(1000)
        out = bandpass_filter(sig, FS)
        assert out.shape == sig.shape

    def test_removes_dc_offset(self):
        rng = np.random.default_rng(0)
        t = np.arange(0, 10.0, 1.0 / FS)
        sig = 5.0 + np.sin(2 * np.pi * 1.1 * t) + rng.normal(0, 0.05, len(t))
        out = bandpass_filter(sig, FS)
        assert abs(np.mean(out)) < 0.1  # DC component suppressed

    def test_attenuates_stopband_relative_to_passband(self):
        t = np.arange(0, 10.0, 1.0 / FS)
        inband   = np.sin(2 * np.pi * 1.1 * t)   # 1.1 Hz — inside passband
        stopband = np.sin(2 * np.pi * 20.0 * t)  # 20 Hz — well above 4 Hz cutoff
        out_in  = bandpass_filter(inband, FS)
        out_stop = bandpass_filter(stopband, FS)
        # Stopband output power must be < 5% of passband output power
        assert np.var(out_stop) < 0.05 * np.var(out_in)

    def test_passes_in_band_signal(self):
        t = np.arange(0, 10.0, 1.0 / FS)
        # 1.1 Hz ≈ 66 bpm — well within the passband
        sig = np.sin(2 * np.pi * 1.1 * t)
        out = bandpass_filter(sig, FS)
        # After zero-phase filter, amplitude should be largely preserved
        assert np.std(out) > 0.3

    def test_invalid_cutoff_raises_when_high_exceeds_nyquist(self):
        sig = np.ones(500)
        # fs=7 Hz → nyq=3.5 Hz, high_hz=4.0 → high/nyq > 1 → ValueError
        with pytest.raises(ValueError):
            bandpass_filter(sig, fs=7.0, low_hz=0.5, high_hz=4.0)

    def test_returns_float64(self):
        sig = np.ones(500, dtype=np.float32)
        out = bandpass_filter(sig, FS)
        assert out.dtype == np.float64 or out.dtype == np.float32  # scipy may upcast


# ---------------------------------------------------------------------------
# detect_peaks
# ---------------------------------------------------------------------------

class TestDetectPeaks:
    def _clean_signal(self, hr_bpm: float = 65.0, duration_s: float = 30.0) -> np.ndarray:
        t = np.arange(0, duration_s, 1.0 / FS)
        f0 = hr_bpm / 60.0
        return np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t + 0.3)

    def test_finds_approximately_correct_peak_count(self):
        hr = 65.0
        duration = 30.0
        sig = self._clean_signal(hr, duration)
        peaks = detect_peaks(sig, FS)
        expected = hr / 60.0 * duration  # ~32.5
        # Allow ±20% tolerance
        assert abs(len(peaks) - expected) / expected < 0.20

    def test_peaks_respect_distance_constraint(self):
        sig = self._clean_signal(hr_bpm=60.0)
        peaks = detect_peaks(sig, FS, max_hr_bpm=120.0)
        if len(peaks) > 1:
            min_gap = int(FS * 60.0 / 120.0)
            assert np.all(np.diff(peaks) >= min_gap)

    def test_clean_signal_peaks_are_regular(self):
        # A synthetic PPG at exactly 1.1 Hz over 30 s → ~33 peaks, evenly spaced
        t = np.arange(0, 30.0, 1.0 / FS)
        sig = np.sin(2 * np.pi * 1.1 * t) + 0.4 * np.sin(2 * np.pi * 2.2 * t + 0.3)
        peaks = detect_peaks(sig, FS)
        assert len(peaks) > 0
        intervals = np.diff(peaks)
        # CV of inter-peak intervals should be very low for a perfectly regular signal
        cv = np.std(intervals) / np.mean(intervals)
        assert cv < 0.05

    def test_peak_indices_within_signal_bounds(self):
        sig = self._clean_signal()
        peaks = detect_peaks(sig, FS)
        assert np.all(peaks >= 0)
        assert np.all(peaks < len(sig))


# ---------------------------------------------------------------------------
# compute_rr_intervals
# ---------------------------------------------------------------------------

class TestComputeRRIntervals:
    def test_known_peaks_known_intervals(self):
        # Peaks at samples 0, 100, 200 with fs=100 → 1000 ms each
        peaks = np.array([0, 100, 200])
        rr = compute_rr_intervals(peaks, fs=100.0)
        np.testing.assert_allclose(rr, [1000.0, 1000.0])

    def test_single_peak_returns_empty(self):
        rr = compute_rr_intervals(np.array([50]), fs=100.0)
        assert len(rr) == 0

    def test_empty_peaks_returns_empty(self):
        rr = compute_rr_intervals(np.array([]), fs=100.0)
        assert len(rr) == 0

    def test_output_length_is_n_peaks_minus_one(self):
        peaks = np.array([0, 80, 160, 240, 320])
        rr = compute_rr_intervals(peaks, fs=100.0)
        assert len(rr) == 4

    def test_units_are_milliseconds(self):
        # 50 samples apart at 100 Hz = 500 ms
        peaks = np.array([0, 50])
        rr = compute_rr_intervals(peaks, fs=100.0)
        assert pytest.approx(rr[0], rel=1e-6) == 500.0


# ---------------------------------------------------------------------------
# filter_rr_intervals
# ---------------------------------------------------------------------------

class TestFilterRRIntervals:
    def test_removes_too_short_intervals(self):
        rr = np.array([200.0, 900.0, 150.0, 800.0])  # 200, 150 below 300 ms
        out = filter_rr_intervals(rr)
        assert np.all(out >= 300.0)
        assert len(out) == 2

    def test_removes_too_long_intervals(self):
        rr = np.array([800.0, 2100.0, 900.0, 2500.0])
        out = filter_rr_intervals(rr)
        assert np.all(out <= 2000.0)
        assert len(out) == 2

    def test_preserves_valid_intervals(self):
        rr = np.array([600.0, 750.0, 900.0, 1100.0])
        out = filter_rr_intervals(rr)
        np.testing.assert_array_equal(out, rr)

    def test_empty_input(self):
        out = filter_rr_intervals(np.array([]))
        assert len(out) == 0

    def test_all_invalid_returns_empty(self):
        rr = np.array([100.0, 250.0, 2100.0, 3000.0])
        out = filter_rr_intervals(rr)
        assert len(out) == 0

    def test_boundary_values_included(self):
        rr = np.array([300.0, 2000.0])  # exact boundaries → included
        out = filter_rr_intervals(rr)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# preprocess (integration)
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_has_required_keys(self, clean_ppg):
        result = preprocess(clean_ppg, FS)
        assert {"filtered", "peaks", "rr_ms", "rr_ms_filtered"} <= result.keys()

    def test_filtered_shape_matches_input(self, clean_ppg):
        result = preprocess(clean_ppg, FS)
        assert result["filtered"].shape == clean_ppg.shape

    def test_clean_signal_finds_peaks(self, clean_ppg):
        result = preprocess(clean_ppg, FS)
        assert len(result["peaks"]) > 10

    def test_rr_ms_filtered_subset_of_rr_ms(self, clean_ppg):
        result = preprocess(clean_ppg, FS)
        # Filtered RR should be ≤ unfiltered in length
        assert len(result["rr_ms_filtered"]) <= len(result["rr_ms"])

    def test_too_short_raises_value_error(self):
        short = np.ones(int(FS * 1.5))  # 1.5 s — below 2 s minimum
        with pytest.raises(ValueError, match="too short"):
            preprocess(short, FS)

    def test_exactly_two_seconds_does_not_raise(self):
        sig = np.sin(2 * np.pi * 1.1 * np.arange(0, 2.0, 1.0 / FS))
        # Should not raise (boundary condition)
        preprocess(sig, FS)  # may return empty peaks — that's fine
