"""Tests for src.features."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.features import (
    extract_features,
    heart_rate,
    pnn50,
    rmssd,
    sdnn,
    signal_quality_index,
)


# ---------------------------------------------------------------------------
# heart_rate
# ---------------------------------------------------------------------------

class TestHeartRate:
    def test_one_second_rr_is_60bpm(self):
        rr = np.array([1000.0, 1000.0, 1000.0])
        assert pytest.approx(heart_rate(rr)) == 60.0

    def test_500ms_rr_is_120bpm(self):
        rr = np.full(5, 500.0)
        assert pytest.approx(heart_rate(rr)) == 120.0

    def test_empty_rr_returns_nan(self):
        assert math.isnan(heart_rate(np.array([])))

    def test_single_value(self):
        rr = np.array([750.0])
        assert pytest.approx(heart_rate(rr), rel=1e-4) == 80.0

    def test_mixed_intervals_uses_mean(self):
        # mean of [800, 1200] = 1000 ms → 60 bpm
        rr = np.array([800.0, 1200.0])
        assert pytest.approx(heart_rate(rr)) == 60.0


# ---------------------------------------------------------------------------
# sdnn
# ---------------------------------------------------------------------------

class TestSDNN:
    def test_known_value(self):
        # [900, 1000, 1100]: std(ddof=1) = 100.0
        rr = np.array([900.0, 1000.0, 1100.0])
        assert pytest.approx(sdnn(rr), rel=1e-6) == 100.0

    def test_constant_rr_is_zero(self):
        rr = np.full(10, 1000.0)
        assert pytest.approx(sdnn(rr), abs=1e-10) == 0.0

    def test_single_value_returns_nan(self):
        assert math.isnan(sdnn(np.array([1000.0])))

    def test_empty_returns_nan(self):
        assert math.isnan(sdnn(np.array([])))

    def test_nonnegative(self, features_dict):
        val = features_dict["sdnn_ms"]
        assert math.isnan(val) or val >= 0.0


# ---------------------------------------------------------------------------
# rmssd
# ---------------------------------------------------------------------------

class TestRMSSD:
    def test_known_value(self):
        # diffs = [100, -200, 100], squared = [10000, 40000, 10000], mean = 20000, sqrt ≈ 141.42
        rr = np.array([1000.0, 1100.0, 900.0, 1000.0])
        expected = math.sqrt(20000.0)
        assert pytest.approx(rmssd(rr), rel=1e-5) == expected

    def test_constant_rr_is_zero(self):
        rr = np.full(10, 900.0)
        assert pytest.approx(rmssd(rr), abs=1e-10) == 0.0

    def test_single_value_returns_nan(self):
        assert math.isnan(rmssd(np.array([1000.0])))

    def test_empty_returns_nan(self):
        assert math.isnan(rmssd(np.array([])))

    def test_nonnegative(self, features_dict):
        val = features_dict["rmssd_ms"]
        assert math.isnan(val) or val >= 0.0


# ---------------------------------------------------------------------------
# pnn50
# ---------------------------------------------------------------------------

class TestPNN50:
    def test_all_diffs_above_50_gives_100_percent(self):
        rr = np.array([1000.0, 1060.0, 1000.0, 1070.0])
        # diffs: 60, -60, 70 → all abs > 50
        assert pytest.approx(pnn50(rr)) == 100.0

    def test_no_diffs_above_50_gives_zero(self):
        rr = np.array([1000.0, 1010.0, 1020.0, 1030.0])
        # diffs: 10, 10, 10 → none > 50
        assert pytest.approx(pnn50(rr)) == 0.0

    def test_half_diffs_above_50(self):
        # diffs: 60 (>50), -50 (not >50) → 1/2 = 50%
        rr = np.array([1000.0, 1060.0, 1010.0])
        assert pytest.approx(pnn50(rr)) == 50.0

    def test_boundary_50ms_not_counted(self):
        rr = np.array([1000.0, 1050.0])
        # diff = 50: NOT > 50, so 0%
        assert pytest.approx(pnn50(rr)) == 0.0

    def test_single_value_returns_nan(self):
        assert math.isnan(pnn50(np.array([1000.0])))

    def test_empty_returns_nan(self):
        assert math.isnan(pnn50(np.array([])))

    def test_range_is_0_to_100(self, features_dict):
        val = features_dict["pnn50_pct"]
        assert math.isnan(val) or (0.0 <= val <= 100.0)


# ---------------------------------------------------------------------------
# signal_quality_index
# ---------------------------------------------------------------------------

class TestSQI:
    def _make_inputs(self, hr_bpm: float = 65.0, fs: float = 100.0, duration_s: float = 30.0):
        from src.preprocessing import preprocess
        t = np.arange(0, duration_s, 1.0 / fs)
        f0 = hr_bpm / 60.0
        raw = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t + 0.3)
        raw += np.random.default_rng(7).normal(0, 0.03, len(t))
        p = preprocess(raw, fs)
        return raw, p["filtered"], p["peaks"], p["rr_ms_filtered"], fs

    def test_clean_signal_sqi_above_threshold(self):
        raw, filt, peaks, rr, fs = self._make_inputs()
        sqi = signal_quality_index(raw, filt, peaks, rr, fs)
        assert not math.isnan(sqi)
        assert sqi > 0.5

    def test_sqi_bounded_in_0_1(self):
        raw, filt, peaks, rr, fs = self._make_inputs()
        sqi = signal_quality_index(raw, filt, peaks, rr, fs)
        assert 0.0 <= sqi <= 1.0

    def test_no_peaks_returns_nan(self):
        raw = np.ones(3000)
        filt = np.ones(3000)
        sqi = signal_quality_index(raw, filt, peaks=np.array([]), rr_ms=np.array([]), fs=100.0)
        assert math.isnan(sqi)

    def test_single_peak_returns_nan(self):
        raw = np.ones(3000)
        filt = np.ones(3000)
        sqi = signal_quality_index(raw, filt, peaks=np.array([100]), rr_ms=np.array([]), fs=100.0)
        assert math.isnan(sqi)

    def test_zero_power_raw_signal_does_not_crash(self):
        from src.preprocessing import preprocess
        t = np.arange(0, 30.0, 0.01)
        f0 = 1.1
        raw = np.sin(2 * np.pi * f0 * t)
        p = preprocess(raw, 100.0)
        # Zero-mean raw after DC removal — just confirm no exception
        sqi = signal_quality_index(np.zeros_like(raw), p["filtered"], p["peaks"], p["rr_ms_filtered"], 100.0)
        # sqi may be nan or 0 but must not raise
        assert math.isnan(sqi) or 0.0 <= sqi <= 1.0


# ---------------------------------------------------------------------------
# extract_features (integration)
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    EXPECTED_KEYS = {"hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct", "sqi", "n_beats", "recording_duration_s", "rr_ms"}

    def test_all_keys_present(self, features_dict):
        assert self.EXPECTED_KEYS <= features_dict.keys()

    def test_scalar_fields_are_float(self, features_dict):
        scalar_keys = self.EXPECTED_KEYS - {"rr_ms", "n_beats"}
        for key in scalar_keys:
            assert isinstance(features_dict[key], float), f"{key} should be float"

    def test_n_beats_is_int(self, features_dict):
        assert isinstance(features_dict["n_beats"], int)

    def test_rr_ms_is_array(self, features_dict):
        assert isinstance(features_dict["rr_ms"], np.ndarray)

    def test_recording_duration_matches_signal(self, preproc_result, clean_ppg):
        fs = 100.0
        feats = extract_features(preproc_result, clean_ppg, fs)
        expected = len(clean_ppg) / fs
        assert pytest.approx(feats["recording_duration_s"]) == expected

    def test_hr_in_physiological_range(self, features_dict):
        hr = features_dict["hr_bpm"]
        assert math.isnan(hr) or (20.0 <= hr <= 250.0)

    def test_sqi_in_unit_interval(self, features_dict):
        sqi = features_dict["sqi"]
        assert math.isnan(sqi) or (0.0 <= sqi <= 1.0)

    def test_n_beats_positive(self, features_dict):
        assert features_dict["n_beats"] >= 0
