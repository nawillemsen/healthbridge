"""Shared fixtures for HealthBridge test suite."""

from __future__ import annotations

import numpy as np
import pytest

from src.features import extract_features
from src.preprocessing import preprocess


FS = 100.0  # Hz used across all fixtures


def _make_clean_ppg(hr_bpm: float = 65.0, duration_s: float = 30.0, fs: float = FS) -> np.ndarray:
    """Synthetic PPG: fundamental + two harmonics, minimal noise."""
    rng = np.random.default_rng(0)
    t = np.arange(0, duration_s, 1.0 / fs)
    f0 = hr_bpm / 60.0
    sig = (
        np.sin(2 * np.pi * f0 * t)
        + 0.4 * np.sin(2 * np.pi * 2 * f0 * t + 0.3)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t + 0.6)
    )
    sig += rng.normal(0, 0.03, len(t))
    return sig


@pytest.fixture(scope="session")
def fs():
    return FS


@pytest.fixture(scope="session")
def clean_ppg():
    return _make_clean_ppg(hr_bpm=65.0)


@pytest.fixture(scope="session")
def noisy_ppg():
    rng = np.random.default_rng(1)
    t = np.arange(0, 30.0, 1.0 / FS)
    return rng.normal(0, 1.0, len(t))  # pure noise — no detectable peaks


@pytest.fixture(scope="session")
def preproc_result(clean_ppg):
    return preprocess(clean_ppg, FS)


@pytest.fixture(scope="session")
def features_dict(preproc_result, clean_ppg):
    return extract_features(preproc_result, clean_ppg, FS)


@pytest.fixture
def minimal_features():
    """Hand-crafted features dict suitable for prompts/evaluation tests."""
    return {
        "hr_bpm": 65.2,
        "sdnn_ms": 48.3,
        "rmssd_ms": 41.7,
        "pnn50_pct": 23.1,
        "sqi": 0.91,
        "n_beats": 65,
        "recording_duration_s": 60.0,
        "rr_ms": np.array([920.0, 930.0, 910.0, 940.0]),
    }
