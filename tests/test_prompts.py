"""Tests for src.prompts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.prompts import SYSTEM_PROMPT, _fmt, build_messages


class TestBuildMessages:
    def test_returns_list(self, minimal_features):
        result = build_messages(minimal_features)
        assert isinstance(result, list)

    def test_single_turn(self, minimal_features):
        result = build_messages(minimal_features)
        assert len(result) == 1

    def test_role_is_user(self, minimal_features):
        msg = build_messages(minimal_features)[0]
        assert msg["role"] == "user"

    def test_content_is_string(self, minimal_features):
        msg = build_messages(minimal_features)[0]
        assert isinstance(msg["content"], str)

    def test_contains_hr_value(self, minimal_features):
        content = build_messages(minimal_features)[0]["content"]
        hr_str = f"{minimal_features['hr_bpm']:.1f}"
        assert hr_str in content

    def test_contains_sdnn_value(self, minimal_features):
        content = build_messages(minimal_features)[0]["content"]
        assert f"{minimal_features['sdnn_ms']:.1f}" in content

    def test_contains_rmssd_value(self, minimal_features):
        content = build_messages(minimal_features)[0]["content"]
        assert f"{minimal_features['rmssd_ms']:.1f}" in content

    def test_contains_pnn50_value(self, minimal_features):
        content = build_messages(minimal_features)[0]["content"]
        assert f"{minimal_features['pnn50_pct']:.1f}" in content

    def test_sqi_good_label(self, minimal_features):
        # sqi = 0.91 → "good"
        content = build_messages(minimal_features)[0]["content"]
        assert "good" in content

    def test_sqi_fair_label(self, minimal_features):
        feats = {**minimal_features, "sqi": 0.55}
        content = build_messages(feats)[0]["content"]
        assert "fair" in content

    def test_sqi_poor_label(self, minimal_features):
        feats = {**minimal_features, "sqi": 0.25}
        content = build_messages(feats)[0]["content"]
        assert "poor" in content

    def test_nan_sqi_shows_unavailable(self, minimal_features):
        feats = {**minimal_features, "sqi": float("nan")}
        content = build_messages(feats)[0]["content"]
        assert "unavailable" in content

    def test_nan_sqi_labelled_poor(self, minimal_features):
        feats = {**minimal_features, "sqi": float("nan")}
        content = build_messages(feats)[0]["content"]
        assert "poor" in content

    def test_recording_duration_present(self, minimal_features):
        content = build_messages(minimal_features)[0]["content"]
        assert f"{minimal_features['recording_duration_s']:.1f}" in content

    def test_n_beats_present(self, minimal_features):
        content = build_messages(minimal_features)[0]["content"]
        assert str(minimal_features["n_beats"]) in content


class TestSystemPrompt:
    def test_is_nonempty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_contains_uncertainty_guidance(self):
        assert "uncertainty" in SYSTEM_PROMPT.lower() or "may suggest" in SYSTEM_PROMPT.lower()

    def test_contains_no_diagnosis_guidance(self):
        assert "diagnos" in SYSTEM_PROMPT.lower()

    def test_contains_sqi_caveat(self):
        assert "sqi" in SYSTEM_PROMPT.lower() or "signal quality" in SYSTEM_PROMPT.lower()


class TestFmt:
    def test_nan_returns_na(self):
        assert _fmt(float("nan")) == "N/A"

    def test_normal_float_one_decimal(self):
        assert _fmt(65.27) == "65.3"

    def test_zero_decimal(self):
        assert _fmt(65.27, decimals=0) == "65"

    def test_two_decimals(self):
        assert _fmt(1.005, decimals=2) == "1.00"  # float rounding

    def test_negative_value(self):
        result = _fmt(-3.14159, decimals=2)
        assert result == "-3.14"
