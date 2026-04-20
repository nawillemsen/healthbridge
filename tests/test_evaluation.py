"""Tests for src.evaluation (no API calls, no BERTScore model downloads)."""

from __future__ import annotations

import json
import math
from unittest.mock import patch

import numpy as np
import pytest

from src.evaluation import (
    aggregate_scores,
    score_rouge,
    score_rouge_batch,
    sqi_gate,
    evaluate_gold_set,
)


# ---------------------------------------------------------------------------
# score_rouge
# ---------------------------------------------------------------------------

class TestScoreRouge:
    def test_identical_texts_score_one(self):
        text = "The heart rate of 65 bpm is within the normal resting range."
        scores = score_rouge(text, text)
        assert scores["rouge1"] == pytest.approx(1.0)
        assert scores["rouge2"] == pytest.approx(1.0)
        assert scores["rougeL"] == pytest.approx(1.0)

    def test_completely_different_texts_score_low(self):
        cand = "apple banana cherry"
        ref = "delta echo foxtrot"
        scores = score_rouge(cand, ref)
        assert scores["rouge1"] == pytest.approx(0.0)
        assert scores["rouge2"] == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        cand = "heart rate is normal and resting"
        ref  = "heart rate is elevated above resting range"
        scores = score_rouge(cand, ref)
        assert 0.0 < scores["rouge1"] < 1.0

    def test_returns_all_three_keys(self):
        scores = score_rouge("foo bar", "foo baz")
        assert {"rouge1", "rouge2", "rougeL"} == scores.keys()

    def test_scores_are_floats_in_unit_interval(self):
        scores = score_rouge("some text here", "other text there")
        for v in scores.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_empty_candidate(self):
        scores = score_rouge("", "some reference text")
        assert scores["rouge1"] == pytest.approx(0.0)

    def test_empty_reference(self):
        scores = score_rouge("some candidate text", "")
        assert scores["rouge1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_rouge_batch
# ---------------------------------------------------------------------------

class TestScoreRougeBatch:
    def test_returns_correct_length(self):
        cands = ["text one", "text two", "text three"]
        refs  = ["ref one",  "ref two",  "ref three"]
        result = score_rouge_batch(cands, refs)
        assert len(result) == 3

    def test_each_element_has_rouge_keys(self):
        result = score_rouge_batch(["a b c"], ["a b d"])
        assert {"rouge1", "rouge2", "rougeL"} <= result[0].keys()

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            score_rouge_batch(["a", "b"], ["x"])

    def test_self_score_batch_all_ones(self):
        texts = ["heart rate 65 bpm", "hrv is low today", "signal quality poor"]
        results = score_rouge_batch(texts, texts)
        for r in results:
            assert r["rouge1"] == pytest.approx(1.0)

    def test_empty_lists_return_empty(self):
        assert score_rouge_batch([], []) == []


# ---------------------------------------------------------------------------
# sqi_gate
# ---------------------------------------------------------------------------

class TestSQIGate:
    def _make_results(self, sqis):
        return [{"id": f"c{i}", "sqi": s, "rouge1": 0.5} for i, s in enumerate(sqis)]

    def test_splits_correctly_at_threshold(self):
        results = self._make_results([0.9, 0.7, 0.35, 0.28])
        passing, flagged = sqi_gate(results, threshold=0.40)
        assert len(passing) == 2
        assert len(flagged) == 2

    def test_nan_sqi_always_flagged(self):
        results = self._make_results([float("nan"), 0.8])
        passing, flagged = sqi_gate(results)
        assert len(flagged) == 1
        assert flagged[0]["id"] == "c0"

    def test_exact_threshold_passes(self):
        results = self._make_results([0.40])
        passing, flagged = sqi_gate(results, threshold=0.40)
        assert len(passing) == 1
        assert len(flagged) == 0

    def test_just_below_threshold_flagged(self):
        results = self._make_results([0.3999])
        passing, flagged = sqi_gate(results, threshold=0.40)
        assert len(flagged) == 1

    def test_empty_input(self):
        passing, flagged = sqi_gate([])
        assert passing == []
        assert flagged == []

    def test_all_passing(self):
        results = self._make_results([0.8, 0.9, 0.75])
        passing, flagged = sqi_gate(results, threshold=0.40)
        assert len(passing) == 3
        assert len(flagged) == 0

    def test_all_flagged(self):
        results = self._make_results([0.1, 0.2, 0.3])
        passing, flagged = sqi_gate(results, threshold=0.40)
        assert len(passing) == 0
        assert len(flagged) == 3

    def test_missing_sqi_key_treated_as_nan(self):
        results = [{"id": "x", "rouge1": 0.5}]  # no 'sqi' key
        passing, flagged = sqi_gate(results)
        assert len(flagged) == 1


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------

class TestAggregateScores:
    def test_empty_input_returns_empty_dict(self):
        assert aggregate_scores([]) == {}

    def test_known_mean_and_std(self):
        results = [
            {"rouge1": 0.8, "rouge2": 0.6},
            {"rouge1": 0.6, "rouge2": 0.4},
        ]
        agg = aggregate_scores(results)
        assert agg["rouge1"]["mean"] == pytest.approx(0.7, abs=1e-4)
        assert agg["rouge1"]["std"]  == pytest.approx(0.1414, abs=1e-3)
        assert agg["rouge1"]["n"] == 2

    def test_single_result_std_is_zero(self):
        results = [{"rouge1": 0.75}]
        agg = aggregate_scores(results)
        assert agg["rouge1"]["std"] == pytest.approx(0.0)

    def test_sqi_key_excluded(self):
        results = [{"rouge1": 0.8, "sqi": 0.9}]
        agg = aggregate_scores(results)
        assert "sqi" not in agg

    def test_nan_values_ignored(self):
        results = [
            {"rouge1": 0.8},
            {"rouge1": float("nan")},
            {"rouge1": 0.6},
        ]
        agg = aggregate_scores(results)
        assert agg["rouge1"]["n"] == 2
        assert agg["rouge1"]["mean"] == pytest.approx(0.7)

    def test_all_nan_key_omitted(self):
        results = [{"rouge1": float("nan")}, {"rouge1": float("nan")}]
        agg = aggregate_scores(results)
        assert "rouge1" not in agg

    def test_non_numeric_keys_skipped(self):
        results = [{"rouge1": 0.9, "label": "case_001", "candidate": "some text"}]
        agg = aggregate_scores(results)
        assert "label" not in agg
        assert "candidate" not in agg
        assert "rouge1" in agg

    def test_n_matches_non_nan_count(self):
        results = [{"v": 1.0}, {"v": 2.0}, {"v": float("nan")}, {"v": 3.0}]
        agg = aggregate_scores(results)
        assert agg["v"]["n"] == 3


# ---------------------------------------------------------------------------
# evaluate_gold_set (mocked LLM)
# ---------------------------------------------------------------------------

class TestEvaluateGoldSet:
    def _write_gold(self, tmp_path, cases):
        p = tmp_path / "gold_set.json"
        p.write_text(json.dumps(cases))
        return p

    def _fake_case(self, case_id: str, sqi: float = 0.85):
        return {
            "id": case_id,
            "label": f"Test case {case_id}",
            "features": {
                "hr_bpm": 65.0, "sdnn_ms": 45.0, "rmssd_ms": 38.0,
                "pnn50_pct": 22.0, "sqi": sqi,
                "n_beats": 65, "recording_duration_s": 60.0, "rr_ms": [],
            },
            "reference": "The heart rate of 65 bpm is within the normal resting range.",
        }

    def _mock_llm(self, text="Mock interpretation of the signal."):
        return (text, {"input_tokens": 200, "output_tokens": 100,
                       "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})

    def test_empty_gold_returns_error(self, tmp_path):
        p = self._write_gold(tmp_path, [])
        result = evaluate_gold_set(p, api_key="fake", skip_bert=True)
        assert "error" in result

    def test_result_structure_keys(self, tmp_path):
        cases = [self._fake_case("c1"), self._fake_case("c2")]
        p = self._write_gold(tmp_path, cases)
        with patch("src.llm_client.get_interpretation", return_value=self._mock_llm()):
            report = evaluate_gold_set(p, api_key="fake", skip_bert=True)
        assert {"results", "passing", "flagged", "aggregate_all", "aggregate_passing",
                "n_total", "n_passing", "n_flagged"} <= report.keys()

    def test_n_total_matches_gold_length(self, tmp_path):
        cases = [self._fake_case(f"c{i}") for i in range(3)]
        p = self._write_gold(tmp_path, cases)
        with patch("src.llm_client.get_interpretation", return_value=self._mock_llm()):
            report = evaluate_gold_set(p, api_key="fake", skip_bert=True)
        assert report["n_total"] == 3

    def test_sqi_gate_applied(self, tmp_path):
        cases = [self._fake_case("high", sqi=0.9), self._fake_case("low", sqi=0.2)]
        p = self._write_gold(tmp_path, cases)
        with patch("src.llm_client.get_interpretation", return_value=self._mock_llm()):
            report = evaluate_gold_set(p, api_key="fake", skip_bert=True, sqi_threshold=0.40)
        assert report["n_passing"] == 1
        assert report["n_flagged"] == 1

    def test_rouge_scores_present_in_results(self, tmp_path):
        cases = [self._fake_case("c1")]
        p = self._write_gold(tmp_path, cases)
        with patch("src.llm_client.get_interpretation", return_value=self._mock_llm()):
            report = evaluate_gold_set(p, api_key="fake", skip_bert=True)
        result = report["results"][0]
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result

    def test_self_score_rouge_perfect_when_candidate_equals_reference(self, tmp_path):
        ref = "The heart rate of 65 bpm is within the normal resting range."
        cases = [self._fake_case("c1")]
        cases[0]["reference"] = ref
        p = self._write_gold(tmp_path, cases)
        with patch("src.llm_client.get_interpretation", return_value=(ref, {"input_tokens": 10, "output_tokens": 10, "cache_read_input_tokens": 0})):
            report = evaluate_gold_set(p, api_key="fake", skip_bert=True)
        assert report["results"][0]["rouge1"] == pytest.approx(1.0)

    def test_progress_callback_called(self, tmp_path):
        cases = [self._fake_case(f"c{i}") for i in range(3)]
        p = self._write_gold(tmp_path, cases)
        calls = []
        with patch("src.llm_client.get_interpretation", return_value=self._mock_llm()):
            evaluate_gold_set(p, api_key="fake", skip_bert=True,
                              progress_callback=lambda done, total: calls.append((done, total)))
        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_latency_and_token_fields_in_results(self, tmp_path):
        cases = [self._fake_case("c1")]
        p = self._write_gold(tmp_path, cases)
        with patch("src.llm_client.get_interpretation", return_value=self._mock_llm()):
            report = evaluate_gold_set(p, api_key="fake", skip_bert=True)
        r = report["results"][0]
        assert "latency_s" in r
        assert "input_tokens" in r
        assert "output_tokens" in r
