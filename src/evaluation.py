"""Evaluation utilities: ROUGE, BERTScore, SQI gate, and gold-set runner."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer

# BERTScore is imported lazily — it triggers a model download on first use.
# Callers pass skip_bert=True to skip it when running in lightweight contexts.


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
_SCORER = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)


def score_rouge(candidate: str, reference: str) -> dict[str, float]:
    """Return ROUGE-1/2/L F1 scores for a single candidate/reference pair."""
    scores = _SCORER.score(reference, candidate)
    return {k: round(scores[k].fmeasure, 4) for k in _ROUGE_TYPES}


def score_rouge_batch(candidates: list[str], references: list[str]) -> list[dict[str, float]]:
    """ROUGE scores for parallel lists of candidates and references."""
    if len(candidates) != len(references):
        raise ValueError("candidates and references must have equal length.")
    return [score_rouge(c, r) for c, r in zip(candidates, references)]


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def score_bertscore(
    candidates: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    device: Optional[str] = None,
) -> list[dict[str, float]]:
    """BERTScore P/R/F1 for parallel lists.

    Uses distilbert-base-uncased for speed; swap model_type for a heavier
    model (e.g. roberta-large) if assignment rubric requires it.
    """
    from bert_score import score as _bert_score  # noqa: PLC0415

    kwargs: dict = dict(
        cands=candidates,
        refs=references,
        model_type=model_type,
        verbose=False,
    )
    if device is not None:
        kwargs["device"] = device

    P, R, F1 = _bert_score(**kwargs)
    return [
        {
            "bertscore_precision": round(float(p), 4),
            "bertscore_recall": round(float(r), 4),
            "bertscore_f1": round(float(f), 4),
        }
        for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist())
    ]


# ---------------------------------------------------------------------------
# SQI gate
# ---------------------------------------------------------------------------

def sqi_gate(
    results: list[dict],
    threshold: float = 0.40,
) -> tuple[list[dict], list[dict]]:
    """Split results into (passing, flagged) based on SQI threshold.

    A result dict is expected to contain a top-level 'sqi' key.
    NaN SQI is always flagged.
    """
    passing, flagged = [], []
    for r in results:
        sqi = r.get("sqi", float("nan"))
        if math.isnan(sqi) or sqi < threshold:
            flagged.append(r)
        else:
            passing.append(r)
    return passing, flagged


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_scores(results: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean ± std for every numeric metric key across results.

    Returns {metric_name: {mean: float, std: float, n: int}}.
    Skips non-numeric fields and ignores NaN values.
    """
    if not results:
        return {}

    # Collect all numeric keys from the first result that has them
    metric_keys = [
        k for k, v in results[0].items()
        if isinstance(v, (int, float)) and k != "sqi"
    ]

    agg: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        vals = [r[key] for r in results if key in r and not math.isnan(float(r[key]))]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        agg[key] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
            "n": len(arr),
        }
    return agg


# ---------------------------------------------------------------------------
# Gold-set evaluation runner
# ---------------------------------------------------------------------------

def evaluate_gold_set(
    gold_path: str | Path,
    api_key: str,
    sqi_threshold: float = 0.40,
    skip_bert: bool = False,
    bert_model: str = "distilbert-base-uncased",
    progress_callback=None,
    llm_model: str | None = None,
) -> dict:
    """Run the full evaluation pipeline against a gold set JSON file.

    Gold set schema (list of objects):
    [
      {
        "id": "case_001",
        "label": "Human-readable label",
        "features": { <extract_features output fields> },
        "reference": "Gold-standard interpretation text."
      }, ...
    ]

    Parameters
    ----------
    gold_path       Path to gold_set.json
    api_key         Groq API key
    sqi_threshold   Cases with SQI below this are flagged (still scored)
    skip_bert       If True, skip BERTScore (faster, no model download)
    bert_model      HuggingFace model ID for BERTScore
    progress_callback  Optional callable(done: int, total: int) for UI updates
    llm_model       Groq model ID; defaults to COMPARISON_MODEL

    Returns
    -------
    dict with keys:
        results         list[dict] — per-case scores + metadata
        passing         list[dict] — cases above SQI threshold
        flagged         list[dict] — low-SQI cases
        aggregate_all   aggregate_scores over all results
        aggregate_passing aggregate_scores over passing results only
        n_total, n_passing, n_flagged
    """
    from src.llm_client import COMPARISON_MODEL, get_interpretation, make_client  # noqa: PLC0415

    model = llm_model or COMPARISON_MODEL
    client = make_client(api_key)

    gold: list[dict] = json.loads(Path(gold_path).read_text())
    if not gold:
        return {"error": "Gold set is empty.", "results": []}

    candidates: list[str] = []
    results: list[dict] = []

    # --- Generate LLM candidates ---
    for i, case in enumerate(gold):
        feats = case["features"]
        # Ensure rr_ms is a numpy array if present (may be list in JSON)
        if "rr_ms" in feats:
            feats["rr_ms"] = np.array(feats["rr_ms"])

        t0 = time.perf_counter()
        text, usage = get_interpretation(feats, api_key, model=model, client=client)
        elapsed = time.perf_counter() - t0

        candidates.append(text)
        results.append({
            "id": case.get("id", f"case_{i:03d}"),
            "label": case.get("label", ""),
            "sqi": feats.get("sqi", float("nan")),
            "candidate": text,
            "reference": case["reference"],
            "latency_s": round(elapsed, 2),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        })

        if progress_callback:
            progress_callback(i + 1, len(gold))

    references = [r["reference"] for r in results]

    # --- ROUGE ---
    rouge_scores = score_rouge_batch(candidates, references)
    for r, rs in zip(results, rouge_scores):
        r.update(rs)

    # --- BERTScore ---
    if not skip_bert:
        bert_scores = score_bertscore(candidates, references, model_type=bert_model)
        for r, bs in zip(results, bert_scores):
            r.update(bs)

    # --- SQI gate ---
    passing, flagged = sqi_gate(results, threshold=sqi_threshold)

    return {
        "results": results,
        "passing": passing,
        "flagged": flagged,
        "aggregate_all": aggregate_scores(results),
        "aggregate_passing": aggregate_scores(passing),
        "n_total": len(results),
        "n_passing": len(passing),
        "n_flagged": len(flagged),
    }
