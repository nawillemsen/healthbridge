"""Prompt templates for the HealthBridge LLM interpretation layer."""

from __future__ import annotations

SYSTEM_PROMPT = """You are HealthBridge, a clinical-signal interpretation assistant embedded in a \
photoplethysmography (PPG) analysis tool. Your role is to translate numerical physiological \
features into clear, plain-language summaries that a non-specialist can understand.

Guidelines you must always follow:
1. Write for a lay audience — no unexplained clinical jargon.
2. Structure your response in three short paragraphs:
   • What the numbers show (factual summary of HR and HRV metrics).
   • What this may suggest (possible physiological context, e.g. rest vs. stress vs. exercise).
   • Caveats and next steps (limitations of a single PPG recording, reasons to consult a clinician).
3. Always include uncertainty language ("may suggest", "could indicate", "is consistent with").
4. Never diagnose, prescribe, or make definitive medical claims.
5. If signal quality is low (SQI < 0.5), prominently state that results should be interpreted \
with caution and may not reflect the user's true physiology.
6. Keep the total response under 250 words."""


def build_messages(features: dict) -> list[dict]:
    """Build the messages list for the Anthropic API call.

    The system prompt is returned as a top-level system string (handled separately
    in llm_client). This function returns the human turn only.
    """
    sqi = features["sqi"]
    sqi_str = f"{sqi:.2f}" if sqi == sqi else "unavailable"  # nan check

    quality_label = "poor" if (sqi != sqi or sqi < 0.4) else ("fair" if sqi < 0.65 else "good")

    user_text = f"""Please interpret the following PPG-derived physiological features for this recording.

Recording details:
  • Duration: {features['recording_duration_s']:.1f} seconds
  • Detected beats: {features['n_beats']}
  • Signal Quality Index (SQI): {sqi_str} ({quality_label})

Heart rate & HRV metrics:
  • Heart Rate (HR):  {_fmt(features['hr_bpm'])} bpm
  • SDNN:            {_fmt(features['sdnn_ms'])} ms
  • RMSSD:           {_fmt(features['rmssd_ms'])} ms
  • pNN50:           {_fmt(features['pnn50_pct'])} %

Provide a plain-language interpretation following your guidelines."""

    return [{"role": "user", "content": user_text}]


def _fmt(value: float, decimals: int = 1) -> str:
    if value != value:  # nan
        return "N/A"
    return f"{value:.{decimals}f}"
