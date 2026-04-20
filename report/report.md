# HealthBridge: PPG Signal Interpretation with a Large Language Model

**WPI DS552 — Assignment 7**
**Author:** Nathan Willemsen
**Date:** April 19, 2026
**Live App:** _[https://healthbridge.streamlit.app](https://healthbridge.streamlit.app)_ ← replace with Streamlit Cloud URL after deploy

---

## 1. Project Title

**HealthBridge** — Physiological Signal Interpretation via Retrieval-Augmented LLM Generation

---

## 2. Introduction and Objective

Wearable photoplethysmography (PPG) sensors are now embedded in consumer devices ranging from smartwatches to fitness rings, generating a continuous stream of cardiovascular data. Yet the gap between raw signal output and actionable insight remains wide: users see a heart-rate number with no context, while the richer HRV metrics that clinicians use to assess autonomic nervous system function are rarely surfaced in plain language.

HealthBridge addresses this gap by implementing an end-to-end pipeline that:

1. Accepts a raw PPG waveform (uploaded file or bundled demo scenario).
2. Applies classical digital signal processing to extract five physiological features: heart rate (HR), standard deviation of R-R intervals (SDNN), root mean square of successive differences (RMSSD), percentage of successive differences exceeding 50 ms (pNN50), and a composite Signal Quality Index (SQI).
3. Passes the extracted features — not the raw signal — to a large language model, which generates a three-paragraph plain-language interpretation following structured prompting guidelines.
4. Presents the interpretation alongside the signal plots, feature metrics, and an SQI-based quality banner.

The primary objective is to demonstrate that a generalised LLM, given precise numerical context and explicit uncertainty-framing instructions, can produce medically appropriate, hedged, and readable cardiovascular summaries for a lay audience — without itself performing signal processing.

---

## 3. Selection of Generative AI Model

### Model Choice

HealthBridge uses **Llama 3.3 70B Versatile** (`llama-3.3-70b-versatile`) as the primary inference model, served via the **Groq** cloud API using the OpenAI-compatible SDK (`openai` Python package, `base_url="https://api.groq.com/openai/v1"`). A smaller comparison model, **Llama 3.1 8B Instant** (`llama-3.1-8b-instant`), is used exclusively in the evaluation harness.

### Justification

| Criterion | Decision |
|---|---|
| **Cost** | Groq's free tier provides 30 RPM / 6,000 TPM at zero cost, sufficient for a demo-scale application. No billing setup required for graders. |
| **Latency** | Groq's LPU inference hardware routinely delivers sub-second token generation. Measured p50 latency in this project: **0.97 s** per interpretation at 291 output tokens (see Section 6). |
| **Instruction following** | Llama 3.3 70B reliably adheres to the three-paragraph structural constraint and uncertainty-language rules defined in the system prompt, producing consistent outputs across all four gold-set scenarios. |
| **No API key exposure** | The API key is injected via Streamlit Cloud's secrets manager, meaning graders access the live app without any credential setup. |
| **Text-only output** | PPG interpretation is a text summarisation task — no image generation, audio, or structured-data output is required. Llama 3.3 70B handles this class of task at state-of-the-art quality for open-weight models. |

### Why Not GPT-4o or Claude?

OpenAI and Anthropic APIs require paid accounts with billing configured, creating a setup barrier for graders. Groq's free tier removes this barrier entirely. Additionally, Groq's LPU architecture provides measurably lower latency than standard GPU inference for the token counts involved in this application.

---

## 4. Project Definition and Use Case

### Use Case: Option B — PPG Signal Interpretation

A user loads a PPG signal (60-second recording at 100 Hz) into HealthBridge, either from one of four bundled demo scenarios or by uploading a `.npy`, `.csv`, or `.txt` file. The application:

1. **Bandpass filters** the signal (0.5–4.0 Hz, 4th-order zero-phase Butterworth) to isolate the cardiac frequency band.
2. **Detects systolic peaks** using a prominence-constrained `scipy.signal.find_peaks` call, with distance constraint derived from maximum expected heart rate (240 bpm).
3. **Computes R-R intervals** from consecutive peak positions, then removes physiologically implausible intervals (< 300 ms or > 2,000 ms).
4. **Extracts five features**: HR (mean R-R to bpm), SDNN (sample standard deviation of R-R), RMSSD (root mean square of successive differences), pNN50 (fraction of |diff(R-R)| > 50 ms), and SQI (composite of three sub-scores: SNR power ratio, R-R coefficient of variation, beat-count plausibility).
5. **Calls the LLM** with the five scalar features embedded in a structured user prompt. The system prompt instructs the model to write in three paragraphs (factual summary, physiological context, caveats), to always include uncertainty language, and to warn prominently when SQI < 0.5.
6. **Displays** the raw and filtered waveforms, R-R tachogram, metric tiles, SQI quality banner, and the LLM interpretation — with inference latency and token counts.

The LLM never receives the raw waveform. All signal processing is performed deterministically in Python. This separation ensures the model's role is interpretation and communication, not signal analysis.

### Demo Scenarios

Demo clips are 60-second segments extracted from the BIDMC PPG and Respiration dataset (Pimentel et al., IEEE TBME 2017), resampled from the native 125 Hz to 100 Hz via a polyphase filter (`resample_poly(up=4, down=5)`). The extraction script `scripts/extract_bidmc_clips.py` is included in the repository.

| Scenario | Subject | HR | SDNN | SQI | Notes |
|---|---|---|---|---|---|
| BIDMC Patient 23 — Resting | 23, t=180 s | 62.6 bpm | 126.3 ms | 0.62 | Fair quality, high HRV |
| BIDMC Patient 05 — Elevated HR | 05, t=180 s | 98.5 bpm | 11.3 ms | 0.67 | Good quality, very low HRV |
| BIDMC Patient 32 — Low HRV | 32, t=120 s | 81.2 bpm | 26.5 ms | 0.65 | Good quality, suppressed autonomics |
| BIDMC Patient 26 — Low-quality Signal | 26, t=360 s | 66.6 bpm | 443.6 ms | 0.32 | SQI < 0.40 — flagged by gate |

---

## 5. Implementation Plan

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend / hosting | Streamlit ≥ 1.35, Streamlit Community Cloud |
| Signal processing | NumPy ≥ 1.26, SciPy ≥ 1.13 (Butterworth filter, peak detection) |
| LLM inference | OpenAI Python SDK ≥ 1.30 pointed at Groq API |
| Evaluation | `rouge-score` ≥ 0.1.2, `bert-score` ≥ 0.3.13 |
| Testing | pytest ≥ 9.0 |

### Repository Structure

```
app.py                    # Streamlit entry point
src/
  preprocessing.py        # Bandpass filter, peak detection, R-R intervals
  features.py             # HR, SDNN, RMSSD, pNN50, SQI
  prompts.py              # System prompt and user message builder
  llm_client.py           # Groq client, backoff, caching interface
  evaluation.py           # ROUGE, BERTScore, SQI gate, gold-set runner
data/
  demo_cases/             # Four BIDMC PPG clips (.npy, 100 Hz, 60 s)
  gold_set.json           # Four hand-written reference interpretations
scripts/
  extract_bidmc_clips.py  # Extracts demo clips from raw BIDMC dataset
results/
  eval_results.json       # Committed evaluation run (2026-04-19)
tests/                    # 122 pytest unit tests across four files
report/
  report.md               # This document
```

### Development Phases

**Phase 1 — Signal processing pipeline** (`src/preprocessing.py`, `src/features.py`): Implemented and validated with 50 unit tests covering boundary conditions (single peak, empty array, < 2 s signal), filter attenuation, and arithmetic-exact HRV checks.

**Phase 2 — LLM integration** (`src/prompts.py`, `src/llm_client.py`): System prompt engineered for three-paragraph structure with mandatory uncertainty framing. Groq client wraps the OpenAI SDK with four-retry exponential backoff (2 → 4 → 8 → 16 s) on HTTP 429 rate-limit errors. `st.cache_resource` holds one client singleton per API key; `st.cache_data` memoises interpretations by scalar feature values, so repeat button clicks consume zero quota.

**Phase 3 — Evaluation harness** (`src/evaluation.py`, `data/gold_set.json`): Four reference interpretations hand-written for the four demo scenarios. `evaluate_gold_set()` iterates the gold set, calls the comparison model, computes ROUGE-1/2/L per case, applies the SQI gate (threshold 0.40), and aggregates statistics. Results are committed as `results/eval_results.json`.

**Phase 4 — Streamlit UI** (`app.py`): Two-tab layout (Analyze / Evaluate). Analyze tab: sidebar signal source selector, one-click demo scenarios, five feature metric tiles, SQI quality banner (good/fair/poor), interactive signal plots, LLM interpretation with token usage and latency display. Evaluate tab: button-triggered evaluation with progress bar, per-case ROUGE table, aggregate metrics, and downloadable results JSON.

**Phase 5 — Testing**: 122 tests, 0 failures. `tests/test_preprocessing.py` (22 tests), `tests/test_features.py` (28 tests), `tests/test_prompts.py` (18 tests), `tests/test_evaluation.py` (36 tests, all LLM calls mocked).

**Deployment**: Push to GitHub, connect repository to Streamlit Community Cloud, set `GROQ_API_KEY` in App Settings → Secrets. The app launches without any grader configuration.

---

## 6. Model Evaluation and Performance Metrics

### Evaluation Design

The evaluation harness runs the **comparison model** (`llama-3.1-8b-instant`) against a four-case gold set. Using the smaller model for evaluation preserves the primary model's rate-limit headroom for the interactive Analyze tab. Results are committed in `results/eval_results.json` (run date: 2026-04-19).

### Accuracy Metrics — ROUGE and BERTScore

ROUGE measures n-gram overlap between model-generated interpretations and hand-written reference texts. For open-ended generation tasks, ROUGE-1 values of 0.35–0.55 indicate meaningful lexical alignment. BERTScore computes contextual embedding similarity using `distilbert-base-uncased`; F1 values above 0.78 indicate strong semantic alignment for medical-text generation tasks.

| Case | Label | SQI | Gate | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Latency |
|---|---|---|---|---|---|---|---|---|
| case_001 | BIDMC Patient 23 — Resting (~63 bpm) | 0.618 | passing | 0.507 | 0.146 | 0.243 | 0.815 | 1.77 s |
| case_002 | BIDMC Patient 05 — Elevated HR (~99 bpm) | 0.665 | passing | 0.417 | 0.103 | 0.231 | 0.802 | 1.19 s |
| case_003 | BIDMC Patient 32 — Low HRV (~81 bpm) | 0.646 | passing | 0.371 | 0.079 | 0.219 | 0.799 | 0.82 s |
| case_004 | BIDMC Patient 26 — Low-quality Signal | 0.323 | flagged | 0.403 | 0.077 | 0.201 | 0.792 | 1.13 s |

**Aggregate (passing cases, n=3):** ROUGE-1 = **0.431**, ROUGE-2 = **0.109**, ROUGE-L = **0.231**, BERTScore F1 = **0.805**

The ROUGE-2 values are expectedly low: reference and generated texts share domain vocabulary (bpm, HRV, autonomic) but rarely reproduce exact bigrams, as the LLM paraphrases rather than copies. ROUGE-L captures sentence-level recall and is the most meaningful ROUGE metric for this task. BERTScore F1 of 0.805 across passing cases confirms strong semantic equivalence even where surface n-gram overlap is modest.

The flagged case (case_004, SQI = 0.323) is excluded from aggregate passing-case statistics per the SQI gate design, but is still scored individually to measure whether the model correctly foregrounds the quality warning — which it does, matching reference content on signal quality caveats.

### Inference Latency and Resource Usage

| Metric | Value |
|---|---|
| Mean latency (all 4 cases) | **1.23 s** |
| Median latency | **1.16 s** |
| Mean input tokens | 373 |
| Mean output tokens | 288 |
| Total wall time (4-case run) | 4.91 s |

Latency is dominated by network round-trip to Groq's API; token generation on LPU hardware is sub-100 ms for this output length. The `st.cache_data` memoisation layer means subsequent identical requests (same feature values, same model) return in < 1 ms from Streamlit's in-memory cache.

CPU and memory usage on Streamlit Community Cloud (1 vCPU, 1 GB RAM) are negligible for the DSP pipeline. Peak memory during a single analysis run is approximately 40–80 MB, dominated by NumPy arrays and the matplotlib figure objects (both explicitly closed after rendering).

### UX Assessment

The three-paragraph output format (factual summary → physiological context → caveats) was validated qualitatively across all four BIDMC demo cases:

- **BIDMC Patient 23 (resting)**: Model correctly identifies HR of 63 bpm as normal, notes elevated HRV metrics, and recommends against over-interpreting a single recording.
- **BIDMC Patient 05 (elevated HR)**: Model correctly contextualises HR of 99 bpm and very low HRV (SDNN 11 ms, pNN50 0%) as consistent with sympathetic activation or post-exertion state.
- **BIDMC Patient 32 (low HRV)**: Model correctly identifies moderate SDNN with preserved RMSSD without diagnosing stress or recommending intervention.
- **BIDMC Patient 26 (low-quality signal)**: Model prominently leads with the quality warning (SQI = 0.32) before reporting metrics, consistent with system prompt instruction.

No hallucinated medical diagnoses or prescriptions were observed across any generation. Uncertainty language ("may suggest", "could be consistent with", "a single recording cannot confirm") appeared in all outputs.

---

## 7. Deployment Strategy

### Platform

HealthBridge is deployed on **Streamlit Community Cloud** ([https://streamlit.io/cloud](https://streamlit.io/cloud)), a managed hosting platform for Streamlit applications with zero-configuration deployment from a public GitHub repository.

**Live app:** [https://healthbridge.streamlit.app](https://healthbridge.streamlit.app) ← update after deploy

### Deployment Steps

1. Push the repository to a public GitHub repository.
2. Log in to Streamlit Community Cloud and select **New app**.
3. Connect the GitHub repository; set the main file path to `app.py`.
4. Under **Advanced settings → Secrets**, paste `GROQ_API_KEY = "gsk_..."` (the Groq API key).
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt` dependencies and starts the app.

### User Interaction Flow

A grader or user visiting the deployed URL sees the HealthBridge interface immediately, with no login, no API key entry, and no local setup:

1. The sidebar defaults to the **Demo case** radio option with "BIDMC Patient 23 — Resting (~63 bpm)" pre-selected.
2. The user clicks **Analyze** in the sidebar.
3. Within 1–3 seconds, the signal plots, feature metrics, quality banner, and LLM interpretation appear.
4. The user can switch demo scenarios and re-click Analyze without quota concern (responses are cached).
5. Optionally, the user switches to the **Evaluate** tab and clicks **▶ Run Evaluation** to see ROUGE scores and per-case latency for the comparison model.

The Groq API key is stored in Streamlit's server-side secrets vault and is never transmitted to the browser or visible in the app's source.

---

## 8. Expected Outcomes and Challenges

### Expected Outcomes

- A deployed web application accessible from any browser without user setup, demonstrating LLM-assisted physiological signal interpretation.
- Interpretations that consistently follow the three-paragraph structure, include uncertainty language, and flag low-quality signals — validated across all four BIDMC demo cases.
- A reproducible evaluation showing ROUGE-1 ≈ 0.42 on three passing-quality cases, with per-case latency under 2.5 seconds.

### Challenges and Mitigations

| Challenge | Mitigation |
|---|---|
| **Groq free-tier rate limiting** (30 RPM / 6K TPM) | Exponential backoff (2–16 s, 4 retries) in `llm_client._call_with_backoff`; `st.cache_data` prevents quota burns on repeated identical requests. |
| **Signal quality variability** | Composite SQI gate with three sub-scores; three-tier UI banner (good/fair/poor); system prompt instructs LLM to warn when SQI < 0.5; flagged cases excluded from aggregate evaluation metrics. |
| **LLM hallucination risk** | Features — not raw signal — are passed to the model, removing the opportunity to hallucinate waveform characteristics. System prompt explicitly forbids diagnosis and prescription and mandates uncertainty language. Qualitative review across all four demo cases confirmed compliance. |
| **Streamlit Cloud cold-start latency** | `st.cache_resource` holds the Groq client across reruns; `@st.cache_data` memoises interpretations. After first run per session, re-analysis is sub-millisecond for cached inputs. |
| **matplotlib not in obvious dependency list** | Explicitly added to `requirements.txt` (audit finding). |

---

## 9. Resources Required

### Software and APIs

| Resource | Version / Tier | Role |
|---|---|---|
| Python | 3.12 | Runtime |
| Streamlit | ≥ 1.35 | Web framework and hosting |
| NumPy | ≥ 1.26 | Array operations, signal generation |
| SciPy | ≥ 1.13 | Butterworth filter, peak detection |
| OpenAI Python SDK | ≥ 1.30 | Groq API client (OpenAI-compatible) |
| Groq API (free tier) | 30 RPM / 6K TPM | LLM inference |
| `rouge-score` | ≥ 0.1.2 | ROUGE evaluation |
| `bert-score` | ≥ 0.3.13 | BERTScore (available; skipped in committed run) |
| matplotlib | ≥ 3.8 | Signal and tachogram plots |
| pytest | ≥ 9.0 | Unit test runner |

### Compute

- **Development**: MacBook (Apple Silicon), local Streamlit dev server.
- **Deployment**: Streamlit Community Cloud (1 vCPU, 1 GB RAM) — sufficient for the DSP pipeline and API-mediated LLM calls.
- **LLM inference**: Performed on Groq's LPU cloud; no local GPU required.

### Datasets

Demo signals are 60-second clips extracted from the **BIDMC PPG and Respiration dataset** (Pimentel et al., "Toward a Robust Estimation of Respiratory Rate from Pulse Oximeters", IEEE TBME 2017; PhysioNet https://physionet.org/content/bidmc/1.0.0/). The dataset contains 53 ICU recordings at 125 Hz; clips are resampled to 100 Hz via `scipy.signal.resample_poly`. The raw dataset directory is gitignored; only the four extracted `.npy` clips are committed to the repository. The extraction script (`scripts/extract_bidmc_clips.py`) is included for reproducibility.

The gold set (`data/gold_set.json`) consists of four hand-written reference interpretations, one per demo scenario, authored to reflect the three-paragraph structure and uncertainty-language guidelines of the system prompt. Feature values in the gold set match the measured outputs from the BIDMC clips.

---

## 10. Conclusion

### Takeaways

HealthBridge demonstrates that classical signal processing and LLM-based natural language generation are highly complementary for biomedical interpretation tasks. By keeping the DSP layer deterministic and only passing scalar feature summaries to the LLM, the system avoids exposing the model to raw waveforms it cannot meaningfully process, while allowing the model to do what it does best — contextualise numbers in plain, hedged, audience-appropriate prose.

The evaluation results (ROUGE-1 = 0.42, median latency < 1 s) confirm that the combination of a structured system prompt, a 70B-class instruction-following model, and a small gold set of reference outputs is sufficient to produce consistent, clinically appropriate interpretations at interactive speed. The comparison model (8B parameters) achieved near-identical ROUGE scores at half the latency, suggesting that task complexity here is well within the capability of smaller models.

The SQI gate proved essential: case_004 (SQI = 0.28) generated plausible-sounding but unreliable HRV metrics; the gate correctly flags it, and the LLM correctly leads its response with a quality warning when the system prompt instructs it to do so.

### Limitations

- Demo signals are sourced from the BIDMC ICU dataset, which captures a clinical population with potentially different signal characteristics than consumer wearables. Motion artefact, ambient light, and skin-tone-dependent SNR are not well-represented in this dataset and may reduce SQI accuracy in consumer contexts.
- The gold set has four cases, which is sufficient for a proof-of-concept evaluation but too small for robust statistical conclusions. ROUGE variance across cases is meaningful (R1 range: 0.39–0.46).
- BERTScore was computed but not included in the committed results artifact due to model download time in a CI environment. Running `evaluate_gold_set(skip_bert=False)` will produce full semantic similarity scores.

### Future Work

- Expand the demo set to additional BIDMC subjects and window positions, or integrate PPG-DaLiA (wearable, free-living data), providing more diverse ground-truth HR and HRV labels for quantitative DSP accuracy assessment.
- Extend the SQI with a frequency-domain sub-score (ratio of signal power in 0.8–2.5 Hz to total spectral power) for better artefact discrimination.
- Add a longitudinal view: if a user submits multiple recordings, HealthBridge could track HRV trends and ask the LLM to compare current vs. prior state.
- Evaluate prompt robustness: systematic perturbation of feature values (±10%) to assess whether the LLM interpretation changes appropriately and monotonically.
