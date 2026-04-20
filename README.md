# HealthBridge

WPI DS552 Assignment 7 — PPG Signal Interpretation with LLM

**Live app:** [https://healthbridge-ds552.streamlit.app](https://healthbridge-ds552.streamlit.app)

HealthBridge accepts a raw photoplethysmography (PPG) signal, extracts physiological features (heart rate, HRV, signal quality index) using classical DSP, and uses **Llama 3.3 70B** (via Groq) to generate a plain-language interpretation with appropriate uncertainty caveats.

## Architecture

```
app.py                        # Streamlit UI entry point (Analyze + Evaluate tabs)
src/
  preprocessing.py            # Bandpass filter, peak detection, R-R intervals
  features.py                 # HR, HRV (SDNN, RMSSD, pNN50), SQI
  prompts.py                  # Prompt templates
  llm_client.py               # Groq client (OpenAI SDK) with rate-limit backoff
  evaluation.py               # ROUGE / BERTScore / SQI-gate evaluation
data/
  demo_cases/                 # Real BIDMC PPG clips (.npy, 100 Hz, 60 s)
  gold_set.json               # Reference interpretations for evaluation
scripts/
  extract_bidmc_clips.py      # Extracts demo clips from raw BIDMC dataset
results/
  eval_results.json           # Committed evaluation run (2026-04-19)
tests/                        # 122 pytest unit tests
report/
  report.md                   # Assignment writeup
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# Add your GROQ_API_KEY to secrets.toml
```

## Run locally

```bash
streamlit run app.py
```

## Deploy

Push to GitHub and connect to [Streamlit Community Cloud](https://streamlit.io/cloud).
Set `GROQ_API_KEY` in the app's **Settings → Secrets** panel. Graders need no local setup.

## Tests

```bash
pytest tests/ -v
```
