"""HealthBridge — PPG Signal Interpretation via Groq (Llama 3.3 70B)."""

from __future__ import annotations

import io
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import streamlit as st

from src.evaluation import evaluate_gold_set
from src.features import extract_features
from src.llm_client import COMPARISON_MODEL, PRIMARY_MODEL, get_interpretation, make_client
from src.preprocessing import preprocess

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def _groq_client(api_key: str):
    """One Groq client per unique API key, reused across reruns."""
    return make_client(api_key)


@st.cache_data(show_spinner=False)
def _cached_interpretation(
    hr_bpm: float,
    sdnn_ms: float,
    rmssd_ms: float,
    pnn50_pct: float,
    sqi: float,
    n_beats: int,
    recording_duration_s: float,
    api_key: str,
    model: str,
) -> tuple[str, dict]:
    """Cache interpretation by feature values + model so repeat clicks are free."""
    feats = {
        "hr_bpm": hr_bpm, "sdnn_ms": sdnn_ms, "rmssd_ms": rmssd_ms,
        "pnn50_pct": pnn50_pct, "sqi": sqi, "n_beats": n_beats,
        "recording_duration_s": recording_duration_s,
        "rr_ms": np.array([]),
    }
    return get_interpretation(feats, api_key, model=model, client=_groq_client(api_key))


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HealthBridge",
    page_icon="HB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Demo clip loader
# ---------------------------------------------------------------------------

_DEMO_CLIPS: dict[str, str] = {
    "BIDMC Patient 23 — Resting (~63 bpm)":       "bidmc23_resting_63bpm.npy",
    "BIDMC Patient 05 — Elevated HR (~99 bpm)":   "bidmc05_elevated_99bpm.npy",
    "BIDMC Patient 32 — Low HRV (~81 bpm)":       "bidmc32_low_hrv_81bpm.npy",
    "BIDMC Patient 26 — Low-quality Signal":       "bidmc26_low_quality_67bpm.npy",
}
_DEMO_DIR = Path("data/demo_cases")


def _load_demo_clip(label: str) -> np.ndarray:
    path = _DEMO_DIR / _DEMO_CLIPS[label]
    return np.load(path).astype(np.float64)


# ---------------------------------------------------------------------------
# Signal parsing
# ---------------------------------------------------------------------------

def _parse_upload(file, fs: float) -> tuple[np.ndarray, str]:
    raw = file.read()
    name = file.name.lower()
    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(raw))
    elif name.endswith(".csv") or name.endswith(".txt"):
        try:
            arr = np.loadtxt(io.StringIO(raw.decode()), delimiter=",")
        except Exception as exc:
            return None, f"Could not parse file: {exc}"
        if arr.ndim > 1:
            arr = arr[:, 0]
    else:
        return None, "Unsupported file type. Upload a .npy, .csv, or .txt file."
    return arr.astype(np.float64).ravel(), None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_signal(raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray, fs: float) -> plt.Figure:
    t = np.arange(len(raw)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#c9d1d9")
        ax.spines[:].set_color("#30363d")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")

    axes[0].plot(t, raw, color="#58a6ff", linewidth=0.7, label="Raw PPG")
    axes[0].set_ylabel("Amplitude (a.u.)")
    axes[0].legend(loc="upper right", fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9")
    axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    axes[1].plot(t, filtered, color="#3fb950", linewidth=0.8, label="Filtered PPG")
    if len(peaks):
        axes[1].scatter(
            t[peaks], filtered[peaks],
            color="#f85149", s=18, zorder=5, label=f"Peaks ({len(peaks)})"
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].legend(loc="upper right", fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9")

    fig.tight_layout(pad=1.2)
    return fig


def _plot_rr(rr_ms: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 2.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#c9d1d9")
    ax.spines[:].set_color("#30363d")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")

    ax.plot(rr_ms, color="#d2a8ff", linewidth=1.0, marker="o", markersize=3)
    ax.axhline(np.mean(rr_ms), color="#f0883e", linewidth=1, linestyle="--",
               label=f"Mean {np.mean(rr_ms):.0f} ms")
    ax.set_xlabel("Beat index")
    ax.set_ylabel("R-R interval (ms)")
    ax.legend(fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9")
    fig.tight_layout(pad=1.0)
    return fig


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _fmt(val: float, decimals: int = 1) -> str:
    return "N/A" if math.isnan(val) else f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("HealthBridge")
    st.caption("PPG Signal Interpreter · DS552 Assignment 7")
    st.divider()

    source = st.radio("Signal source", ["Demo case", "Upload file"], index=0)

    if source == "Demo case":
        demo_choice = st.selectbox(
            "Select scenario",
            list(_DEMO_CLIPS.keys()),
        )
        fs = 100.0
        st.caption(f"Sampling rate: {fs:.0f} Hz · Duration: 60 s")
    else:
        uploaded = st.file_uploader(
            "Upload PPG signal",
            type=["npy", "csv", "txt"],
            help="Single-column CSV/TXT or NumPy .npy array of amplitude values.",
        )
        fs = st.number_input("Sampling rate (Hz)", min_value=10.0, max_value=2000.0,
                             value=100.0, step=10.0)

    st.divider()

    try:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("API key loaded from secrets.")
    except (KeyError, FileNotFoundError):
        api_key = st.text_input("Groq API key", type="password", placeholder="gsk_...")

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area — header + tabs
# ---------------------------------------------------------------------------

st.markdown("## HealthBridge — PPG Interpretation")
st.markdown(
    "Upload or select a PPG signal. HealthBridge extracts heart rate and HRV metrics using "
    "classical DSP, then asks **Llama 3.3 70B** (via Groq) to generate a plain-language "
    "interpretation with appropriate uncertainty caveats."
)
st.divider()

tab_analyze, tab_evaluate = st.tabs(["Analyze", "Evaluate"])

# ===========================================================================
# TAB 1 — ANALYZE
# ===========================================================================

with tab_analyze:
    if not analyze_btn:
        st.info("Configure your signal source in the sidebar and click **Analyze** to begin.",
)

    else:
        # --- Load signal ---
        raw = None
        signal_label = ""
        if source == "Demo case":
            raw = _load_demo_clip(demo_choice)
            signal_label = demo_choice
        else:
            if not uploaded:
                st.error("Please upload a file before clicking Analyze.")
            else:
                raw, err = _parse_upload(uploaded, fs)
                if err:
                    st.error(err)
                    raw = None
                signal_label = uploaded.name if uploaded else ""

        if raw is not None:
            # --- Preprocess ---
            with st.spinner("Preprocessing signal…"):
                try:
                    preproc = preprocess(raw, fs)
                except ValueError as exc:
                    st.error(f"Preprocessing failed: {exc}")
                    preproc = None

            if preproc is not None:
                feats = extract_features(preproc, raw, fs)
                rr_ms = feats["rr_ms"]

                # Signal plots
                with st.expander("Signal plots", expanded=True):
                    fig_sig = _plot_signal(raw, preproc["filtered"], preproc["peaks"], fs)
                    st.pyplot(fig_sig, use_container_width=True)
                    plt.close(fig_sig)

                    if len(rr_ms) >= 2:
                        st.caption("R-R interval tachogram")
                        fig_rr = _plot_rr(rr_ms)
                        st.pyplot(fig_rr, use_container_width=True)
                        plt.close(fig_rr)

                # Feature metrics
                st.markdown("### Extracted Features")
                col_hr, col_sdnn, col_rmssd, col_pnn50, col_sqi = st.columns(5)
                with col_hr:
                    st.metric("Heart Rate", f"{_fmt(feats['hr_bpm'])} bpm")
                with col_sdnn:
                    st.metric("SDNN", f"{_fmt(feats['sdnn_ms'])} ms")
                with col_rmssd:
                    st.metric("RMSSD", f"{_fmt(feats['rmssd_ms'])} ms")
                with col_pnn50:
                    st.metric("pNN50", f"{_fmt(feats['pnn50_pct'])} %")
                with col_sqi:
                    st.metric("Signal Quality (SQI)", _fmt(feats["sqi"], 2))

                sqi_val = feats["sqi"]
                if math.isnan(sqi_val) or sqi_val < 0.40:
                    st.warning(
                        "**Low signal quality (SQI < 0.40).** Peak detection may be unreliable. "
                        "The interpretation below should be treated with significant caution.",
                    )
                elif sqi_val < 0.65:
                    st.info(
                        f"**Fair signal quality (SQI = {sqi_val:.2f}).** "
                        "Results are usable but may contain noise artefacts.",
                    )
                else:
                    st.success(f"Good signal quality (SQI = {sqi_val:.2f}).")

                with st.expander("Recording details"):
                    st.write({
                        "Duration (s)": f"{feats['recording_duration_s']:.1f}",
                        "Detected beats": feats["n_beats"],
                        "Valid R-R intervals": len(rr_ms),
                        "Sampling rate (Hz)": fs,
                        "Signal label": signal_label,
                    })

                # LLM Interpretation
                st.divider()
                st.markdown("### Clinical Interpretation")

                if not api_key:
                    st.warning("Enter your Groq API key in the sidebar to generate an interpretation.")
                else:
                    with st.spinner("Asking Llama 3.3 70B via Groq…"):
                        t0 = time.perf_counter()
                        try:
                            interpretation, usage = _cached_interpretation(
                                hr_bpm=feats["hr_bpm"],
                                sdnn_ms=feats["sdnn_ms"],
                                rmssd_ms=feats["rmssd_ms"],
                                pnn50_pct=feats["pnn50_pct"],
                                sqi=feats["sqi"],
                                n_beats=feats["n_beats"],
                                recording_duration_s=feats["recording_duration_s"],
                                api_key=api_key,
                                model=PRIMARY_MODEL,
                            )
                            latency_s = time.perf_counter() - t0
                            ok = True
                        except Exception as exc:
                            st.error(f"LLM call failed: {exc}")
                            ok = False

                    if ok:
                        st.markdown(interpretation)
                        with st.expander("Token usage & latency"):
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Input tokens", usage["input_tokens"])
                            col_b.metric("Output tokens", usage["output_tokens"])
                            col_c.metric("Latency", f"{latency_s:.2f} s")


# ===========================================================================
# TAB 2 — EVALUATE
# ===========================================================================

with tab_evaluate:
    st.markdown("### Model Evaluation")
    st.markdown(
        "Runs the 4-case gold set through **`llama-3.1-8b-instant`** (comparison model), "
        "scores each output against hand-written reference interpretations using ROUGE-1/2/L, "
        "and applies the SQI gate (threshold 0.40) to flag low-quality cases."
    )

    if not api_key:
        st.warning("Groq API key required. Set it in the sidebar to run evaluation.")
    else:
        run_eval_btn = st.button("Run Evaluation", type="primary")

        if run_eval_btn:
            progress_bar = st.progress(0, text="Starting…")
            status_placeholder = st.empty()

            def _progress(done: int, total: int):
                frac = done / total
                progress_bar.progress(frac, text=f"Case {done}/{total}")
                status_placeholder.caption(f"Scored {done} of {total} cases…")

            with st.spinner("Running evaluation (llama-3.1-8b-instant)…"):
                try:
                    report = evaluate_gold_set(
                        gold_path="data/gold_set.json",
                        api_key=api_key,
                        sqi_threshold=0.40,
                        skip_bert=True,
                        llm_model=COMPARISON_MODEL,
                        progress_callback=_progress,
                    )
                    progress_bar.progress(1.0, text="Complete")
                    status_placeholder.empty()
                    eval_ok = True
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")
                    eval_ok = False

            if eval_ok:
                st.success(
                    f"Evaluated {report['n_total']} cases — "
                    f"{report['n_passing']} passing SQI gate, "
                    f"{report['n_flagged']} flagged.",
                )

                # Aggregate metrics
                st.markdown("#### Aggregate ROUGE (passing cases only)")
                agg = report["aggregate_passing"]
                if agg:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("ROUGE-1 F1", f"{agg.get('rouge1', {}).get('mean', 0):.3f}")
                    col2.metric("ROUGE-2 F1", f"{agg.get('rouge2', {}).get('mean', 0):.3f}")
                    col3.metric("ROUGE-L F1", f"{agg.get('rougeL', {}).get('mean', 0):.3f}")

                # Per-case table
                st.markdown("#### Per-case results")
                rows = []
                for r in report["results"]:
                    sqi_flag = "flagged" if r["sqi"] < 0.40 else "passing"
                    rows.append({
                        "ID": r["id"],
                        "Label": r["label"],
                        "SQI": f"{r['sqi']:.2f}",
                        "Gate": sqi_flag,
                        "ROUGE-1": f"{r.get('rouge1', 0):.3f}",
                        "ROUGE-2": f"{r.get('rouge2', 0):.3f}",
                        "ROUGE-L": f"{r.get('rougeL', 0):.3f}",
                        "Latency (s)": f"{r.get('latency_s', 0):.2f}",
                        "In tokens": r.get("input_tokens", 0),
                        "Out tokens": r.get("output_tokens", 0),
                    })
                st.dataframe(rows, use_container_width=True)

                # Candidate outputs
                with st.expander("Generated interpretations"):
                    for r in report["results"]:
                        st.markdown(f"**{r['id']} — {r['label']}**")
                        st.markdown(r["candidate"])
                        st.divider()

                # Raw JSON download
                import json as _json
                st.download_button(
                    "Download results JSON",
                    data=_json.dumps(report, indent=2, default=str),
                    file_name="eval_results.json",
                    mime="application/json",
                )
