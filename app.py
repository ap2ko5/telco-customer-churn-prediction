"""
app.py — Streamlit Dashboard for the Stacked Churn Intelligence System
=======================================================================
A web dashboard that lets non-technical business users upload a CSV,
run churn predictions, and explore results visually.

Run with:
  pip install streamlit
  streamlit run app.py

Features:
  - Tab 1  Overview    : KPI cards (total customers, % at risk, $ at risk)
  - Tab 2  Charts      : Risk band bar chart + probability histogram
  - Tab 3  Customers   : Sortable, filterable full prediction table
  - Tab 4  Importance  : SHAP feature importance image (if available)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── page config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Make src/ importable ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib

from config import (
    ESTIMATED_CONTRACT_MONTHS,
    FALLBACK_MONTHLY_CHARGES,
    MODELS_DIR,
    OUTPUTS_DIR,
    RISK_BANDS,
)
from data_loader import load_data
from preprocessor import transform as preprocess_transform
from xgb_model import predict_proba_xgb
from nn_model import predict_proba_nn
from stacking import stack_predict
from risk_segmentation import add_risk_band
from business_impact import compute_business_impact


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────

BAND_COLORS = {
    "Low":      "#27ae60",
    "Medium":   "#f1c40f",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}

CUSTOM_CSS = """
<style>
/* Dark sidebar */
[data-testid="stSidebar"] {
    background-color: #12151e;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label {
    color: #c8d3f5;
}

/* KPI metric cards */
.kpi-card {
    background: linear-gradient(135deg, #1e2235 0%, #252a3d 100%);
    border: 1px solid #2d3350;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.kpi-label  { color: #8892b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value  { color: #cdd6f4; font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
.kpi-sub    { color: #6e7491; font-size: 0.8rem; }

/* Risk badge pills */
.badge-Low      { background:#1a4731; color:#2ecc71; border-radius:6px; padding:2px 8px; }
.badge-Medium   { background:#4a3a10; color:#f1c40f; border-radius:6px; padding:2px 8px; }
.badge-High     { background:#4a2a10; color:#e67e22; border-radius:6px; padding:2px 8px; }
.badge-Critical { background:#4a1010; color:#e74c3c; border-radius:6px; padding:2px 8px; }

/* Tab styling */
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    """
    Load all four trained model artifacts from the models/ directory.
    Returns (preprocessor, xgb_model, nn_model, meta_model) or raises.
    """
    def find_latest(prefix: str) -> Path:
        matches = sorted(MODELS_DIR.glob(f"{prefix}_*.joblib"))
        if matches:
            return matches[-1]
        fallback = MODELS_DIR / f"{prefix}.joblib"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No artifact for '{prefix}' in {MODELS_DIR}")

    preproc_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    if not preproc_path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found.\nRun `train_pipeline.py` first."
        )

    preprocessor = joblib.load(preproc_path)
    xgb_model    = joblib.load(find_latest("xgb_model"))
    nn_model     = joblib.load(find_latest("nn_model"))
    meta_model   = joblib.load(find_latest("meta_model"))
    return preprocessor, xgb_model, nn_model, meta_model


@st.cache_data(show_spinner="Running predictions…")
def run_predictions(csv_bytes: bytes, target_col: str) -> pd.DataFrame:
    """
    Full prediction pipeline on an uploaded CSV:
      1. Load + clean data
      2. Preprocess with the trained pipeline
      3. XGBoost + NN → stacked probability
      4. Risk bands + business impact
    """
    import io
    # Write bytes to a temporary path (Streamlit UploadedFile is bytes-like)
    tmp_path = PROJECT_ROOT / "_tmp_upload.csv"
    tmp_path.write_bytes(csv_bytes)

    X, y, numeric_cols, categorical_cols = load_data(str(tmp_path), target_col=target_col)
    preprocessor, xgb_model, nn_model, meta_model = load_artifacts()

    X_t = preprocess_transform(preprocessor, X)

    p_xgb   = predict_proba_xgb(xgb_model, X_t)
    p_nn    = predict_proba_nn(nn_model, X_t)
    p_stack = stack_predict(meta_model, p_xgb, p_nn)

    result_df = X.copy().reset_index(drop=True)
    result_df["actual_churn"]     = y.values
    result_df["churn_probability"] = p_stack.round(4)
    result_df["xgb_prob"]         = p_xgb.round(4)
    result_df["nn_prob"]          = p_nn.round(4)

    result_df = add_risk_band(result_df)
    result_df = compute_business_impact(result_df)

    tmp_path.unlink(missing_ok=True)
    return result_df


def fmt_currency(val: float) -> str:
    """Format a dollar value with K/M suffix."""
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:,.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 Churn Intelligence")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        help="Upload the raw customer data CSV. The 'Churn' column is used as the label.",
    )
    target_col = st.text_input("Target Column Name", value="Churn")
    band_filter = st.multiselect(
        "Filter by Risk Band",
        options=["Low", "Medium", "High", "Critical"],
        default=["Low", "Medium", "High", "Critical"],
    )
    st.markdown("---")
    st.markdown("### ℹ About")
    st.caption(
        "This dashboard loads trained model artifacts from `models/`. "
        "Run `train_pipeline.py` first to generate them."
    )

    # Show pipeline info if available
    info_path = MODELS_DIR / "pipeline_info.json"
    if info_path.exists():
        with st.expander("📋 Last Training Info"):
            info = json.loads(info_path.read_text())
            st.json({
                "run_id":             info.get("run_id"),
                "calibration_method": info.get("calibration_method"),
                "n_features":         len(info.get("numeric_columns", [])) +
                                      len(info.get("categorical_columns", [])),
            })


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

st.title("Stacked Churn Intelligence Dashboard")
st.caption("XGBoost + Neural Network Stacking Ensemble · AI-augmented retention analytics")

# ── Check if models are available ─────────────────────────────────────────────
models_ready = (MODELS_DIR / "preprocessing_pipeline.joblib").exists()

if not models_ready:
    st.error(
        "⚠️ **No model artifacts found!**\n\n"
        "Please run the training pipeline first:\n"
        "```bash\n"
        "python src/train_pipeline.py --data data/customer_churn.csv --target Churn --no-ai\n"
        "```"
    )
    st.stop()

# ── No file uploaded yet ───────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload a customer CSV in the sidebar to begin analysis.")

    # Try to use the pre-computed predictions from outputs/
    pred_path = OUTPUTS_DIR / "churn_predictions.csv"
    if pred_path.exists():
        st.success(f"💾 Pre-computed predictions found at `{pred_path}`")
        if st.button("Load Pre-computed Predictions"):
            df = pd.read_csv(pred_path)
            st.session_state["df"] = df
            st.session_state["source"] = "pre-computed"
            st.rerun()
    st.stop()

# ── Process the uploaded file ─────────────────────────────────────────────────
try:
    csv_bytes = uploaded_file.read()
    with st.spinner("Running the stacked ensemble pipeline…"):
        df = run_predictions(csv_bytes, target_col)
    st.session_state["df"] = df
    st.session_state["source"] = "uploaded"
except Exception as e:
    st.error(f"❌ Error during prediction: {e}")
    st.stop()

# ── Use session state DF ───────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.stop()

df: pd.DataFrame = st.session_state["df"]

# Apply band filter
df_filtered = df[df["churn_band"].isin(band_filter)]


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_overview, tab_charts, tab_customers, tab_importance = st.tabs([
    "📊 Overview", "📈 Risk Charts", "👥 Customer Table", "🔍 Feature Importance"
])


# ────────────────────────────────────────────────────────
# Tab 1 — Overview KPIs
# ────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Pipeline Summary")

    total      = len(df)
    high_risk  = df["churn_band"].isin(["High", "Critical"]).sum()
    pct_risk   = high_risk / total * 100
    total_risk = df["expected_revenue_loss"].sum()
    avg_prob   = df["churn_probability"].mean()

    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        (col1, "Total Customers",   f"{total:,}",           "in dataset"),
        (col2, "High/Critical Risk", f"{high_risk:,}",      f"{pct_risk:.1f}% of total"),
        (col3, "Total Revenue at Risk", fmt_currency(total_risk), "expected loss"),
        (col4, "Avg Churn Prob",    f"{avg_prob:.1%}",      "across all customers"),
    ]
    for col, label, value, sub in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Band Distribution")
    band_counts = df["churn_band"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"]
    ).fillna(0).astype(int)

    band_df = pd.DataFrame({
        "Band":  band_counts.index,
        "Count": band_counts.values,
        "Pct":   (band_counts.values / total * 100).round(1),
    })
    for _, row in band_df.iterrows():
        color = BAND_COLORS.get(row["Band"], "#ccc")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
            f'<div style="width:120px;font-weight:600;color:{color};">{row["Band"]}</div>'
            f'<div style="background:{color};height:20px;border-radius:4px;'
            f'width:{max(int(row["Pct"] * 4), 4)}px;opacity:0.85;"></div>'
            f'<div style="color:#8892b0;">{row["Count"]:,} customers ({row["Pct"]}%)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Top 10 at-risk customers
    st.markdown("---")
    st.subheader("🔴 Top 10 Highest Revenue Risk Customers")
    top10_cols = ["churn_probability", "churn_band", "expected_revenue_loss"]
    if "tenure" in df.columns:
        top10_cols = ["tenure"] + top10_cols
    if "MonthlyCharges" in df.columns:
        top10_cols = top10_cols + ["MonthlyCharges"]
    available = [c for c in top10_cols if c in df.columns]
    st.dataframe(
        df.sort_values("expected_revenue_loss", ascending=False)
          .head(10)[available]
          .style.background_gradient(subset=["churn_probability"], cmap="Reds")
          .format({"churn_probability": "{:.1%}", "expected_revenue_loss": "${:,.2f}"}),
        use_container_width=True,
    )


# ────────────────────────────────────────────────────────
# Tab 2 — Risk Charts (using Streamlit native charts)
# ────────────────────────────────────────────────────────
with tab_charts:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Band Distribution")
        band_chart = pd.DataFrame({
            "Customers": df["churn_band"].value_counts().reindex(
                ["Low", "Medium", "High", "Critical"]
            ).fillna(0).astype(int)
        })
        st.bar_chart(band_chart)

    with c2:
        st.subheader("Churn Probability Distribution")
        hist_data = pd.DataFrame({"Churn Probability": df["churn_probability"]})
        st.bar_chart(hist_data["Churn Probability"].value_counts(bins=30).sort_index())

    # SHAP image from outputs/ (if available)
    shap_path = OUTPUTS_DIR / "shap_importance.png"
    if shap_path.exists():
        st.subheader("SHAP Feature Importance")
        st.image(str(shap_path), use_container_width=True)

    prob_dist_path = OUTPUTS_DIR / "prob_distribution.png"
    if prob_dist_path.exists():
        st.subheader("Probability Distribution (from last pipeline run)")
        st.image(str(prob_dist_path), use_container_width=True)


# ────────────────────────────────────────────────────────
# Tab 3 — Full Customer Table
# ────────────────────────────────────────────────────────
with tab_customers:
    st.subheader(f"Customer Predictions ({len(df_filtered):,} customers)")

    # Search bar
    search = st.text_input("🔍 Search any column value", "")
    display_df = df_filtered.copy()
    if search:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    # Column selector
    all_cols = display_df.columns.tolist()
    priority = ["churn_probability", "churn_band", "expected_revenue_loss",
                "tenure", "MonthlyCharges", "Contract", "InternetService"]
    default_show = [c for c in priority if c in all_cols][:8]
    chosen_cols = st.multiselect("Columns to show", all_cols, default=default_show)
    if not chosen_cols:
        chosen_cols = default_show

    st.dataframe(
        display_df[chosen_cols]
          .sort_values("churn_probability", ascending=False)
          .reset_index(drop=True)
          .style.format({
              "churn_probability":    "{:.1%}",
              "expected_revenue_loss": "${:,.2f}",
          }),
        use_container_width=True,
        height=500,
    )

    # Download button
    csv_out = display_df[chosen_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_out,
        file_name="churn_predictions_filtered.csv",
        mime="text/csv",
    )


# ────────────────────────────────────────────────────────
# Tab 4 — Feature Importance
# ────────────────────────────────────────────────────────
with tab_importance:
    st.subheader("Feature Importance (SHAP — XGBoost)")

    shap_path = OUTPUTS_DIR / "shap_importance.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP feature importance from last pipeline run", use_container_width=True)
        st.caption(
            "**How to read this chart:**\n\n"
            "Each bar shows how much that feature affects the churn prediction on average.\n"
            "Longer bar = more influential feature. "
            "Generated using TreeSHAP on the XGBoost base model."
        )
    else:
        st.info(
            "SHAP importance plot not found.\n\n"
            "Run `train_pipeline.py` first, which automatically generates "
            "`outputs/shap_importance.png`."
        )

    # Show pipeline info table
    info_path = MODELS_DIR / "pipeline_info.json"
    if info_path.exists():
        st.markdown("---")
        st.subheader("Pipeline Configuration (from last training run)")
        info = json.loads(info_path.read_text())
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**XGBoost Parameters**")
            st.json(info.get("xgb_params", {}))
        with col_b:
            st.markdown("**Neural Network Parameters**")
            st.json(info.get("nn_params", {}))
