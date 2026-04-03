"""
app.py — Streamlit Dashboard for the Stacked Churn Intelligence System
=======================================================================
A web dashboard that lets non-technical business users upload a CSV,
run churn predictions, and explore results visually.

Run with:
  pip install streamlit
  streamlit run app.py

Features:
  - Tab 1  Overview    : KPI cards (total customers, % at risk, ₹ at risk)
  - Tab 2  Charts      : Risk band bar chart + probability histogram
  - Tab 3  Customers   : Sortable, filterable full prediction table
  - Tab 4  Importance  : SHAP feature importance image (if available)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── page config (must be FIRST Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).parent

import joblib

from src.config import (
    ESTIMATED_CONTRACT_MONTHS,
    FALLBACK_MONTHLY_CHARGES,
    MODELS_DIR,
    OUTPUTS_DIR,
    RISK_BANDS,
)
from src.data_loader import load_data
from src.preprocessor import transform as preprocess_transform
from src.xgb_model import predict_proba_xgb
from src.nn_model import predict_proba_nn
from src.stacking import stack_predict
from src.risk_segmentation import add_risk_band
from src.business_impact import compute_business_impact


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
    """Format a rupee value in Indian numbering system (Lakhs/Crores)."""
    if val >= 10_000_000:  # 1 crore or more
        return f"₹{val/10_000_000:.2f}Cr"
    elif val >= 100_000:  # 1 lakh or more
        return f"₹{val/100_000:.2f}L"
    elif val >= 1_000:
        return f"₹{_indian_grouping(str(int(val)))}"
    return f"₹{val:,.0f}"


def _indian_grouping(num_str: str) -> str:
    """Apply Indian digit grouping (rightmost 3, then groups of 2)."""
    num_str = num_str.replace(",", "")
    if len(num_str) <= 3:
        return num_str
    result = num_str[-3:]
    remaining = num_str[:-3]
    while remaining:
        if len(remaining) <= 2:
            result = remaining + "," + result
            break
        else:
            result = remaining[-2:] + "," + result
            remaining = remaining[:-2]
    return result


def format_indian_rupees(val: float) -> str:
    """Format value as Indian currency with proper grouping."""
    if pd.isna(val):
        return ""
    integer_part = int(val)
    decimal_part = val - integer_part
    formatted_int = _indian_grouping(str(integer_part))
    return f"₹{formatted_int}{decimal_part:.2f}"[:-3] if decimal_part == 0 else f"₹{formatted_int}.{int(decimal_part * 100):02d}"


def _is_http_url(value: str) -> bool:
        """Basic URL check for Power BI embed links."""
        return value.startswith("http://") or value.startswith("https://")


def render_powerbi_embed(
        embed_url: str,
        embed_mode: str,
        access_token: str,
        report_id: str,
        height_px: int,
) -> None:
        """
        Render Power BI report in Streamlit.

        - public mode: iframe (publish-to-web links)
        - secure mode: powerbi-client with Azure AD embed token
        """
        if not embed_url:
                st.info("Add a Power BI embed URL in the sidebar to display a live report.")
                return

        if not _is_http_url(embed_url):
                st.error("Power BI embed URL must start with http:// or https://")
                return

        if embed_mode == "public":
                components.iframe(embed_url, height=height_px, scrolling=True)
                return

        if not access_token:
                st.warning("Secure embed mode requires an access token. Add one in the sidebar or set PBI_ACCESS_TOKEN.")
                return

        safe_embed_url = json.dumps(embed_url)
        safe_access_token = json.dumps(access_token)
        safe_report_id = json.dumps(report_id.strip()) if report_id else '""'

        html = f"""
        <div id=\"pbi-report-container\" style=\"width: 100%; height: {height_px}px; border-radius: 12px; overflow: hidden;\"></div>
        <script src=\"https://cdn.jsdelivr.net/npm/powerbi-client@2.23.1/dist/powerbi.js\"></script>
        <script>
            const models = window['powerbi-client'].models;
            const config = {{
                type: 'report',
                tokenType: models.TokenType.Embed,
                accessToken: {safe_access_token},
                embedUrl: {safe_embed_url},
                id: {safe_report_id},
                permissions: models.Permissions.All,
                settings: {{
                    panes: {{
                        filters: {{ visible: false }},
                        pageNavigation: {{ visible: true }}
                    }},
                    background: models.BackgroundType.Transparent
                }}
            }};

            const container = document.getElementById('pbi-report-container');
            const powerbi = window.powerbi;
            powerbi.reset(container);
            powerbi.embed(container, config);
        </script>
        """
        components.html(html, height=height_px + 10, scrolling=False)


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
    st.markdown("### 📊 Power BI Integration")
    default_pbi_url = os.getenv("PBI_EMBED_URL", "")
    default_pbi_token = os.getenv("PBI_ACCESS_TOKEN", "")
    default_report_id = os.getenv("PBI_REPORT_ID", "")

    powerbi_embed_url = st.text_input(
        "Power BI Embed URL",
        value=default_pbi_url,
        help="Use either a Publish-to-web URL (public mode) or report embed URL (secure mode).",
    )
    powerbi_embed_mode = st.selectbox(
        "Embed Mode",
        options=["public", "secure"],
        index=0,
        help="Public uses iframe. Secure uses access token + Power BI JavaScript SDK.",
    )
    powerbi_report_id = st.text_input(
        "Report ID (optional)",
        value=default_report_id,
        help="Used for secure embedding. Leave empty if embedUrl already includes report context.",
    )
    powerbi_access_token = st.text_input(
        "Access Token (secure mode)",
        value=default_pbi_token,
        type="password",
    )
    powerbi_height = st.slider("Embed Height", min_value=420, max_value=1200, value=700, step=20)

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

st.markdown("### Executive Power BI View")
st.caption("Embed your business dashboard to combine executive BI tracking with churn model outputs in one app.")
render_powerbi_embed(
    embed_url=powerbi_embed_url.strip(),
    embed_mode=powerbi_embed_mode,
    access_token=powerbi_access_token.strip(),
    report_id=powerbi_report_id.strip(),
    height_px=powerbi_height,
)
st.markdown("---")

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
        "Pct":   ((band_counts.to_numpy(dtype=float) / max(total, 1)) * 100).round(1),
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
    
    # Apply Indian currency formatting
    display_top10 = df.sort_values("expected_revenue_loss", ascending=False).head(10)[available].copy()
    if "expected_revenue_loss" in display_top10.columns:
        display_top10["expected_revenue_loss"] = display_top10["expected_revenue_loss"].apply(
            lambda x: f"₹{_indian_grouping(str(int(x)))}.{int((x - int(x)) * 100):02d}" if pd.notna(x) else ""
        )
    
    st.dataframe(
        display_top10.style.background_gradient(
            subset=["churn_probability"] if "churn_probability" in display_top10.columns else [], 
            cmap="Reds"
        ).format({"churn_probability": "{:.1%}"} if "churn_probability" in display_top10.columns else {}),
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

    # Apply Indian currency formatting
    display_final = display_df[chosen_cols].sort_values("churn_probability", ascending=False).reset_index(drop=True).copy()
    if "expected_revenue_loss" in display_final.columns:
        display_final["expected_revenue_loss"] = display_final["expected_revenue_loss"].apply(
            lambda x: f"₹{_indian_grouping(str(int(x)))}.{int((x - int(x)) * 100):02d}" if pd.notna(x) else ""
        )
    
    st.dataframe(
        display_final.style.format({"churn_probability": "{:.1%}"} if "churn_probability" in display_final.columns else {}),
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
