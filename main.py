"""
main.py — AI-Data-Preprocessing
Streamlit UI: upload → clean → EDA → download.
"""

import streamlit as st
import pandas as pd
from io import BytesIO

from cleaning import run_full_cleaning
from eda import run_full_eda
from utils import (
    load_file,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    get_schema_summary,
    numeric_profile,
    format_cleaning_report,
)

# ── Page Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Data-Preprocessing",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #888; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .insight-card {
        background: #1e1e2e; border-left: 4px solid #6C63FF;
        border-radius: 8px; padding: 12px 16px;
        margin: 6px 0; font-size: 0.95rem; color: #e0e0e0;
    }
    .stat-box {
        background: #1e1e2e; border-radius: 10px;
        padding: 16px; text-align: center; margin: 4px;
    }
    .stat-number { font-size: 1.9rem; font-weight: 700; color: #6C63FF; }
    .stat-label  { font-size: 0.8rem; color: #aaa; margin-top: 2px; }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
    .stDownloadButton > button:hover { opacity: 0.88; }
    .section-header {
        font-size: 1.35rem; font-weight: 700;
        color: #6C63FF; margin: 1.2rem 0 0.5rem;
        border-bottom: 2px solid #2a2a3e; padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧹 AI-Data-Preprocessing</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload any CSV or Excel file — get back a fully cleaned dataset, '
    'EDA charts, and actionable insights instantly.</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    iqr_threshold = st.slider("IQR Outlier Threshold", 1.0, 3.0, 1.5, 0.25,
                               help="Lower = more aggressive outlier capping")
    max_cat_bars = st.slider("Max Categories in Bar Chart", 5, 30, 15)
    show_raw     = st.checkbox("Show raw data preview", value=True)
    show_schema  = st.checkbox("Show schema table", value=True)
    export_fmt   = st.radio("Download format", ["CSV", "Excel"], horizontal=True)

    st.markdown("---")
    st.markdown("**What gets cleaned:**")
    st.markdown("""
- ✅ Column names → snake_case  
- ✅ Duplicate rows removed  
- ✅ Auto data-type detection  
- ✅ Missing values (median / mode / ffill)  
- ✅ Outliers capped (IQR)  
- ✅ Categorical normalization  
""")

# ── File Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📁 Drop your CSV or Excel file here",
    type=["csv", "xlsx", "xls"],
    help="Supports comma, semicolon, tab-delimited CSVs and .xlsx / .xls files.",
)

if uploaded is None:
    st.info("👆 Upload a file to get started. No account needed, no data stored.")
    st.stop()


# ── Load + Clean ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def process_file(file_bytes: bytes, filename: str, iqr_thresh: float):
    buf = BytesIO(file_bytes)
    buf.name = filename
    df_raw = load_file(buf)
    df_clean, report = run_full_cleaning(df_raw.copy(), iqr_threshold=iqr_thresh)
    return df_raw, df_clean, report

@st.cache_data(show_spinner=False)
def cached_eda(df_raw, df_clean, report, shape):
    return run_full_eda(df_raw, df_clean, report)

with st.spinner("🔍 Loading and cleaning your data..."):
    file_bytes = uploaded.getvalue()
    try:
        df_raw, df_clean, report = process_file(file_bytes, uploaded.name, iqr_threshold)
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

# ── Top Stats Row ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-number">{df_raw.shape[0]:,}</div>
        <div class="stat-label">Raw Rows</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-number">{df_clean.shape[0]:,}</div>
        <div class="stat-label">Clean Rows</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="stat-box">
        <div class="stat-number">{df_raw.shape[1]}</div>
        <div class="stat-label">Columns</div></div>""", unsafe_allow_html=True)
with c4:
    nulls = int(df_raw.isnull().sum().sum())
    st.markdown(f"""<div class="stat-box">
        <div class="stat-number">{nulls:,}</div>
        <div class="stat-label">Nulls Found</div></div>""", unsafe_allow_html=True)
with c5:
    dupes = report.get("duplicates_removed", 0)
    st.markdown(f"""<div class="stat-box">
        <div class="stat-number">{dupes:,}</div>
        <div class="stat-label">Dupes Removed</div></div>""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────────
tab_raw, tab_clean, tab_report, tab_eda, tab_download, tab_code = st.tabs([
    "📋 Raw Data", "✨ Cleaned Data", "🧹 Cleaning Report", "📈 EDA & Charts", "⬇️ Download", "🧑‍💻 View Code"
])

# ─── Tab 1: Raw Data ──────────────────────────────────────────────────────────────
with tab_raw:
    st.markdown(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    if show_schema:
        st.markdown('<div class="section-header">Schema</div>', unsafe_allow_html=True)
        st.dataframe(get_schema_summary(df_raw), use_container_width=True)
    if show_raw:
        st.markdown('<div class="section-header">Preview (first 500 rows)</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(500), use_container_width=True)
    if df_raw.shape[0] > 50000:
        st.warning("Large dataset detected. Performance may slow.")
    
# ─── Tab 2: Cleaned Data ─────────────────────────────────────────────────────────
with tab_clean:
    st.markdown(f"**Shape:** {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    st.dataframe(df_clean.head(500), use_container_width=True)

    st.markdown('<div class="section-header">Numeric Profile</div>', unsafe_allow_html=True)
    np_df = numeric_profile(df_clean)
    if not np_df.empty:
        st.dataframe(np_df, use_container_width=True)
    else:
        st.info("No numeric columns found.")

# ─── Tab 3: Cleaning Report ───────────────────────────────────────────────────────
with tab_report:
    st.markdown(format_cleaning_report(report))

    with st.expander("🔍 Raw Report Dictionary"):
        for key, val in report.items():
            st.write(f"**{key}:**", val)

# ─── Tab 4: EDA & Charts ─────────────────────────────────────────────────────────
with tab_eda:
    with st.spinner("📊 Running EDA..."):
        
        
         eda = cached_eda(df_raw, df_clean, report,df_raw.shape)
    # Insights
    st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)
    for insight in eda["insights"]:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    # Summary Stats
    st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(eda["summary_stats"], use_container_width=True)

    # Null charts
    st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if eda["null_heatmap"]:
            st.image(eda["null_heatmap"], use_container_width=True)
        else:
            st.success("✅ No missing values in raw data.")
    with col_b:
        if eda["null_bar"]:
            st.image(eda["null_bar"], use_container_width=True)

    # Correlation
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    if eda["correlation_matrix"]:
        st.image(eda["correlation_matrix"], use_container_width=True)
    else:
        st.info("Need ≥ 2 numeric columns for correlation matrix.")

    # Boxplots
    st.markdown('<div class="section-header">Boxplots (Outlier View)</div>', unsafe_allow_html=True)
    if eda["boxplots"]:
        st.image(eda["boxplots"], use_container_width=True)

    # Distributions
    st.markdown('<div class="section-header">Distributions</div>', unsafe_allow_html=True)
    dist_charts = eda["distributions"]
    if dist_charts:
        cols_dist = st.columns(2)
        for i, (col_name, img) in enumerate(dist_charts.items()):
            with cols_dist[i % 2]:
                st.image(img, use_container_width=True)
    else:
        st.info("No numeric columns to plot.")

    # Categorical
    st.markdown('<div class="section-header">Categorical Value Counts</div>', unsafe_allow_html=True)
    cat_charts = eda["categorical_counts"]
    if cat_charts:
        cols_cat = st.columns(2)
        for i, (col_name, img) in enumerate(cat_charts.items()):
            with cols_cat[i % 2]:
                st.image(img, use_container_width=True)
    else:
        st.info("No categorical columns to plot.")

# ─── Tab 5: Download ─────────────────────────────────────────────────────────────
with tab_download:
    st.markdown('<div class="section-header">⬇️ Download Cleaned Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"File will contain **{df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns**.")

    base_name = uploaded.name.rsplit(".", 1)[0] + "_cleaned"

    if export_fmt == "CSV":
        st.download_button(
            label="⬇️ Download CSV",
            data=dataframe_to_csv_bytes(df_clean),
            file_name=f"{base_name}.csv",
            mime="text/csv",
        )
    else:
        st.download_button(
            label="⬇️ Download Excel",
            data=dataframe_to_excel_bytes(df_clean),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("---")
    st.markdown("**Cleaning Summary:**")
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Rows before | {df_raw.shape[0]:,} |
| Rows after | {df_clean.shape[0]:,} |
| Duplicates removed | {report.get('duplicates_removed', 0)} |
| Columns with nulls filled | {len(report.get('missing_values', {}))} |
| Columns with outliers capped | {len(report.get('outliers', {}))} |
| Dtype conversions | {len(report.get('dtype_changes', {}))} |
| Columns renamed | {len(report.get('column_names', []))} |
""")

# ─── Tab 6: View Code ─────────────────────────────────────────────────────────────
with tab_code:
    st.markdown('<div class="section-header">🧑‍💻 Code Used To Clean This Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        "This is the exact, real Python code that just ran on your file — "
        "not a summary. Every function below was applied, in order, to produce "
        "the **Cleaned Data** tab."
    )

    import inspect
    import cleaning as cleaning_module
    import utils as utils_module

    CLEANING_STEPS = [
        ("1️⃣ Clean Column Names", cleaning_module.clean_column_names,
         "Lowercases and snake_cases every column header so names are consistent "
         "and safe to reference in code (e.g. `Customer Name` → `customer_name`)."),
        ("2️⃣ Remove Duplicate Rows", cleaning_module.remove_duplicates,
         "Drops exact duplicate rows using `drop_duplicates()` so repeated records "
         "don't skew counts, sums, or averages."),
        ("3️⃣ Fix Data Types", cleaning_module.fix_data_types,
         "Auto-detects columns that are secretly numbers or dates stored as text "
         "(e.g. \"1,200\" or \"2024-01-05\") and converts them with "
         "`pd.to_numeric` / `pd.to_datetime`."),
        ("4️⃣ Normalize Categorical Text", cleaning_module.normalize_categoricals,
         "Strips whitespace and standardizes casing (e.g. \" pune \", \"PUNE\" → "
         "\"Pune\") so the same category isn't counted as multiple different values."),
        ("5️⃣ Handle Missing Values", cleaning_module.handle_missing_values,
         "Fills NaNs based on column dtype: **median** for numeric columns, "
         "**forward/back-fill** for dates, and **mode** (most frequent value) for text — "
         "using `fillna()` in each case."),
        ("6️⃣ Handle Outliers (IQR)", cleaning_module.remove_outliers_iqr,
         "Computes Q1, Q3 and IQR per numeric column, then **caps** (clips) any value "
         "outside [Q1 - k×IQR, Q3 + k×IQR] instead of deleting the row, so extreme "
         "values stop distorting stats without losing data."),
        ("🔗 Orchestrator — runs all steps above in order", cleaning_module.run_full_cleaning,
         "Calls each function above, in this exact sequence, and builds the "
         "Cleaning Report you saw in the previous tab."),
    ]

    for title, func, explanation in CLEANING_STEPS:
        with st.expander(title, expanded=False):
            st.markdown(explanation)
            st.code(inspect.getsource(func), language="python")

    st.markdown('<div class="section-header">🧰 Supporting Helper Functions (utils.py)</div>', unsafe_allow_html=True)
    st.caption("File loading, missing-value placeholder handling, and report formatting used behind the scenes.")
    with st.expander("📂 Show utils.py helper functions"):
        for name in ["load_file", "_read_csv_smart", "get_schema_summary",
                     "numeric_profile", "format_cleaning_report"]:
            fn = getattr(utils_module, name)
            st.markdown(f"**`{name}`**")
            st.code(inspect.getsource(fn), language="python")

    st.markdown("---")
    st.markdown(
        "💡 Want to learn the concepts step-by-step (missing values, outliers, "
        "encoding, scaling, EDA) outside the app? Check **`full_preprocessing_tutorial.py`** "
        "in the project repo — a standalone, fully-commented script you can run directly with "
        "`python full_preprocessing_tutorial.py`."
    )
