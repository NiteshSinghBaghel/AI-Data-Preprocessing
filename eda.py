"""
eda.py — AI-Data-Preprocessing
Performs Exploratory Data Analysis and generates visual charts + text insights.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from typing import Optional


# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = "coolwarm"
FIG_DPI = 120
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.0)


def _fig_to_bytes(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=FIG_DPI)
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Summary Statistics ──────────────────────────────────────────────────────────
def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all columns."""
    desc = df.describe(include="all").T
    desc["dtype"] = df.dtypes
    desc["null_count"] = df.isna().sum()
    desc["null_pct"] = (df.isna().mean() * 100).round(2)
    desc["unique"] = df.nunique()
    return desc


# ── Null Analysis ───────────────────────────────────────────────────────────────
def plot_null_heatmap(df: pd.DataFrame) -> Optional[BytesIO]:
    """Heatmap of missing values across columns."""
    null_df = df.isnull()
    if null_df.sum().sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.6), 4))
    sns.heatmap(null_df, yticklabels=False, cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Value Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Columns")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return _fig_to_bytes(fig)


def plot_null_bar(df: pd.DataFrame) -> Optional[BytesIO]:
    """Bar chart of null percentages per column."""
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(null_pct) * 0.7), 4))
    bars = ax.bar(null_pct.index, null_pct.values, color=sns.color_palette("Reds_r", len(null_pct)))
    ax.set_title("Missing Value % per Column", fontsize=14, fontweight="bold")
    ax.set_ylabel("Null %")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, null_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── Correlation Matrix ──────────────────────────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame) -> Optional[BytesIO]:
    """Heatmap of Pearson correlations for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=PALETTE,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 8}, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix (Numeric Columns)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── Distribution Plots ──────────────────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame) -> dict[str, BytesIO]:
    """KDE + histogram for each numeric column."""
    charts = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return charts

    for col in num_cols:
        series = df[col].dropna()
        if series.nunique() < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

        # Histogram
        axes[0].hist(series, bins=min(40, series.nunique()), color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[0].set_title(f"{col} — Histogram", fontsize=11, fontweight="bold")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        # KDE
        try:
            series.plot.kde(ax=axes[1], color="#C44E52", linewidth=2)
        except Exception:
            axes[1].plot(series.sort_values(), np.linspace(0, 1, len(series)), color="#C44E52")
        axes[1].set_title(f"{col} — KDE", fontsize=11, fontweight="bold")
        axes[1].set_xlabel(col)

        plt.tight_layout()
        charts[col] = _fig_to_bytes(fig)

    return charts


# ── Categorical Counts ──────────────────────────────────────────────────────────
def plot_categorical_counts(df: pd.DataFrame, max_categories: int = 20) -> dict[str, BytesIO]:
    """Bar chart of value counts for categorical columns."""
    charts = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        vc = df[col].value_counts().head(max_categories)
        if vc.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(vc) * 0.5), 4))
        bars = ax.bar(vc.index.astype(str), vc.values,
                      color=sns.color_palette("tab10", len(vc)))
        ax.set_title(f"{col} — Value Counts (Top {len(vc)})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        charts[col] = _fig_to_bytes(fig)

    return charts


# ── Boxplots ────────────────────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame) -> Optional[BytesIO]:
    """Side-by-side boxplots for all numeric columns (normalized)."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return None

    # Normalize for display
    normed = (num_df - num_df.mean()) / (num_df.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(max(8, len(num_df.columns) * 0.9), 5))
    normed.boxplot(ax=ax, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                   medianprops=dict(color="#C44E52", linewidth=2))
    ax.set_title("Boxplots — Numeric Columns (Z-Normalized)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Z-Score")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── Text Insights ───────────────────────────────────────────────────────────────
def generate_text_insights(df_raw: pd.DataFrame, df_clean: pd.DataFrame, report: dict) -> list[str]:
    """Generate bullet-point text insights from cleaning report + EDA."""
    insights = []

    # Shape
    insights.append(f"📊 Dataset: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns "
                    f"(after cleaning: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns)")

    # Duplicates
    dupes = report.get("duplicates_removed", 0)
    if dupes:
        insights.append(f"🔁 {dupes} duplicate rows removed ({dupes / df_raw.shape[0] * 100:.1f}% of data).")
    else:
        insights.append("✅ No duplicate rows found.")

    # Missing values
    mv = report.get("missing_values", {})
    if mv:
        insights.append(f"🕳️ Missing values handled in {len(mv)} column(s): " +
                        "; ".join([f"`{k}` ({v})" for k, v in list(mv.items())[:5]]) +
                        ("..." if len(mv) > 5 else ""))
    else:
        insights.append("✅ No missing values detected in the dataset.")

    # Outliers
    out = report.get("outliers", {})
    if out:
        total_outliers = sum(
            int(v.split(" ")[0]) for v in out.values() if v.split(" ")[0].isdigit()
        )
        insights.append(f"📉 Outliers capped in {len(out)} column(s) (~{total_outliers} values).")
    else:
        insights.append("✅ No significant outliers found.")

    # Dtype fixes
    dtype_ch = report.get("dtype_changes", {})
    if dtype_ch:
        insights.append(f"🔧 Data type corrections applied to {len(dtype_ch)} column(s): " +
                        ", ".join([f"`{k}`" for k in list(dtype_ch.keys())[:5]]))

    # Categorical normalization
    cat_ch = report.get("categorical_changes", {})
    if cat_ch:
        insights.append(f"🔤 Categorical normalization reduced unique values in {len(cat_ch)} column(s).")

    # Correlation highlights
    num_df = df_clean.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr().abs()
        for i in range(len(corr)):
            corr.iat[i, i] = 0
        max_corr = corr.max().max()
        if max_corr > 0.7:
            idx = corr.stack().idxmax()
            insights.append(f"🔗 Strong correlation detected: `{idx[0]}` ↔ `{idx[1]}` "
                            f"(r = {corr.loc[idx[0], idx[1]]:.2f}).")

    # High-null columns
    null_pct = df_raw.isnull().mean() * 100
    high_null = null_pct[null_pct > 30]
    if not high_null.empty:
        insights.append(f"⚠️ Columns with >30% missing values: " +
                        ", ".join([f"`{c}` ({v:.1f}%)" for c, v in high_null.items()]))

    # Skewness
    for col in num_df.columns:
        sk = num_df[col].skew()
        if abs(sk) > 2:
            insights.append(f"📐 `{col}` is highly skewed (skewness = {sk:.2f}) — "
                            "consider log transform for modeling.")
            break  # one example is enough

    return insights


# ── Full EDA Runner ─────────────────────────────────────────────────────────────
def run_full_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame, report: dict) -> dict:
    """
    Run all EDA steps on the cleaned DataFrame.
    Returns a dict with summary, charts, and insights.
    """
    return {
        "summary_stats": get_summary_stats(df_clean),
        "null_heatmap": plot_null_heatmap(df_raw),
        "null_bar": plot_null_bar(df_raw),
        "correlation_matrix": plot_correlation_matrix(df_clean),
        "distributions": plot_distributions(df_clean),
        "categorical_counts": plot_categorical_counts(df_clean),
        "boxplots": plot_boxplots(df_clean),
        "insights": generate_text_insights(df_raw, df_clean, report),
    }