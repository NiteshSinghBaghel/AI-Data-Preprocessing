"""
utils.py — AI-Data-Preprocessing
Shared utility functions: logging, file I/O, format detection.
"""

import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
import datetime


# ── Logging ─────────────────────────────────────────────────────────────────────
def log_step(message: str) -> None:
    """Simple timestamped console logger."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}")


# ── File Loading ─────────────────────────────────────────────────────────────────
def load_file(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.
    Accepts a Streamlit UploadedFile or a file path string.
    """
    if isinstance(uploaded_file, (str, Path)):
        path = Path(uploaded_file)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _read_csv_smart(path)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    else:
        # Streamlit UploadedFile
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return _read_csv_smart(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {name}")


def _read_csv_smart(source) -> pd.DataFrame:
    """Try multiple delimiters + encodings for robust CSV loading."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    delimiters = [",", ";", "\t", "|"]

    for enc in encodings:
        for delim in delimiters:
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, sep=delim, encoding=enc, on_bad_lines="warn")
                if df.shape[1] > 1:  # at least 2 columns means delimiter worked
                    return df
            except Exception:
                continue

    # Last resort: default pandas CSV
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source)


# ── File Export ─────────────────────────────────────────────────────────────────
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to UTF-8 CSV bytes (for download)."""
    buf = BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to Excel bytes (for download)."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
    return buf.getvalue()


# ── Schema Helper ────────────────────────────────────────────────────────────────
def get_schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a concise schema table: column, dtype, non-null count, null %, unique."""
    rows = []
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_pct = round((df[col].isna().mean()) * 100, 2)
        rows.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Non-Null": non_null,
            "Null %": null_pct,
            "Unique": df[col].nunique(),
            "Sample": str(df[col].dropna().iloc[0]) if non_null > 0 else "N/A",
        })
    return pd.DataFrame(rows)


# ── Numeric Column Inspector ─────────────────────────────────────────────────────
def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Quick numeric stats: mean, median, std, skew, kurtosis."""
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "mean": num.mean(),
        "median": num.median(),
        "std": num.std(),
        "min": num.min(),
        "max": num.max(),
        "skewness": num.skew(),
        "kurtosis": num.kurtosis(),
    }).round(4)


# ── Cleaning Report Formatter ────────────────────────────────────────────────────
def format_cleaning_report(report: dict) -> str:
    """Convert cleaning report dict to a human-readable markdown string."""
    lines = ["## 🧹 Cleaning Report\n"]

    col_names = report.get("column_names", [])
    lines.append(f"**Column Renames:** {len(col_names)} column(s) renamed.")
    for c in col_names[:10]:
        lines.append(f"  - {c}")
    if len(col_names) > 10:
        lines.append(f"  - ... and {len(col_names) - 10} more")

    lines.append(f"\n**Duplicates Removed:** {report.get('duplicates_removed', 0)}")

    dtype_ch = report.get("dtype_changes", {})
    if dtype_ch:
        lines.append(f"\n**Dtype Changes ({len(dtype_ch)}):**")
        for k, v in list(dtype_ch.items())[:10]:
            lines.append(f"  - `{k}`: {v}")

    mv = report.get("missing_values", {})
    if mv:
        lines.append(f"\n**Missing Values Filled ({len(mv)}):**")
        for k, v in list(mv.items())[:10]:
            lines.append(f"  - `{k}`: {v}")

    outliers = report.get("outliers", {})
    if outliers:
        lines.append(f"\n**Outliers Capped ({len(outliers)}):**")
        for k, v in list(outliers.items())[:10]:
            lines.append(f"  - `{k}`: {v}")

    cat_ch = report.get("categorical_changes", {})
    if cat_ch:
        lines.append(f"\n**Categorical Normalization ({len(cat_ch)}):**")
        for k, v in list(cat_ch.items())[:10]:
            lines.append(f"  - `{k}`: {v}")

    return "\n".join(lines)