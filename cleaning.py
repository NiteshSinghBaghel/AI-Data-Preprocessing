"""
cleaning.py — AI-Data-Preprocessing
Handles all data cleaning operations automatically.
"""

import pandas as pd
import numpy as np
import re

from pyparsing import col
from utils import log_step


def clean_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Lowercase + underscore column names."""
    old_cols = df.columns.tolist()
    df.columns = [
        re.sub(r"\s+", "_", col.strip().lower())
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        for col in df.columns
    ]
    new_cols = df.columns.tolist()
    changes = [f"`{o}` → `{n}`" for o, n in zip(old_cols, new_cols) if o != n]
    return df, changes


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def fix_data_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Auto-detect and convert data types."""
    changes = {}
    for col in df.columns:
        original_dtype = str(df[col].dtype)

        # Try numeric conversion
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
            if converted.notna().sum() / max(df[col].notna().sum(), 1) > 0.8:
                df[col] = converted
                changes[col] = f"{original_dtype} → float64 (numeric)"
                continue

            # Try datetime conversion
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if converted.notna().sum() / max(df[col].notna().sum(), 1) > 0.7:
                    df[col] = converted
                    changes[col] = f"{original_dtype} → datetime64 (date)"
                    continue
            except Exception:
                pass

        # Fix numeric types safely (NO forced Int64)
        if df[col].dtype in ["int64", "float64"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, changes


def normalize_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize string/object columns — strip, lowercase, title-case."""
    changes = {}
    for col in df.columns:
        if df[col].dtype == object:
            unique_before = df[col].nunique()
            df[col] = df[col].astype(str).str.strip()
            # Detect if column looks like a category
            if df[col].nunique() < 0.5 * len(df) and len(df) > 5:
                df[col] = df[col].str.lower().str.title()
            unique_after = df[col].nunique()
            if unique_before != unique_after:
                changes[col] = f"Unique values: {unique_before} → {unique_after}"
    return df, changes

#handle missing values based on dtype:
def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fill missing values based on dtype:
    - Numeric  → median
    - Datetime → forward fill
    - Object   → mode
    """

    # ✅ ADD THIS LINE (IMPORTANT FIX)
    df.replace(["", " ", "NA", "N/A", "null", "None", "not_available", "abc"], pd.NA, inplace=True)

    report = {}
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            report[col] = f"{null_count} nulls filled with median ({fill_val:.4g})"

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].ffill().bfill()
            report[col] = f"{null_count} nulls forward/back filled (datetime)"

        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val)
            report[col] = f"{null_count} nulls filled with mode ('{fill_val}')"

    return df, report

def remove_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> tuple[pd.DataFrame, dict]:
    """
    Cap outliers using IQR method on numeric columns.
    Values outside [Q1 - k*IQR, Q3 + k*IQR] are capped (not dropped).
    """
    report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            report[col] = f"{outliers} outliers capped to [{lower:.4g}, {upper:.4g}]"

    return df, report


def run_full_cleaning(df: pd.DataFrame,iqr_threshold=1.5) -> tuple[pd.DataFrame, dict]:
    """
    Orchestrates all cleaning steps in order.
    Returns cleaned DataFrame + full cleaning report.
    """
    report = {}

    log_step("Cleaning column names...")
    df, col_changes = clean_column_names(df)
    report["column_names"] = col_changes

    log_step("Removing duplicate rows...")
    df, dupes_removed = remove_duplicates(df)
    report["duplicates_removed"] = dupes_removed

    log_step("Fixing data types...")
    df, dtype_changes = fix_data_types(df)
    report["dtype_changes"] = dtype_changes

    log_step("Normalizing categorical values...")
    df, cat_changes = normalize_categoricals(df)
    report["categorical_changes"] = cat_changes

    log_step("Handling missing values...")
    df, missing_report = handle_missing_values(df)
    report["missing_values"] = missing_report

    log_step("Removing outliers (IQR method)...")
    df, outlier_report = remove_outliers_iqr(df)
    report["outliers"] = outlier_report

    return df, report