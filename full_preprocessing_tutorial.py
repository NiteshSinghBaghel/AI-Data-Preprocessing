"""
full_preprocessing_tutorial.py — AI-Data-Preprocessing
=========================================================
A SELF-CONTAINED, STEP-BY-STEP walkthrough of Data Preprocessing
& Cleaning in pandas — written as a learning reference.

Every stage below is a separate, clearly commented function so you can
read this file top-to-bottom and understand WHAT each step does, WHY it
matters, and HOW it's implemented. The `main()` at the bottom runs the
whole pipeline end-to-end on a small sample dataset (so it works with
zero setup) but every function also works on any DataFrame you pass in.

Run it directly:
    python full_preprocessing_tutorial.py

Sections covered:
  1. Creating / Loading raw data
  2. Initial Exploration (shape, dtypes, head, info)
  3. Handling Missing Values   (isnull, dropna, fillna: mean/median/mode/ffill/bfill)
  4. Removing Duplicates
  5. Fixing Data Types
  6. Handling Outliers (IQR method)
  7. Text / Categorical Cleaning (strip, lower, standardize)
  8. Encoding Categorical Variables (Label Encoding & One-Hot Encoding)
  9. Feature Scaling (Normalization & Standardization)
 10. Exploratory Data Analysis (EDA): describe(), value_counts(), correlation
 11. Saving the final cleaned dataset
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# 1. CREATING / LOADING RAW DATA
# ──────────────────────────────────────────────────────────────────────────
def load_sample_data() -> pd.DataFrame:
    """
    In a real project you would load data with:
        df = pd.read_csv("your_file.csv")
        df = pd.read_excel("your_file.xlsx")

    Here we build a small, intentionally messy DataFrame in memory so this
    script runs standalone and demonstrates every problem a real dataset
    usually has: missing values, duplicates, mixed types, outliers, and
    inconsistent text formatting.
    """
    data = {
        "Name":   ["Alice", "bob", "Charlie", "  DAVE ", "Eve", "Alice", None, "Frank"],
        "Age":    [25, np.nan, 30, 22, 29, 25, 40, 250],       # 250 = outlier, NaN = missing
        "Salary": [50000, 60000, np.nan, 52000, 58000, 50000, 61000, 59000],
        "City":   ["Pune", "mumbai", "Delhi", "PUNE", None, "Pune", "Delhi", "Mumbai"],
        "Joined": ["2021-01-05", "2020-11-23", None, "2022-03-15",
                   "2019-07-01", "2021-01-05", "2020-05-30", "2022-01-10"],
    }
    return pd.DataFrame(data)


# ──────────────────────────────────────────────────────────────────────────
# 2. INITIAL EXPLORATION
# ──────────────────────────────────────────────────────────────────────────
def explore_data(df: pd.DataFrame) -> None:
    """
    Always look at your data before touching it. This tells you:
      - df.shape       -> (rows, columns)
      - df.head()      -> first 5 rows, a visual sanity check
      - df.info()      -> dtypes + non-null counts per column
      - df.isnull().sum() -> exact count of missing values per column
      - df.duplicated().sum() -> how many exact duplicate rows exist
    """
    print("\n===== 2. INITIAL EXPLORATION =====")
    print(f"Shape (rows, cols): {df.shape}")
    print("\nFirst 5 rows:\n", df.head())
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


# ──────────────────────────────────────────────────────────────────────────
# 3. HANDLING MISSING VALUES
# ──────────────────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing data (NaN / None) breaks most ML models and statistics, so we
    must decide, PER COLUMN, how to deal with it. Common strategies:

    a) df.isnull() / df.isna()
         Boolean mask marking every missing cell. `.sum()` on it gives you
         a count per column — this is always the first diagnostic step.

    b) df.dropna()
         Deletes rows (or columns) containing NaNs entirely.
         Use only when missing data is rare and safe to discard —
         dropping too many rows loses information.

    c) df['col'].fillna(value)
         Fills NaNs with a chosen value. The right "value" depends on the
         column's meaning and dtype:
           - Numeric column  -> fillna(df['col'].mean())   (or .median())
                 mean: good for roughly symmetric distributions
                 median: robust to outliers/skewed distributions (safer default)
           - Categorical/text -> fillna(df['col'].mode()[0])
                 fills with the most frequently occurring category
           - Time-ordered data -> fillna(method='ffill') / 'bfill'
                 carries the previous/next valid value forward/backward —
                 good for time series where values change slowly

    Below we apply median for numeric columns, mode for text columns, and
    forward/backward fill for the date column, matching the dtype of each.
    """
    print("\n===== 3. HANDLING MISSING VALUES =====")
    df = df.copy()

    # --- 3a. Diagnose: how many nulls, where? ---
    print("Nulls BEFORE fixing:\n", df.isnull().sum())

    # --- 3b. Numeric columns -> fill with MEDIAN ---
    # Median is preferred over mean here because it's not skewed by
    # outliers (like the 250 in 'Age').
    for col in df.select_dtypes(include=[np.number]).columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  '{col}': filled NaNs with median = {median_val}")

    # --- 3c. Text/categorical columns -> fill with MODE (most common value) ---
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().sum() > 0 and col != "Joined":
            mode_val = df[col].mode(dropna=True)
            fill_value = mode_val[0] if len(mode_val) else "Unknown"
            df[col] = df[col].fillna(fill_value)
            print(f"  '{col}': filled NaNs with mode = '{fill_value}'")

    # --- 3d. Date column -> forward-fill then back-fill ---
    # Convert to real datetime first so pandas understands ordering.
    df["Joined"] = pd.to_datetime(df["Joined"], errors="coerce")
    df["Joined"] = df["Joined"].ffill().bfill()
    print("  'Joined': filled NaT with forward-fill then back-fill")

    print("\nNulls AFTER fixing:\n", df.isnull().sum())
    return df


# ──────────────────────────────────────────────────────────────────────────
# 4. REMOVING DUPLICATES
# ──────────────────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    df.duplicated()      -> boolean mask of rows that are exact repeats
                             of an earlier row (all columns identical).
    df.drop_duplicates() -> removes those repeated rows, keeping the
                             first occurrence by default.

    Duplicate rows silently bias averages, counts, and model training,
    so this should run after nulls are handled (so formatting differences
    like "Pune" vs " Pune " don't mask duplicates — see step 7).
    """
    print("\n===== 4. REMOVING DUPLICATES =====")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"Removed {removed} duplicate row(s). New shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────
# 5. FIXING DATA TYPES
# ──────────────────────────────────────────────────────────────────────────
def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data often arrives with the wrong dtype (e.g. numbers stored as text
    because of a stray comma or currency symbol). We force the correct
    dtype per column using pd.to_numeric / pd.to_datetime with
    errors="coerce" — invalid values become NaN instead of crashing.
    """
    print("\n===== 5. FIXING DATA TYPES =====")
    df = df.copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    print(df.dtypes)
    return df


# ──────────────────────────────────────────────────────────────────────────
# 6. HANDLING OUTLIERS (IQR METHOD)
# ──────────────────────────────────────────────────────────────────────────
def handle_outliers_iqr(df: pd.DataFrame, columns=None, factor: float = 1.5) -> pd.DataFrame:
    """
    Outliers are extreme values (like Age = 250) that distort averages
    and model training. The IQR (Interquartile Range) method is a
    standard, distribution-free way to detect them:

        Q1 = 25th percentile,  Q3 = 75th percentile
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

    Any value outside [lower_bound, upper_bound] is considered an
    outlier. Instead of deleting those rows (which loses data), we CAP
    ("clip") them to the nearest bound — this keeps the row but removes
    the distortion.
    """
    print("\n===== 6. HANDLING OUTLIERS (IQR) =====")
    df = df.copy()
    columns = columns or df.select_dtypes(include=[np.number]).columns

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"  '{col}': {n_outliers} outlier(s) capped to range "
              f"[{lower:.2f}, {upper:.2f}]")

    return df


# ──────────────────────────────────────────────────────────────────────────
# 7. TEXT / CATEGORICAL CLEANING
# ──────────────────────────────────────────────────────────────────────────
def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Free-text and categorical columns are rarely clean:
      - Extra whitespace:  "  DAVE "  vs  "DAVE"
      - Inconsistent case: "mumbai" vs "Mumbai" vs "MUMBAI"

    .str.strip()      removes leading/trailing whitespace
    .str.lower()       normalizes case for comparison
    .str.title()       makes it human-readable again ("Mumbai")

    Doing this BEFORE removing duplicates or grouping is important,
    otherwise "Pune" and " pune " are treated as different categories.
    """
    print("\n===== 7. TEXT / CATEGORICAL CLEANING =====")
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.title()
        print(f"  '{col}': trimmed whitespace + standardized casing")
    return df


# ──────────────────────────────────────────────────────────────────────────
# 8. ENCODING CATEGORICAL VARIABLES
# ──────────────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Machine learning models need numbers, not text, so categorical
    columns must be encoded:

    a) Label Encoding — maps each unique category to an integer
       (e.g. Pune=0, Mumbai=1, Delhi=2). Simple, but implies a false
       "order" between categories — best for tree-based models.

    b) One-Hot Encoding (pd.get_dummies) — creates one binary (0/1)
       column PER category. No false ordering is implied, which is why
       it's usually preferred for linear/distance-based models.

    Here we one-hot encode 'City' as a demonstration, keeping 'Name'
    untouched since it's a unique identifier, not a feature.
    """
    print("\n===== 8. ENCODING CATEGORICAL VARIABLES =====")
    df = df.copy()
    df_encoded = pd.get_dummies(df, columns=["City"], prefix="City")
    print("One-hot encoded 'City' into columns:",
          [c for c in df_encoded.columns if c.startswith("City_")])
    return df_encoded


# ──────────────────────────────────────────────────────────────────────────
# 9. FEATURE SCALING
# ──────────────────────────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Numeric columns on very different scales (Age: 20-90, Salary: 20000-90000)
    can dominate distance-based algorithms (KNN, K-Means, gradient descent).
    Two common fixes:

    a) Min-Max Normalization -> squashes values into [0, 1]:
           x_scaled = (x - min) / (max - min)

    b) Standardization (Z-score) -> centers around mean 0, std 1:
           x_scaled = (x - mean) / std

    We add normalized columns here (suffix "_norm") while KEEPING the
    original columns, so you can compare before/after.
    """
    print("\n===== 9. FEATURE SCALING (Min-Max Normalization) =====")
    df = df.copy()
    columns = columns or df.select_dtypes(include=[np.number]).columns

    for col in columns:
        min_val, max_val = df[col].min(), df[col].max()
        if max_val > min_val:
            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
            print(f"  '{col}' -> '{col}_norm' scaled to [0, 1]")
    return df


# ──────────────────────────────────────────────────────────────────────────
# 10. EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    """
    EDA means summarizing and understanding data through statistics
    and simple checks (before/instead of plotting):

    - df.describe()        -> count, mean, std, min, quartiles, max for
                               every numeric column in one table
    - df['col'].value_counts() -> frequency of each category — great for
                               spotting imbalance or unexpected values
    - df.corr(numeric_only=True) -> Pearson correlation between numeric
                               columns; values near +1/-1 mean two columns
                               move together strongly
    """
    print("\n===== 10. EXPLORATORY DATA ANALYSIS (EDA) =====")
    print("\nSummary statistics:\n", df.describe(include="all").T)

    if "City" in df.columns:
        print("\nCity value counts:\n", df["City"].value_counts())

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        print("\nCorrelation matrix:\n", numeric_df.corr().round(2))


# ──────────────────────────────────────────────────────────────────────────
# 11. SAVING THE FINAL CLEANED DATASET
# ──────────────────────────────────────────────────────────────────────────
def save_clean_data(df: pd.DataFrame, path: str = "cleaned_output.csv") -> None:
    """
    Once every step above has run, persist the result so it can be reused
    (e.g. fed into a model, loaded into Excel, or shared with a team).
    index=False avoids writing pandas' internal row-numbers as a column.
    """
    df.to_csv(path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────
# MAIN — runs every step above, in order, on the sample dataset
# ──────────────────────────────────────────────────────────────────────────
def main():
    df = load_sample_data()
    explore_data(df)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = fix_data_types(df)
    df = handle_outliers_iqr(df, columns=["Age", "Salary"])
    df = clean_text_columns(df)

    run_eda(df)  # inspect BEFORE encoding, so category names are readable

    df_encoded = encode_categoricals(df)
    df_final = scale_features(df_encoded, columns=["Age", "Salary"])

    print("\n===== FINAL CLEANED DATAFRAME =====")
    print(df_final)

    save_clean_data(df_final)


if __name__ == "__main__":
    main()
