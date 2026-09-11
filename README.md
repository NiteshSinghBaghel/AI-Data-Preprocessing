# 🧹 AI-Data-Preprocessing
Project Live  : https://autoclean-ai.streamlit.app/

> Upload any CSV or Excel file — get back a fully cleaned dataset, EDA charts, and actionable insights.

---

## 📁 Project Structure

```
AI-Data-Preprocessing/
├── main.py          ← Streamlit UI (entry point)
├── cleaning.py      ← All data cleaning logic
├── eda.py           ← EDA: charts + text insights
├── utils.py         ← File I/O, logging, helpers
├── api.py           ← Optional FastAPI REST backend
├── requirements.txt ← Python dependencies
└── README.md
```
DEMO PICTURES

![Alt Text](Screenshot1.png)
![Alt Text](Screenshot2.png)
![Alt Text](Screenshot3.png)
![Alt Text](Screenshot4.png)
---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run main.py
```

Open your browser at **http://localhost:8501**

---

## 🌐 Optional FastAPI Backend

```bash
uvicorn api:app --reload --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Health check |
| POST   | `/clean` | Upload file → download cleaned CSV |
| POST   | `/report`| Upload file → JSON cleaning report + insights |
| POST   | `/schema`| Upload file → raw schema JSON |

**Example cURL:**
```bash
# Clean a file and download result
curl -X POST http://localhost:8000/clean \
     -F "file=@data.csv" \
     --output cleaned_data.csv

# Get JSON report
curl -X POST http://localhost:8000/report \
     -F "file=@data.csv"
```

---

## 🧹 What Gets Cleaned

| Issue | Method |
|-------|--------|
| Column names | Lowercase + snake_case |
| Duplicate rows | Exact match removal |
| Wrong data types | Auto-detection + conversion |
| Missing values (numeric) | Median fill |
| Missing values (datetime) | Forward/back fill |
| Missing values (text) | Mode fill |
| Outliers | IQR capping (configurable threshold) |
| Categorical inconsistency | Strip + title-case normalization |
| Mixed date formats | `pd.to_datetime` with inference |
| Mixed number formats | Strip commas, convert to float |

---

## 📊 EDA Features

- **Summary statistics** — describe() + dtype + null %
- **Missing value heatmap + bar chart**
- **Correlation matrix** (Pearson, numeric columns)
- **Distribution plots** — histogram + KDE per numeric column
- **Categorical bar charts** — value counts
- **Boxplots** — outlier visualization (z-normalized)
- **Text insights** — auto-generated bullet points

---

## 📦 Dependencies

```
streamlit, pandas, numpy, matplotlib, seaborn, openpyxl, xlrd
fastapi, uvicorn, python-multipart  (for API only)
```

---

## 💡 Tips

- **CSV support:** comma, semicolon, tab, pipe delimiters auto-detected
- **Encoding:** UTF-8, Latin-1, CP1252 tried automatically  
- **IQR threshold** is adjustable in the sidebar (1.0 = aggressive, 3.0 = conservative)
- **Download** as CSV or Excel directly from the UI

---

## 🎓 full_preprocessing_tutorial.py — Learn the Concepts

A standalone, heavily-commented script (`full_preprocessing_tutorial.py`) is
included for learning purposes. It runs the entire preprocessing pipeline
step-by-step on a small built-in sample dataset and prints an explanation of
each stage as it runs:

1. Loading raw data
2. Initial exploration (`shape`, `dtypes`, `head`, `isnull().sum()`)
3. Handling missing values (`isnull`, `dropna`, `fillna` with mean/median/mode/ffill/bfill)
4. Removing duplicates (`duplicated`, `drop_duplicates`)
5. Fixing data types (`pd.to_numeric`, `pd.to_datetime`)
6. Handling outliers (IQR method + capping)
7. Cleaning text/categorical columns (`strip`, `lower`, `title`)
8. Encoding categoricals (Label Encoding vs One-Hot Encoding)
9. Feature scaling (Min-Max normalization)
10. Exploratory Data Analysis (`describe`, `value_counts`, `corr`)
11. Saving the cleaned dataset to CSV

Run it directly — no arguments needed:
```bash
python full_preprocessing_tutorial.py
```
It works on its own sample data, and every function can also be imported
and reused on your own DataFrame.
