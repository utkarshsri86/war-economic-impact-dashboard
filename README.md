# ⚔️ War Economic Impact Dashboard

An end-to-end Data Science project analyzing the economic impact of wars and conflicts using Machine Learning and an interactive Streamlit dashboard.

---

## 📊 Project Structure

| File | Description |
|------|-------------|
| `1_eda.ipynb` | Exploratory Data Analysis — 10 segments |
| `2_ml_pipeline.ipynb` | ML Pipeline — Feature Engineering, Training, Evaluation |
| `dashboard.py` | Interactive Streamlit Dashboard |
| `requirements.txt` | Python dependencies |
| `war_economic_impact_dataset.csv` | Dataset — 100,000 conflict records |
| `model.pkl` | Trained Gradient Boosting model |
| `scaler.pkl` | StandardScaler for feature scaling |
| `features.pkl` | Selected feature list |

---

## 🔍 Features

- 📊 Analyze **100,000 conflict records** across 5 regions
- 💰 Compare GDP, inflation, currency devaluation across conflict types
- 👥 Compare poverty & unemployment **before vs during war**
- 🕵️ Explore black market and informal economy growth
- 🤖 Predict **during-war unemployment** using ML (R²=0.94)
- 🔍 Interactive filters by Region, Conflict Type, Status, Year

---

## 🤖 ML Model Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 2.91% | 3.49% | 0.9367 |
| Random Forest | 2.75% | 3.33% | 0.9422 |
| **Gradient Boosting** ✅ | **2.72%** | **3.29%** | **0.9436** |

**Target:** `During_War_Unemployment_%`
**Top Feature:** `Youth_Unemployment_Change_%` (96% importance)

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/utkarshsri86/war-economic-impact-dashboard.git
cd war-economic-impact-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run ML pipeline (to regenerate model files)
Open and run all cells in `2_ml_pipeline.ipynb`

### 4. Launch dashboard
```bash
streamlit run dashboard.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-1.3-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.17-purple)

- **Data:** Pandas, NumPy
- **ML:** Scikit-learn (Gradient Boosting, Random Forest, Linear Regression)
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Deployment:** Streamlit Cloud

---

## 📁 EDA Segments

1. Basic Dataset Info
2. Missing Values Check
3. Statistical Summary
4. Categorical Columns Analysis
5. Target Variable Distribution
6. GDP Change by Region & Conflict Type
7. Poverty & Unemployment Before vs During War
8. Inflation & Currency Devaluation
9. Black Market & Informal Economy
10. Correlation Heatmap
