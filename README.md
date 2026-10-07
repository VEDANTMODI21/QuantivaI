# QuantivaIQ — Retail Analytics & ML Platform

An end-to-end analytics project that loads real retail and card-transaction data into PostgreSQL, trains machine-learning models for **customer segmentation, demand forecasting, anomaly/fraud detection and product recommendations**, and serves the results through a Flask dashboard and a Power BI report.

> **Status:** student portfolio project. Models are trained and evaluated on public datasets (see [Datasets](#datasets)). The "Simulate Order" button and the live stream tab use **synthetic demo orders** and are labelled as such.

**Live demo:** https://quantivaiq.onrender.com *(free tier — first load can take ~1 minute)*  
**Author:** Vedant Modi · [GitHub](https://github.com/VEDANTMODI21)

---

## Screenshots

| Executive overview | Fraud engine | RFM segments | Demand forecast |
|---|---|---|---|
| `docs/overview.png` | `docs/fraud.png` | `docs/rfm.png` | `docs/forecast.png` |

---

## Datasets

| Module | Dataset | Size | Source |
|---|---|---|---|
| Segmentation (RFM), forecasting, recommendations, order anomalies | **Online Retail II** (UK online retailer, 2009–2011) | 1,067,371 rows / 5,876 customers | UCI Machine Learning Repository — [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Fraud detection | **Credit Card Fraud Detection** (European cardholders, 2013) | 284,807 transactions, 492 fraud | Kaggle (ULB) — [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

**Notes:**
- The two datasets are **independent**. The card-fraud data has anonymised features ($V_1$–$V_{28}$) and no product or customer fields, so it is not joined with the retail data.
- Retail amounts are in **GBP (£)**, as in the source data. No currency conversion is applied.
- Check each dataset's licence on its source page before redistributing. Raw files are **not** committed; see [Setup](#setup).
- The live-stream tab and "Simulate Order" button generate **synthetic** orders for demonstration only. They do not affect model training or the reported metrics.

---

## What it does

| Module | Method | Evaluation |
|---|---|---|
| **Customer segmentation** | RFM scoring + K-Means ($k=4$) | Silhouette score: **0.62** |
| **Demand forecasting** | ARIMA $(5,1,0)$ (order chosen by AIC / grid search), compared with a naive lag baseline | Hold-out MAPE: **14.8%** / RMSE: **182.4** (naive baseline MAPE: **26.3%**) |
| **Fraud detection** | Supervised classifier (Random Forest / XGBoost) with class-imbalance handling; Isolation Forest as an unsupervised baseline | Stratified hold-out — Precision: **0.94**, Recall: **0.82**, F1: **0.88**, PR-AUC: **0.85** |
| **Order anomalies** | Isolation Forest on retail orders/returns (no ground-truth labels, so reported as *anomalies*, not confirmed fraud) | Qualitative review of flagged orders (velocity spikes, refund abuse, outlier transaction amounts) |
| **Recommendations** | Item-based collaborative filtering (Cosine Similarity on sparse User-Item matrix) | Hit-rate@5: **0.78**, Precision@5: **0.64** |

---

## Architecture

```
Raw CSV/Excel  ->  ETL (Python/pandas)  ->  PostgreSQL (tables + materialized views)
                                              |                |
                                              v                v
                                   ML modules (scikit-learn,   Power BI (DirectQuery / CSV)
                                   statsmodels)
                                              |
                                              v
                                   Flask API + dashboard (Docker)
```

**Stack:** Python, pandas, scikit-learn, statsmodels, SQL / PostgreSQL / SQLite, Flask, Power BI, Docker Compose.

---

## Setup

### 1. Quick demo (no database)

```bash
git clone https://github.com/VEDANTMODI21/QuantivaI.git
cd QuantivaI
pip install -r requirements.txt
python run_demo.py
```

### 2. Full setup with real data

1. Download the datasets listed above into `datasets/` (not tracked by git).
2. Start PostgreSQL (local or Docker) and set the connection variables in `.env`:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/quantivaiq
   ```
3. Initialize the schema, load the data, and run the ML models:
   ```bash
   # Initialize tables and views
   python python/db_setup.py

   # Ingest and clean retail transactions
   python python/etl_pipeline.py

   # Execute ML engines
   python python/fraud_detection.py
   python python/forecasting.py
   python python/customer_intelligence.py
   python python/recommendation_engine.py
   ```
4. Start the dashboard:
   ```bash
   python web_dashboard.py
   ```
   Open `http://localhost:8000` in your browser.

### 3. Docker

```bash
docker compose up --build
```

---

## Power BI

Power BI templates and export scripts are provided in [`dashboards/`](dashboards/). Open the template in Power BI Desktop and enter your PostgreSQL connection or use the pre-generated CSV feeds in `dashboards/powerbi_data/`.

---

## Limitations

- Fraud labels come from one public dataset with anonymised features; results do not necessarily transfer to other card portfolios.
- Order anomalies on the retail data are unlabelled, so they indicate unusual orders, not confirmed fraud.
- The live stream and "Simulate Order" features use synthetic data for demonstration.
- The hosted demo runs on a free tier and may be slow after idle periods.

## Possible next steps

- Compare ARIMA with Prophet or gradient-boosted forecasting (LightGBM/CatBoost).
- Add model monitoring, data drift tracking, and scheduled retraining jobs.
- Replace the demo stream with a real Kafka / EventBridge streaming pipeline.

## License

This project is licensed under the [MIT License](LICENSE).

### Dataset Credits & Citations
- **Online Retail II**: UCI Machine Learning Repository (Dua, D. and Graff, C., 2019).
- **Credit Card Fraud Detection**: Machine Learning Group (MLG) - ULB & Worldline / Kaggle.