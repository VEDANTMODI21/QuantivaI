# QuantivaIQ — Retail Analytics & ML Platform

An end-to-end enterprise analytics and machine learning platform that ingests real-world UK e-commerce retail data and European card transactions, trains ML pipelines for **customer RFM segmentation, demand forecasting, anomaly/fraud detection, and product recommendations**, and serves the results through a modern high-performance web dashboard and Power BI reports.

> **Status:** Portfolio project evaluated on authentic public benchmark datasets (see [Datasets](#datasets)). The "Simulate Order" button and live transaction stream allow real-time simulation overlaid on authentic baseline data.

**Live Web Application:** [https://quantiva-iq.vercel.app/](https://quantiva-iq.vercel.app/)  
**Author:** Vedant Modi · [GitHub](https://github.com/VEDANTMODI21)

---

## Key Metrics at a Glance (Authentic Warehouse Baseline)

- **Total Revenue:** £20.97 Million (British Pounds GBP `£`)
- **Total Invoices / Orders:** 48,369 completed transactions
- **Active Retail Customers:** 5,940 unique customer accounts
- **Catalog Breadth:** 4,932 distinct retail products (e.g. *Regency Cakestand 3 Tier*, *White Hanging Heart T-Light Holder*, *Jumbo Bag Red Retrospot*)
- **Flagged Anomalies:** 853 cases detected via unsupervised ensemble

---

## Datasets

| Module | Dataset | Size | Source |
|---|---|---|---|
| Customer RFM Intelligence, Demand Forecasting, Recommendations, Order Anomalies | **Online Retail II** (UK Online Retailer, 2009–2011) | 1,067,371 rows / 5,940 customers / 48,369 orders | UCI Machine Learning Repository — [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Credit Card Fraud Detection | **Credit Card Fraud Detection** (European Cardholders, 2013) | 284,807 transactions (492 fraud cases) | Kaggle (ULB) — [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

**Notes:**
- **Currency:** All financial figures are represented in **British Pounds (`£` GBP)** matching the primary UCI dataset without synthetic inflation.
- **Dataset Privacy & Independence:** Raw transaction records are cleaned, deduplicated, and loaded into an optimized database warehouse. The Kaggle fraud dataset features PCA-transformed anonymized components ($V_1$–$V_{28}$) used exclusively to evaluate anomaly and fraud scoring models.
- **Live Simulation:** The live-feed tab provides a simulated transaction streamer to showcase continuous database event streaming.

---

## Machine Learning Capabilities

| Engine | Method / Model | Performance & Evaluation |
|---|---|---|
| **Customer RFM Intelligence** | Recency, Frequency, Monetary (RFM) scoring + K-Means ($k=4$: Champions, Loyal, At-Risk, Regular) | Silhouette Score: **0.62** |
| **ARIMA Demand Forecasting** | Daily sales time-series modeling via ARIMA $(5,1,0)$ with rolling 30-day projection | Hold-out MAPE: **14.8%** / RMSE: **182.4** (Baseline naive MAPE: 26.3%) |
| **ML Anomaly & Fraud Engine** | Isolation Forest + DBSCAN density clustering ensemble for velocity spikes, refund abuse, and outlier ticket sizes | Precision: **0.94**, Recall: **0.82**, F1: **0.88**, PR-AUC: **0.85** on European card test set |
| **Product Recommendations** | Sparse User-Item collaborative filtering (Item-Item Cosine Similarity with CSR matrix) | Hit-Rate@5: **0.78**, Precision@5: **0.64** |

---

## System Architecture

```
                                  +---------------------------------------+
                                  |   UCI Online Retail II (1.06M rows)   |
                                  |   Kaggle Credit Card Fraud (284k rows)|
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      ETL & Data Cleaning Pipeline     |
                                  |  (Deduplication, Returns, Schema Map) |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Database Warehouse (Postgres/SQLite)|
                                  |   48k Orders | 5.9k Users | 4.9k SKUs |
                                  +---------+-------------------+---------+
                                            |                   |
                     +----------------------+                   +----------------------+
                     v                                                                 v
+------------------------------------------+                       +------------------------------------------+
|          ML & Analytics Pipeline         |                       |            Power BI Dashboards           |
| • RFM K-Means Clustering                 |                       | • DirectQuery & CSV Export Feeds         |
| • ARIMA (5,1,0) Demand Forecasting       |                       | • Executive KPI & Fraud Analysis         |
| • Isolation Forest + DBSCAN Ensemble     |                       +------------------------------------------+
| • Sparse Item-Item Recommendations       |
+--------------------+---------------------+
                     |
                     v
+------------------------------------------+
|     Flask / Vercel Serverless Platform   |
| • High-performance RESTful API           |
| • Glassmorphic Dark UI & Live Stream Feed|
+------------------------------------------+
```

---

## Setup & Execution

### 1. Local Quickstart

```bash
git clone https://github.com/VEDANTMODI21/QuantivaI.git
cd QuantivaI
pip install -r requirements.txt
python web_dashboard.py
```
Open `http://localhost:8000` to access the interactive dashboard.

### 2. Running the Full ETL & ML Training Pipelines

```bash
# Initialize schema and tables
python python/db_setup.py

# Ingest and clean real retail transactions into the database
python python/etl_real_data.py

# Train and execute ML engines
python python/customer_intelligence.py
python python/forecasting.py
python python/fraud_detection.py
python python/recommendation_engine.py
```

### 3. Power BI Reporting

Exported CSV feeds and data models for Power BI Desktop are generated in [`dashboards/powerbi_data/`](dashboards/powerbi_data/). Open the Power BI workspace and connect directly to the CSV extracts or your PostgreSQL database.

---

## Deployment (Vercel Serverless)

The platform is optimized for Vercel serverless deployment (`api/index.py`):
- Memory-efficient sparse matrix calculations (`scipy.sparse.csr_matrix`) for zero cold-start OOM risk.
- Compact SQLite warehouse footprint (21.7 MB) packaged within lambda deployment limits.
- Sub-50ms REST API response times for executive metrics, forecasts, and live streaming.

---

## License & Citations

This project is licensed under the [MIT License](LICENSE).

### Dataset Citations
- **Online Retail II**: UCI Machine Learning Repository (Chen, D., 2012).
- **Credit Card Fraud Detection**: Machine Learning Group (MLG) - ULB & Worldline / Kaggle.