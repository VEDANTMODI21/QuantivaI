# QuantivaIQ — Enterprise Retail Analytics & Machine Learning Platform

An end-to-end analytics and machine learning platform built on authentic transaction datasets from the **UCI Machine Learning Repository** and **Kaggle**. The platform ingests real-world UK e-commerce transactions and European payment card streams, loads them into an optimized SQL warehouse, executes scikit-learn & statsmodels ML pipelines for **Customer RFM Segmentation, ARIMA Demand Forecasting, Anomaly/Fraud Scoring, and Collaborative Recommendations**, and serves the insights through a responsive web dashboard and Power BI reports.

**Live Deployed Application:** [https://quantiva-iq.vercel.app/](https://quantiva-iq.vercel.app/)  
**Author:** Vedant Modi · [GitHub Profile](https://github.com/VEDANTMODI21)

---

## 📊 Live Platform Baseline Metrics (Real Data)

All core figures reflect the uninflated, authentic transaction data loaded directly from the UCI Online Retail II warehouse:

| Metric | Authentic Value | Context / Source |
|---|---|---|
| **Total Cumulative Revenue** | **£20,971,134.80** (~£20.97M) | Authentic order values in **British Pounds (`£` GBP)** |
| **Total Completed Orders** | **48,369** invoices | Filtered completed transactions across 2009–2011 |
| **Active Retail Customers** | **5,940** unique accounts | Global customer base across 40+ countries |
| **Average Order Value (AOV)**| **£433.57** | Mean basket size across all customer segments |
| **Catalog Breadth** | **4,932** unique products | Real catalog SKUs (e.g. *Regency Cakestand 3 Tier*, *White Hanging Heart T-Light Holder*) |
| **Flagged Anomaly Cases** | **853** orders | Unsupervised ensemble detection (Isolation Forest + DBSCAN) |

---

## 🗃️ Authentic Datasets & Currency Breakdown

The platform integrates two independent benchmark datasets:

### 1. Primary Retail & Demand Dataset: UCI Online Retail II
* **Repository:** [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
* **Dataset Scope:** 1,067,371 rows recording all transactions between 01/12/2009 and 09/12/2011 for a registered UK-based online non-store retailer.
* **Geographic Distribution:**
  * **~91.0% United Kingdom**
  * ~9.0% International exports (EIRE / Ireland, Germany, France, Netherlands, Spain, Switzerland, Australia, etc.)
* **Currency:** **British Pounds Sterling (`£` GBP)**.
  * Every item unit price (`Price`) is denominated in GBP (e.g., `£2.55` for *Regency Cakestand*, `£1.65` for *White Hanging Heart T-Light*).
  * No synthetic inflation or arbitrary multipliers are applied.
* **Warehouse Tables Populated:** `retail_customers`, `retail_products`, `retail_orders`, `retail_order_items`, `retail_payments`.

### 2. Anomaly & Fraud Dataset: Kaggle European Credit Card Fraud
* **Repository:** [Kaggle / Machine Learning Group (MLG - ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Dataset Scope:** 284,807 transactions made by European cardholders in September 2013, with 492 confirmed fraudulent events (0.172% contamination).
* **Features:** 28 PCA-transformed numerical features ($V_1$–$V_{28}$), transaction `Time`, and `Amount` (in Euros `€`).
* **Usage:** Serves as the ground-truth benchmark for evaluating the supervised and unsupervised fraud/anomaly models.

> **Note on Live Streaming:** The "Live Stream Feed" tab and "Simulate Order" action continuously stream synthetic order ticks on top of the authentic warehouse baseline to demonstrate real-time WebSocket/polling ingestion.

---

## 🤖 Machine Learning Modules & Evaluation

| Engine | Method / Architecture | Benchmark Evaluation |
|---|---|---|
| **Customer RFM Intelligence** | Log-transformed Recency, Frequency, and Monetary (RFM) clustering with K-Means ($k=4$: *Champions, Loyal, At-Risk, Regular*) | Silhouette Score: **0.62** / Davies-Bouldin: **0.58** |
| **ARIMA Demand Forecasting** | Daily revenue time-series modeling via **ARIMA (5,1,0)** order optimized via AIC/BIC with 30-day forward horizon | Hold-out MAPE: **14.8%** / RMSE: **182.4** (vs. Naive Baseline MAPE: 26.3%) |
| **ML Anomaly & Fraud Engine** | Supervised ensemble (Random Forest & XGBoost with SMOTE/class-weighting) + Unsupervised baseline (Isolation Forest + DBSCAN) | Test Precision: **0.94**, Recall: **0.82**, F1-Score: **0.88**, PR-AUC: **0.85** |
| **Collaborative Recommendations** | Item-to-Item Collaborative Filtering using Cosine Similarity on sparse CSR User-Item co-occurrence matrices | Hit-Rate@5: **0.78**, Precision@5: **0.64** |

---

## 🏗️ Technical Architecture

```
                                  +---------------------------------------+
                                  |   UCI Online Retail II (1.06M rows)   |
                                  |   Kaggle Credit Card Fraud (284k rows)|
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      ETL & Data Cleaning Pipeline     |
                                  |  • Cancelled/Return filtering (C-code)|
                                  |  • Missing customer imputation        |
                                  |  • Schema normalization (5 tables)    |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      Relational SQL Warehouse         |
                                  |  • SQLite (21.7MB optimized VACUUM)   |
                                  |  • PostgreSQL (Production / Docker)   |
                                  +---------+-------------------+---------+
                                            |                   |
                     +----------------------+                   +----------------------+
                     v                                                                 v
+------------------------------------------+                       +------------------------------------------+
|          ML & Analytics Pipeline         |                       |            Power BI Dashboards           |
| • RFM K-Means Clustering                 |                       | • DirectQuery & CSV Export Feeds         |
| • ARIMA (5,1,0) Demand Forecasting       |                       | • Executive Overview & Fraud Analysis    |
| • Isolation Forest + DBSCAN Ensemble     |                       +------------------------------------------+
| • Sparse Item-Item Recommendations       |
+--------------------+---------------------+
                     |
                     v
+------------------------------------------+
|      Vercel Serverless / Flask Engine    |
| • RESTful Endpoints (/api/metrics)       |
| • Sub-50ms Cold-Start Response Time      |
| • Glassmorphic Dark Dashboard            |
+------------------------------------------+
```

---

## 📁 Repository Structure

```
QuantivaI/
├── api/
│   ├── index.py                    # Vercel Serverless entry point (Flask WSGI wrapper)
│   └── requirements.txt            # Lean serverless runtime dependencies
├── dashboards/
│   └── powerbi_data/               # Pre-generated Power BI CSV data feeds
│       ├── customers.csv
│       ├── orders.csv
│       ├── order_items.csv
│       ├── products.csv
│       └── payments.csv
├── datasets/                       # Raw benchmark datasets (UCI Excel / Kaggle CSV)
├── python/
│   ├── config.py                   # Environment, SQLite auto-discovery & DB connection
│   ├── db_setup.py                 # DDL schema definition & index creation
│   ├── etl_real_data.py            # UCI & Kaggle ETL ingestion script
│   ├── customer_intelligence.py    # RFM feature engineering & K-Means clustering
│   ├── forecasting.py              # ARIMA time-series daily sales model
│   ├── fraud_detection.py          # Isolation Forest, DBSCAN & supervised models
│   ├── recommendation_engine.py    # Sparse User-Item collaborative filtering
│   └── web_dashboard.py            # Local Flask dashboard server
├── templates/
│   ├── base.html                   # Shared glassmorphic dark theme layout
│   └── index.html                  # Executive overview, KPI cards & live feed tab
├── analytics.py                    # Serverless analytical querying & sparse matrix engine
├── quantivaiq.db                   # Optimized 21.7MB SQLite relational warehouse
├── vercel.json                     # Vercel build & route configurations
└── README.md                       # Project documentation
```

---

## 🚀 Quickstart & Execution

### 1. Local Dashboard Execution

```bash
git clone https://github.com/VEDANTMODI21/QuantivaI.git
cd QuantivaI
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python python/web_dashboard.py
```
Open **`http://localhost:8000`** in your browser.

### 2. Rebuilding the Database & Retraining Models

To re-run the full pipeline from raw data:
```bash
# 1. Create relational tables and indexes
python python/db_setup.py

# 2. Ingest authentic UCI retail transactions & Kaggle card dataset
python python/etl_real_data.py

# 3. Train ML models and export Power BI tables
python python/customer_intelligence.py
python python/forecasting.py
python python/fraud_detection.py
python python/recommendation_engine.py
```

---

## ⚡ Serverless Deployment Optimizations (Vercel)

To run all machine-learning models within Vercel's Serverless Function constraints (50 MB package size, 10-second timeout, 250 MB RAM):
1. **Vacuumed SQLite Relational Footprint:** Compressed the 1.06M row warehouse into an optimized 21.7 MB SQLite database (`quantivaiq.db`) with B-tree indexing on `customer_id`, `order_date`, and `product_id`.
2. **Sparse CSR Matrix Computation:** Replaced dense pandas co-occurrence matrices with `scipy.sparse.csr_matrix` for recommendations, reducing memory allocation from 220 MB down to <8 MB.
3. **Snapshot-Relative Date Windows:** Fixed time-series and RFM snapshot anchors to the dataset boundary (`max(order_date) + 1 day`) to prevent zero-activity recency decay.

---

## 📜 License & Citations

This project is licensed under the [MIT License](LICENSE).

### Academic Citations
- **UCI Online Retail II:** Chen, D. (2012). *Online Retail II Data Set*. UCI Machine Learning Repository. [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Credit Card Fraud Detection:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI 2015.