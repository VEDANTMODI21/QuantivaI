# QuantivaIQ — Retail Analytics & Machine Learning Platform

An end-to-end analytics and machine learning platform built on authentic transaction datasets from the **UCI Machine Learning Repository (Online Retail II)**. The platform ingests real-world UK e-commerce transactions, loads them into an optimized SQL warehouse (`quantivaiq.db`), executes scikit-learn & statsmodels ML pipelines for **Customer RFM Segmentation, Churn & CLTV Modeling, Autoregressive Demand Forecasting, Unsupervised Anomaly Scoring, and Collaborative Recommendations**, and serves the insights through an interactive web dashboard and Power BI report feeds.

**Live Deployed Application:** [https://quantiva-iq.vercel.app/](https://quantiva-iq.vercel.app/)  
**Author:** Vedant Modi · [GitHub Profile](https://github.com/VEDANTMODI21)

---

## 📊 Live Platform Baseline Metrics (Real Data)

All core figures reflect authentic transaction data loaded directly into the relational warehouse (`quantivaiq.db`) from the UCI Online Retail II dataset:

| Metric | Authentic Value | Context / Source |
|---|---|---|
| **Total Cumulative Revenue** | **£20,972,627.24** (~£20.97M) | Authentic completed order values in **British Pounds (`£` GBP)** |
| **Total Invoices** | **48,369** transactions | Authentic unique transaction identifiers (40,077 Completed, 8,292 Cancelled: 82.9% completion rate) |
| **Active Retail Accounts** | **5,940** accounts | 5,939 registered client profiles across 40+ countries + 1 guest placeholder account (#99999) |
| **Average Order Value (AOV)**| **£433.57** | Mean completed order basket size (£20.97M ÷ 48,369 total invoices = £433.59) |
| **Catalog Breadth** | **4,932** unique products | Authentic catalog SKUs (e.g., *Regency Cakestand 3 Tier*, *White Hanging Heart T-Light Holder*) |
| **High-Value Outliers Flagged** | **55** anomalies | Unsupervised ensemble voting (Isolation Forest + DBSCAN + Z-score) on transaction velocity & value |

---

## 🗃️ Authentic Datasets & Currency Breakdown

The platform processes authentic transactions from the **UCI Online Retail II** repository:

### Primary Retail & Demand Dataset: UCI Online Retail II
* **Repository:** [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
* **Dataset Scope:** 1,067,371 records of transactions occurring between 01/12/2009 and 09/12/2011 for a UK-based online retailer.
* **Geographic Scope:** ~91% United Kingdom, ~9% international exports (EIRE/Ireland, Germany, France, Netherlands, Spain, Switzerland, Australia, etc.).
* **Currency:** **British Pounds Sterling (`£` GBP)**. Prices reflect genuine transaction values without synthetic inflation.
* **Warehouse Tables Populated:** `customers`, `products`, `orders`, `order_items`, `payments`, `categories`, `suppliers`, `inventory`, `fraud_logs`.
* **Note on Streaming Demonstrations:** The live UI includes interactive "Simulate Order" demo triggers to showcase real-time ingest without altering fixed warehouse baseline KPIs.

---

## 🤖 Machine Learning Engine & Performance Benchmarks

| Engine | Algorithm / Architecture | Performance Metric | Implementation File |
|---|---|---|---|
| **Customer RFM Segmentation** | Log/Quantile RFM scoring ($R, F, M \in [1..5]$) into 5 tiers (*VIP*, *Loyal*, *At-Risk*, *Regular*, *Return-only*) | 5,940 customer accounts partitioned across 5 distinct behavioral profiles | [customer_intelligence.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/customer_intelligence.py) |
| **Customer Churn Classifier** | Random Forest Classifier on purchase recency and frequency | **62.0% Accuracy (+11.2 percentage points lift** over 50.85% majority class baseline; Precision: 0.61, Recall: 0.66, F1: 0.63) | [customer_intelligence.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/customer_intelligence.py) |
| **Customer Lifetime Value (CLTV)** | Random Forest Regressor on tenure and monthly velocity | **£300.70 MAE (9.7% of test-set mean CLTV** of £3,092.39; 33.5% of £898.92 median customer CLTV) | [customer_intelligence.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/customer_intelligence.py) |
| **Demand Forecasting** | Feature-engineered Autoregression (Lag-1, Lag-7, Day-of-Week) vs. **ARIMA (5,1,0)** | **36.6% test-period error (MAE £20.6k)** vs. **41.0% error (MAE £23.2k)** for ARIMA during Q4 holiday volume surge (accounting for Saturday warehouse closures) | [forecasting.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/forecasting.py) |
| **Anomaly & Fraud Engine** | Multi-model Ensemble Voting: **Isolation Forest** (`contamination=0.02`), **DBSCAN** (`eps=2.5, min_samples=5`), and **Z-Score** ($|Z| > 3$) | **55 High-Value Outliers** flagged out of 5,878 active accounts (0.94% detection rate) | [fraud_detection.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/fraud_detection.py) |
| **Collaborative Recommendations** | Item-to-Item Collaborative Filtering using Cosine Similarity on sparse CSR User-Item matrices (`scipy.sparse.csr_matrix`) | Top-N cross-sell recommendations with 100% catalog coverage | [recommendation_engine.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/recommendation_engine.py) / [analytics.py](file:///c:/Users/HP/QuantivaI/QuantivaI/analytics.py) |

---

## 🏗️ Technical Architecture

```
                                  +---------------------------------------+
                                  |   UCI Online Retail II (1.06M rows)   |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      ETL & Data Cleaning Pipeline     |
                                  |  • Cancelled invoice handling (C-code)|
                                  |  • Schema normalization & indexing    |
                                  |  • Product categorization & suppliers |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      Relational SQL Warehouse         |
                                  |  • SQLite (22.8 MB quantivaiq.db)     |
                                  |  • PostgreSQL schema & procedures     |
                                  +---------+-------------------+---------+
                                            |                   |
                     +----------------------+                   +----------------------+
                     v                                                                 v
+------------------------------------------+                       +------------------------------------------+
|          ML & Analytics Pipeline         |                       |            Power BI Dashboards           |
| • RFM Quantile & K-Means Segmentation    |                       | • CSV Data Feeds & DirectQuery           |
| • Lagged Autoregression & ARIMA (5,1,0)  |                       | • Customer & Sales Materialized Views    |
| • Isolation Forest + DBSCAN Ensemble     |                       +------------------------------------------+
| • Sparse Item-Item Recommendations       |
+--------------------+---------------------+
                     |
                     v
+------------------------------------------+
|          Flask / Vercel Web Engine       |
| • RESTful API Endpoints (/api/metrics)   |
| • Glassmorphic Dark UI (index.html)      |
| • Simulated Order Demo Stream (10s poll) |
+------------------------------------------+
```

---

## 📁 Repository Structure

```
QuantivaI/
├── api/
│   └── index.py                    # Vercel Serverless entry point (Flask WSGI wrapper)
├── dashboards/
│   ├── powerbi_data/               # Pre-generated Power BI CSV data feeds
│   │   ├── customers.csv
│   │   ├── orders.csv
│   │   ├── order_items.csv
│   │   ├── products.csv
│   │   ├── payments.csv
│   │   ├── customer_segments.csv
│   │   ├── fraud_logs.csv
│   │   ├── inventory.csv
│   │   ├── refunds.csv
│   │   ├── suppliers.csv
│   │   ├── categories.csv
│   │   └── mv_*.csv                # Materialized view exports for Power BI
│   ├── powerbi_template.md         # Power BI dashboard build instructions
│   └── powerbi_dashboard_template.md
├── datasets/                       # Benchmark datasets (UCI Excel)
│   └── online_retail_II.xlsx
├── docs/                           # Documentation assets
├── python/
│   ├── config.py                   # DB connection & logging config
│   ├── db_setup.py                 # SQLite & PostgreSQL DDL schema initialization
│   ├── etl_real_data.py            # UCI retail ETL ingestion pipeline
│   ├── customer_intelligence.py    # RFM segmentation, Churn & CLTV models
│   ├── forecasting.py              # ARIMA & Linear Regression sales forecasting
│   ├── fraud_detection.py          # Isolation Forest, DBSCAN & Z-score ensemble
│   ├── recommendation_engine.py    # Sparse matrix collaborative filtering
│   ├── live_data_generator.py      # Simulated demo order generator
│   ├── export_powerbi_csv.py       # Power BI CSV exporter
│   ├── utils.py                    # Shared database helper functions
│   └── web_dashboard.py            # Core Flask Web Application server
├── sql/
│   ├── schema.sql                  # PostgreSQL table definitions
│   ├── procedures.sql              # Stored procedures & triggers
│   └── analytics_queries.sql       # Analytical KPI & query library
├── templates/
│   ├── base.html                   # Glassmorphic base HTML template
│   └── index.html                  # Executive overview dashboard UI
├── analytics.py                    # Optimized serverless ML analytics module
├── index.html                      # Lightweight single-page dashboard fallback
├── quantivaiq.db                   # Optimized 22.8 MB SQLite relational warehouse
├── run_demo.py                     # Standalone in-memory pipeline demo
├── web_dashboard.py                # Root application launch script
├── vercel.json                     # Vercel serverless deployment config
└── README.md                       # Project documentation
```

---

## 🚀 Quickstart & Execution

### 1. Running the Web Dashboard Locally

```bash
git clone https://github.com/VEDANTMODI21/QuantivaI.git
cd QuantivaI

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run local web server
python web_dashboard.py
```
Open **`http://localhost:8000`** in your browser.

### 2. Standalone In-Memory Demo Execution

To quickly test the ML pipelines without a database setup:
```bash
python run_demo.py
```

### 3. Rebuilding Database & Re-running ETL Pipeline

To populate or refresh the SQLite warehouse (`quantivaiq.db`) from raw dataset files:
```bash
# 1. Initialize schema
python python/db_setup.py

# 2. Ingest authentic UCI retail transactions
python python/etl_real_data.py

# 3. Execute ML pipelines and segment updates
python python/customer_intelligence.py
python python/forecasting.py
python python/fraud_detection.py
python python/recommendation_engine.py

# 4. Export refreshed CSV data feeds for Power BI
python python/export_powerbi_csv.py
```

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main executive web dashboard |
| `/health` | `GET` | Health check endpoint for container / service monitoring |
| `/api/metrics` | `GET` | Complete executive dashboard metrics and ML analytics payload |
| `/api/fraud` | `GET` | Top suspicious customer profiles and anomaly counts |
| `/api/forecast` | `GET` | Historical revenue data and 14-day ARIMA forecast |
| `/api/segments` | `GET` | Customer RFM segment counts and average spend |
| `/api/recommendations` | `GET` | Personalized product recommendations by customer ID |
| `/api/recent-orders` | `GET` | Stream of recent transaction records |
| `/api/simulator/trigger` | `POST` | Injects a live transaction tick for streaming demonstrations |
| `/api/export/csv` | `GET` | Streams executive KPI summary as a downloadable CSV |

---

## 📜 License & Citations

This project is licensed under the [MIT License](LICENSE).

### Academic Citations
- **UCI Online Retail II:** Chen, D. (2012). *Online Retail II Data Set*. UCI Machine Learning Repository. [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)