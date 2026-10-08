# QuantivaIQ — Enterprise Retail Analytics & Machine Learning Platform

An end-to-end analytics and machine learning platform built on authentic transaction datasets from the **UCI Machine Learning Repository** and **Kaggle**. The platform ingests real-world UK e-commerce transactions and European payment card streams, loads them into an optimized SQL warehouse (`quantivaiq.db`), executes scikit-learn & statsmodels ML pipelines for **Customer RFM Segmentation, ARIMA Demand Forecasting, Anomaly/Fraud Scoring, and Collaborative Recommendations**, and serves the insights through a responsive web dashboard and Power BI report feeds.

**Live Deployed Application:** [https://quantiva-iq.vercel.app/](https://quantiva-iq.vercel.app/)  
**Author:** Vedant Modi · [GitHub Profile](https://github.com/VEDANTMODI21)

---

## 📊 Live Platform Baseline Metrics (Real Data)

All core figures reflect authentic transaction data loaded directly into the relational warehouse (`quantivaiq.db`) from the UCI Online Retail II dataset:

| Metric | Authentic Value | Context / Source |
|---|---|---|
| **Total Cumulative Revenue** | **£20,972,627.24** (~£20.97M) | Authentic completed order values in **British Pounds (`£` GBP)** |
| **Total Completed Orders** | **48,372** invoices | Filtered completed transactions across 2009–2011 |
| **Active Retail Customers** | **5,940** unique accounts | Customer accounts across 40+ countries |
| **Average Order Value (AOV)**| **£433.57** | Mean completed order basket size |
| **Catalog Breadth** | **4,932** unique products | Real catalog SKUs (e.g., *Regency Cakestand 3 Tier*, *White Hanging Heart T-Light Holder*) |
| **Warehouse Fraud Cases** | **56** logged entries | High-risk customer/order anomaly logs stored in `fraud_logs` table |

---

## 🗃️ Authentic Datasets & Currency Breakdown

The platform integrates two benchmark datasets:

### 1. Primary Retail & Demand Dataset: UCI Online Retail II
* **Repository:** [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
* **Dataset Scope:** 1,067,371 records of transactions occurring between 01/12/2009 and 09/12/2011 for a UK-based online retailer.
* **Geographic Scope:** ~91% United Kingdom, ~9% international exports (EIRE/Ireland, Germany, France, Netherlands, Spain, Switzerland, Australia, etc.).
* **Currency:** **British Pounds Sterling (`£` GBP)**. Prices reflect genuine transaction values without synthetic inflation.
* **Warehouse Tables Populated:** `customers`, `products`, `orders`, `order_items`, `payments`, `categories`, `suppliers`, `inventory`.

### 2. Anomaly & Fraud Dataset: Kaggle European Credit Card Fraud
* **Repository:** [Kaggle / Machine Learning Group (MLG - ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Dataset Scope:** 284,807 transactions made by European cardholders in September 2013, including 492 fraud events.
* **Features:** 28 PCA-transformed numerical features ($V_1$–$V_{28}$), transaction `Time`, and `Amount`.
* **Usage:** Ground-truth benchmark for evaluating supervised/unsupervised fraud and anomaly models.

---

## 🤖 Machine Learning Engine & Implementation

| Engine | Algorithm / Architecture | Implementation File |
|---|---|---|
| **Customer RFM Intelligence** | Log/Quantile RFM scoring ($R, F, M \in [1..5]$) mapping to segments (*VIP*, *Loyal*, *At-Risk*, *Inactive*, *Regular*) + Random Forest Churn Classifier & CLTV Regressor | [customer_intelligence.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/customer_intelligence.py) |
| **Demand Forecasting** | Daily sales time-series modeling via **ARIMA (5,1,0)** order & Linear Regression with lag features (`lag_1`, `lag_7`, `day_of_week`) | [forecasting.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/forecasting.py) |
| **ML Anomaly & Fraud Engine** | Multi-model Ensemble Voting: **Isolation Forest** (`contamination=0.02`), **DBSCAN** (`eps=2.5, min_samples=5`), and **Z-Score** outliers ($|Z| > 3$) | [fraud_detection.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/fraud_detection.py) |
| **Collaborative Recommendations** | Item-to-Item Collaborative Filtering using Cosine Similarity on sparse CSR User-Item co-occurrence matrices (`scipy.sparse.csr_matrix`) | [recommendation_engine.py](file:///c:/Users/HP/QuantivaI/QuantivaI/python/recommendation_engine.py) / [analytics.py](file:///c:/Users/HP/QuantivaI/QuantivaI/analytics.py) |

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
| • ARIMA (5,1,0) Demand Forecasting       |                       | • Customer & Sales Materialized Views    |
| • Isolation Forest + DBSCAN Ensemble     |                       +------------------------------------------+
| • Sparse Item-Item Recommendations       |
+--------------------+---------------------+
                     |
                     v
+------------------------------------------+
|          Flask / Vercel Web Engine       |
| • RESTful API Endpoints (/api/metrics)   |
| • Glassmorphic Dark UI (index.html)      |
| • Real-time Transaction Simulator Stream |
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
├── datasets/                       # Benchmark datasets (UCI Excel & Kaggle CSV)
│   ├── online_retail_II.xlsx
│   └── creditcard.csv
├── docs/                           # Documentation assets
├── python/
│   ├── config.py                   # DB connection & logging config
│   ├── db_setup.py                 # SQLite & PostgreSQL DDL schema initialization
│   ├── etl_real_data.py            # UCI & Kaggle ETL ingestion pipeline
│   ├── customer_intelligence.py    # RFM segmentation, Churn & CLTV models
│   ├── forecasting.py              # ARIMA & Linear Regression sales forecasting
│   ├── fraud_detection.py          # Isolation Forest, DBSCAN & Z-score ensemble
│   ├── recommendation_engine.py    # Sparse matrix collaborative filtering
│   ├── live_data_generator.py      # Real-time streaming transaction simulator
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
- **Credit Card Fraud Detection:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI 2015.