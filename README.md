# QuantivaIQ — Enterprise Retail Intelligence Platform (INR Currency)

QuantivaIQ is a full-stack retail intelligence platform engineered for enterprise data analytics, automated machine learning fraud detection, demand forecasting, RFM customer segmentation, collaborative filtering recommendations, and interactive executive reporting in **Indian Rupees (`₹` / INR)**.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Power BI](https://img.shields.io/badge/Power%20BI-DirectQuery-F2C811?style=flat-square&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Currency](https://img.shields.io/badge/Currency-INR%20%E2%82%B9-10B981?style=flat-square)](#)

---

## 🎯 Executive Problem & Business Core

Modern retail operations face critical operational bottlenecks:
1. **Unseen Revenue Leakage from Fraud**: High-velocity bot attacks, bulk order reselling, and refund abuse drain store revenue. QuantivaIQ mitigates this by flagging high-risk transaction vectors in real-time before fulfillment.
2. **Stockouts & Overstock Costs**: Miscalculated demand leads to deadstock or missed revenue. QuantivaIQ deploys an ARIMA $(5,1,0)$ time-series model to project 30-day product demand curves in INR (`₹`).
3. **Low Customer Lifetime Value (LTV) & High Churn**: Standard marketing campaigns treat all customers identically. QuantivaIQ uses RFM quartile matrix scoring to segment customer personas.

### Why is the Bronze Category Essential?
In RFM Customer Segmentation, customers are categorized into **Platinum**, **Gold**, **Silver**, and **Bronze** tiers:
- **Bronze Tiers** represent low-recency, low-frequency, or entry-level buyers. 
- **Strategic Value**: Bronze customers constitute the largest slice of an e-commerce customer base. Converting even $5\%$ to $10\%$ of Bronze buyers into Silver or Gold tiers produces the highest marginal return on marketing expenditure, lowering overall Customer Acquisition Costs (CAC).

---

## 🏛️ End-to-End System Architecture & Data Pipeline

QuantivaIQ processes operational retail telemetry through a 5-stage pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │───>│ Feature Extract │───>│   ML Engines    │
│  & Warehousing  │    │  & Aggregation  │    │ (Fraud, ARIMA,  │
│  (SQLite/Postg) │    │  (SQL / Pandas) │    │  RFM, Recs)     │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐             │
│ Executive UI &  │<───│ Flask REST APIs │<────────────┘
│ ApexCharts Dash │    │ & Export Feeds  │
└─────────────────┘    └─────────────────┘
```

### Stage 1: Data Origin & Database Seeding (`quantivaiq.db`)
- **Primary Warehouse**: SQLite / PostgreSQL relational store containing tables for `customers`, `orders`, `order_items`, `products`, `inventory`, `suppliers`, `payments`, `refunds`, `customer_segments`, and `fraud_logs`.
- **Pre-Populated Dataset**: Ships pre-loaded with **50,018 completed retail transactions**, **500 active customers**, **853 flagged fraud anomaly logs**, and **5,000 customer RFM records** in Indian Rupees (`₹` INR).
- **Synthetic ETL Generator** ([`python/etl_pipeline.py`](file:///c:/Users/HP/QuantivaI/QuantivaI/python/etl_pipeline.py)): Simulates realistic purchase transactions with Indian payment gateways (*UPI*, *Credit Card*, *Net Banking*, *Wallet*).
- **Live Traffic Simulator** ([`python/live_data_generator.py`](file:///c:/Users/HP/QuantivaI/QuantivaI/python/live_data_generator.py)): Background event simulator injecting live transaction streams into the warehouse with pre-calculated pricing.

### Stage 2: Feature Engineering & Aggregation (`python/utils.py`)
SQL queries aggregate real-time customer behavioral features:
- `total_orders` & `active_days`: Measures purchase frequency.
- `avg_order_amount` & `max_order_amount`: Monitors monetary variance in INR (`₹`).
- `refund_ratio`: Calculates return frequency relative to orders.
- `recency_days`: Tracks time elapsed since last purchase.

### Stage 3: Machine Learning Analytics Engines

#### 1. Ensemble Fraud Detection Engine (`python/fraud_detection.py`)
Combines 3 distinct mathematical models for robust anomaly detection:
- **Isolation Forest** (`sklearn.ensemble.IsolationForest`): Tree-based isolation of extreme spending vectors (`contamination=0.02`).
- **DBSCAN Clustering** (`sklearn.cluster.DBSCAN`): Identifies low-density spatial noise points (`eps=2.5`, `min_samples=5`).
- **Statistical Z-Score** (`scipy.stats.zscore`): Flags feature vectors exceeding **3 standard deviations** ($|Z| > 3$).
- **Ensemble Voting**: Customer profiles receiving $\ge 2$ model flags are categorized with actionable risk types (*Velocity Fraud*, *Refund Abuse*, *Outlier Spike*).

#### 2. Demand Sales Forecasting (`python/forecasting.py`)
- **ARIMA Time Series Model** (`statsmodels.tsa.arima.model.ARIMA`): Fits auto-regressive integrated moving average model $(5,1,0)$ on daily revenue series in INR (`₹`) to project **30-day demand sales curves**.

#### 3. RFM Customer Segmentation (`python/customer_intelligence.py`)
- Computes **Recency, Frequency, and Monetary (RFM)** quartile scores, bucketing customers into *Platinum*, *Gold*, *Silver*, and *Bronze* tiers.

#### 4. Collaborative Recommendation Engine (`python/recommendation_engine.py`)
- Constructs sparse User-Item interaction matrices and computes **Cosine Similarity** matrices (`sklearn.metrics.pairwise.cosine_similarity`) to generate cross-sell product recommendations for active customer personas.

### Stage 4: Web Application & REST APIs (`python/web_dashboard.py`)
Flask framework powering real-time web routes:
- `GET /`: Renders responsive glassmorphism dark-theme dashboard.
- `GET /api/metrics`: Core KPIs formatted in INR (`₹`) with Lakhs (`₹ L`) and Crores (`₹ Cr`) notation.
- `GET /api/fraud`: Detailed ML anomaly engine outputs and flagged profiles.
- `GET /api/forecast`: Daily revenue baseline and ARIMA 30-day projected sales in `₹`.
- `GET /api/segments`: RFM segmentation counts and tier average spend in `₹`.
- `GET /api/recommendations`: Collaborative filtering item recommendations.
- `GET /api/recent-orders`: Operational transaction feed.
- `POST /api/simulator/trigger`: Triggers instant simulated transactions.
- `GET /api/export/csv`: Streams downloadable CSV executive summary report.

### Stage 5: Executive Frontend UI (`templates/index.html` & `base.html`)
- Built with **ApexCharts**, **Outfit & Inter Google Fonts**, responsive CSS grid, glassmorphism cards, neon status indicators, and wrapped flex tab navigation:
  - 📊 **Executive Overview**: ApexCharts Revenue Trend & ARIMA Demand Forecast curve in `₹`, Top Products Table, Regional Donut Breakdown.
  - 🚨 **ML Fraud Engine**: Multi-model anomaly classification table & risk meters.
  - 🎯 **Customer RFM Intelligence**: Recency, Frequency, Monetary spend matrix & tier breakdown in `₹`.
  - 📈 **ARIMA Demand Forecast**: Projected 30-day demand sales graph.
  - 💡 **ML Recommendations**: Cross-sell item recommendations.
  - ⚡ **Live Order Feed**: Real-time order stream with instant trigger control & toast alerts.

---

## ⚡ Step-by-Step Operating Guide

### 1. Run the Web Application Locally
```powershell
python web_dashboard.py
```
Open **`http://localhost:8000`** in your browser.

### 2. Execute Standalone Analytical Engines
```powershell
# Run ML Fraud Anomaly Engine
python python/fraud_detection.py

# Run ARIMA Demand Forecasting Engine
python python/forecasting.py

# Run RFM Customer Intelligence
python python/customer_intelligence.py

# Run Recommendation Engine
python python/recommendation_engine.py
```

### 3. Re-seed / Re-generate Database
```powershell
# Initialize schema
python python/db_setup.py

# Generate retail transactions & seed warehouse
python python/etl_pipeline.py
```

---

## 📂 Repository Structure

```
QuantivaIQ/
├── analytics.py                       # Combined ML analytics & cache engine
├── web_dashboard.py                   # Root Flask server entrypoint
├── run_demo.py                        # Standalone in-memory ML demo
├── quantivaiq.db                      # Pre-populated SQLite warehouse (50k+ orders in INR)
├── datasets/                          # Source CSV datasets
├── dashboards/                        # Power BI templates & exported CSV data
├── python/                            # Core Python application modules
│   ├── config.py                      # DB connection & logging config
│   ├── db_setup.py                    # Database schema setup script
│   ├── etl_pipeline.py                # Data generation & ETL pipeline
│   ├── fraud_detection.py             # Isolation Forest + DBSCAN ML Fraud Engine
│   ├── forecasting.py                 # ARIMA time-series sales forecasting
│   ├── customer_intelligence.py       # RFM customer segmentation engine
│   ├── recommendation_engine.py       # Item collaborative filtering recommender
│   ├── live_data_generator.py         # Real-time transaction traffic simulator
│   ├── web_dashboard.py               # Flask app & REST API routes
│   └── utils.py                       # SQL helpers & bulk insert utilities
├── sql/                               # SQL schemas & analytical queries
├── templates/                         # HTML5 responsive UI templates
│   ├── base.html                      # Glassmorphism dark layout & navbar
│   └── index.html                     # Multi-tab ApexCharts retail dashboard
├── Dockerfile                         # Production Docker image build
├── docker-compose.yml                 # Multi-container Compose config
├── render.yaml                        # Render.com deployment Blueprint
├── vercel.json                        # Vercel serverless deployment config
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```