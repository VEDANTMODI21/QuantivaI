import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

try:
    # pyrefly: ignore [missing-import]
    from .utils import fetch_data
except ImportError:
    try:
        from python.utils import fetch_data
    except ImportError:
        from utils import fetch_data

_cache = {"t": 0, "v": None}
TTL = 60  # seconds




def _fraud():
    fl = fetch_data("SELECT DISTINCT customer_id, risk_score FROM fraud_logs WHERE customer_id != 99999 ORDER BY risk_score DESC")
    if fl.empty:
        return {"flagged": 0, "top": []}
    flagged_ids = tuple(fl["customer_id"].tolist())
    o = fetch_data(f"SELECT customer_id, AVG(total_amount) as avg FROM orders WHERE customer_id IN {flagged_ids} AND status = 'Completed' GROUP BY customer_id")
    r = fetch_data(f"SELECT customer_id, COUNT(*) as refunds FROM orders WHERE customer_id IN {flagged_ids} AND status = 'Cancelled' GROUP BY customer_id")
    tot = fetch_data(f"SELECT customer_id, COUNT(*) as total_orders FROM orders WHERE customer_id IN {flagged_ids} GROUP BY customer_id")
    m = fl.merge(o, on="customer_id", how="left").merge(r, on="customer_id", how="left").merge(tot, on="customer_id", how="left").fillna(0)
    m["refund_ratio"] = (m["refunds"] / m["total_orders"].clip(lower=1)).round(3)
    m["avg"] = m["avg"].round(2)
    m["votes"] = 3
    m["classification"] = "High-Value Outlier"
    top = m.head(8)[["customer_id", "avg", "refund_ratio", "votes", "classification"]].copy()
    return {"flagged": int(len(fl)), "top": top.to_dict("records")}


def _segments():
    df = fetch_data("SELECT segment_name, COUNT(*) as customers FROM customer_segments GROUP BY segment_name ORDER BY customers DESC")
    if not df.empty:
        return [{"segment": str(r["segment_name"]).replace(" Customers", "").replace(" Accounts", ""), "customers": int(r["customers"])} for _, r in df.iterrows()]
    return []


def _forecast(days=14):
    d = fetch_data("SELECT DATE(order_date) AS ds, SUM(total_amount) AS y, COUNT(*) AS n "
                   "FROM orders WHERE status = 'Completed' AND order_date <= '2011-12-10' "
                   "GROUP BY DATE(order_date) ORDER BY ds")
    d = d[d["n"] >= 15].tail(60)
    hist = [round(float(v), 2) for v in d["y"]]
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fc = ARIMA(d["y"].astype(float).values, order=(5, 1, 0)).fit().forecast(steps=days)
        fc = [round(float(max(v, 0)), 2) for v in fc]
    except Exception:
        fc = []
    return {"history": hist, "forecast": fc}


def _recs(n_users=5):
    from scipy.sparse import csr_matrix
    d = fetch_data("SELECT o.customer_id, oi.product_id, SUM(oi.quantity) AS q FROM orders o "
                   "JOIN order_items oi ON o.order_id = oi.order_id "
                   "JOIN products p ON oi.product_id = p.product_id "
                   "WHERE o.status = 'Completed' AND o.customer_id != 99999 "
                   "AND p.product_name NOT IN ('Manual', 'DOTCOM POSTAGE', 'POSTAGE', 'CARRIAGE', 'Discount', 'BANK CHARGES', 'CRUK Commission') "
                   "AND UPPER(p.product_name) NOT LIKE '%POSTAGE%' "
                   "AND UPPER(p.product_name) NOT LIKE '%MANUAL%' "
                   "GROUP BY o.customer_id, oi.product_id")
    if d.empty:
        return []
    names = fetch_data("SELECT product_id, product_name FROM products").set_index("product_id")["product_name"]
    
    top_pids = d.groupby("product_id")["q"].sum().nlargest(500).index
    d_sub = d[d["product_id"].isin(top_pids)].copy()
    
    cust_ids = d_sub["customer_id"].unique()
    prod_ids = top_pids.values
    cust_map = {cid: idx for idx, cid in enumerate(cust_ids)}
    prod_map = {pid: idx for idx, pid in enumerate(prod_ids)}
    
    row_ind = d_sub["customer_id"].map(cust_map).values
    col_ind = d_sub["product_id"].map(prod_map).values
    data = d_sub["q"].values.astype(np.float32)
    
    mat = csr_matrix((data, (row_ind, col_ind)), shape=(len(cust_ids), len(prod_ids)))
    item_mat = mat.T
    norms = np.sqrt(item_mat.power(2).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    
    user_counts = d_sub.groupby("customer_id")["q"].count()
    top_users = user_counts.nlargest(n_users).index
    
    out = []
    for uid in top_users:
        if uid not in cust_map:
            continue
        u_idx = cust_map[uid]
        u_vector = mat.getrow(u_idx).toarray().flatten()
        bought_indices = np.where(u_vector > 0)[0]
        if len(bought_indices) == 0:
            continue
            
        sub_items = item_mat[bought_indices]
        scores = sub_items.dot(mat).toarray().sum(axis=0)
        scores[bought_indices] = -1.0
        
        top_rec_indices = np.argsort(scores)[-3:][::-1]
        rec_names = [str(names.get(prod_ids[i], f"Product {prod_ids[i]}")) for i in top_rec_indices if scores[i] > 0]
        out.append({"customer_id": int(uid), "items": rec_names})
    return out


def get_analytics():
    if _cache["v"] is None or time.time() - _cache["t"] > TTL:
        res = {}
        for key, fn in (("fraud", _fraud), ("segments", _segments), ("forecast", _forecast), ("recs", _recs)):
            try:
                res[key] = fn()
            except Exception as exc:  # one failing model must not blank the dashboard
                res[key] = {"error": str(exc)}
        _cache.update(t=time.time(), v=res)
    return _cache["v"]
