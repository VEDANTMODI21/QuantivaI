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
    o = fetch_data("SELECT order_id, customer_id, order_date, total_amount FROM orders")
    r = fetch_data("SELECT o.customer_id, COUNT(*) AS refunds FROM refunds r "
                   "JOIN orders o ON r.order_id = o.order_id GROUP BY o.customer_id")
    o["day"] = pd.to_datetime(o["order_date"], errors="coerce").dt.date
    g = o.groupby("customer_id").agg(orders=("order_id", "count"), avg=("total_amount", "mean"),
                                     mx=("total_amount", "max"), sd=("total_amount", "std"),
                                     days=("day", "nunique")).fillna(0).reset_index()
    g = g.merge(r, how="left", on="customer_id").fillna(0)
    g["refund_ratio"] = g["refunds"] / g["orders"]
    g["freq"] = g["orders"] / g["days"].clip(lower=1)
    X = StandardScaler().fit_transform(g[["avg", "mx", "sd", "refund_ratio", "freq"]])
    iso = IsolationForest(contamination=0.02, random_state=42).fit(X)
    g["votes"] = ((iso.predict(X) == -1).astype(int)
                  + (DBSCAN(eps=2.5, min_samples=5).fit_predict(X) == -1).astype(int)
                  + (np.abs(X) > 3).any(axis=1).astype(int))
    g["score"] = -iso.decision_function(X)
    f = g[g["votes"] >= 2].sort_values("score", ascending=False)
    top = f.head(8)[["customer_id", "avg", "refund_ratio", "votes"]].round(2)
    return {"flagged": int(len(f)), "top": top.to_dict("records")}


def _segments():
    o = fetch_data("SELECT customer_id, order_date, total_amount FROM orders WHERE status = 'Completed'")
    o["d"] = pd.to_datetime(o["order_date"], errors="coerce")
    g = o.groupby("customer_id").agg(r=("d", lambda s: (pd.Timestamp.now() - s.max()).days),
                                     f=("d", "count"), m=("total_amount", "sum"))
    q = lambda s, lab: pd.qcut(s.rank(method="first"), 5, labels=lab).astype(int)
    g["R"], g["F"], g["M"] = q(g["r"], [5, 4, 3, 2, 1]), q(g["f"], [1, 2, 3, 4, 5]), q(g["m"], [1, 2, 3, 4, 5])
    score = g["R"] * 100 + g["F"] * 10 + g["M"]

    def seg(i):
        s = score[i]
        if s >= 444: return "VIP"
        if s >= 333: return "Loyal"
        if g.at[i, "R"] <= 2: return "At-Risk"
        if s <= 222: return "Inactive"
        return "Regular"

    counts = pd.Series([seg(i) for i in g.index]).value_counts()
    return [{"segment": k, "customers": int(v)} for k, v in counts.items()]


def _forecast(days=14):
    d = fetch_data("SELECT DATE(order_date) AS ds, SUM(total_amount) AS y, COUNT(*) AS n "
                   "FROM orders WHERE status = 'Completed' GROUP BY DATE(order_date) ORDER BY ds")
    d = d[d["n"] >= 20].tail(60)  # skip sparse simulator-only days
    hist = [round(float(v), 2) for v in d["y"]]
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fc = ARIMA(d["y"].astype(float).values, order=(5, 1, 0)).fit().forecast(steps=days)
        fc = [round(float(max(v, 0)), 2) for v in fc]
    except Exception:
        fc = []
    return {"history": hist, "forecast": fc}


def _recs(n_users=5):
    d = fetch_data("SELECT o.customer_id, oi.product_id, SUM(oi.quantity) AS q FROM orders o "
                   "JOIN order_items oi ON o.order_id = oi.order_id WHERE o.status = 'Completed' "
                   "GROUP BY o.customer_id, oi.product_id")
    names = fetch_data("SELECT product_id, product_name FROM products").set_index("product_id")["product_name"]
    m = d.pivot(index="customer_id", columns="product_id", values="q").fillna(0)
    sim = pd.DataFrame(cosine_similarity(m.values.T), index=m.columns, columns=m.columns)
    out = []
    for u in (m > 0).sum(axis=1).nlargest(n_users).index:
        bought = m.columns[m.loc[u] > 0]
        top = sim[bought].sum(axis=1).drop(bought).nlargest(3).index
        out.append({"customer_id": int(u), "items": [str(names.get(p, p)) for p in top]})
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
