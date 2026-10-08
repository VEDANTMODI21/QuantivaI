import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
try:
    from .config import setup_logging
    from .utils import get_engine, fetch_data, bulk_insert, execute_query
except ImportError:
    from config import setup_logging
    from utils import get_engine, fetch_data, bulk_insert, execute_query


logger = setup_logging("CustomerIntelligence")

class CustomerIntelligence:
    def __init__(self):
        logger.info("Initializing Customer Intelligence System...")

    def update_rfm_segments(self):
        logger.info("Calculating RFM Segments...")
        query = """
            SELECT 
                customer_id,
                MAX(order_date) AS last_order_date,
                COUNT(order_id) AS frequency,
                SUM(total_amount) AS monetary
            FROM orders
            WHERE status = 'Completed'
            GROUP BY customer_id
        """
        rfm_df = fetch_data(query)
        if rfm_df.empty:
            logger.warning("No order data available for RFM.")
            return

        rfm_df['parsed_date'] = pd.to_datetime(rfm_df['last_order_date'], errors='coerce')
        snapshot_date = rfm_df['parsed_date'].max() + pd.Timedelta(days=1) if rfm_df['parsed_date'].notna().any() else pd.Timestamp.now()
        rfm_df['recency'] = (snapshot_date - rfm_df['parsed_date'].fillna(snapshot_date - pd.Timedelta(days=999))).dt.days.clip(lower=0)

        # Calculate quantiles
        rfm_df['R_Quartile'] = pd.qcut(rfm_df['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm_df['F_Quartile'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm_df['M_Quartile'] = pd.qcut(rfm_df['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)

        rfm_df['rfm_score'] = rfm_df['R_Quartile'].astype(str) + rfm_df['F_Quartile'].astype(str) + rfm_df['M_Quartile'].astype(str)

        def map_segment(row):
            score = int(row['rfm_score'])
            if score >= 444: return 'VIP Customers'
            elif score >= 333: return 'Loyal Customers'
            elif row['R_Quartile'] <= 2: return 'At-Risk Customers'
            elif score <= 222: return 'Inactive Customers'
            else: return 'Regular Customers'

        rfm_df['segment_name'] = rfm_df.apply(map_segment, axis=1)

        # Clear and Insert
        execute_query("DELETE FROM customer_segments")
        segments_to_insert = rfm_df[['customer_id', 'segment_name', 'rfm_score', 'recency', 'frequency', 'monetary']]
        bulk_insert(segments_to_insert, "customer_segments")
        logger.info("RFM Segmentation complete.")

    def predict_churn(self):
        logger.info("Training Churn Prediction Model...")
        query = """
            SELECT 
                c.customer_id,
                MAX(o.order_date) AS last_order_date,
                COUNT(o.order_id) AS total_orders,
                SUM(o.total_amount) AS total_spend,
                AVG(o.total_amount) AS avg_order_value
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            WHERE o.status = 'Completed'
            GROUP BY c.customer_id
        """
        df = fetch_data(query)
        if len(df) < 50:
            logger.warning("Not enough data to train churn model.")
            return
            
        last_dt = pd.to_datetime(df['last_order_date'], errors='coerce')
        snapshot_date = last_dt.max() + pd.Timedelta(days=1) if last_dt.notna().any() else pd.Timestamp.now()
        days_since = (snapshot_date - last_dt.fillna(snapshot_date - pd.Timedelta(days=999))).dt.days
        df['is_churned'] = (days_since > 90).astype(int)
        
        features = ['total_orders', 'total_spend', 'avg_order_value']
        X = df[features].fillna(0)
        y = df['is_churned']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Churn Model Accuracy: {acc:.2f}")

    def predict_cltv(self):
        logger.info("Training CLTV Prediction Model...")
        query = """
            SELECT 
                customer_id,
                substr(order_date, 1, 7) as month,
                SUM(total_amount) as amount
            FROM orders
            WHERE status = 'Completed'
            GROUP BY customer_id, substr(order_date, 1, 7)
        """
        df = fetch_data(query)
        if len(df) < 50:
            logger.warning("Not enough data to train CLTV model.")
            return
            
        summary = df.groupby('customer_id').agg(
            active_months=('month', 'nunique'),
            avg_monthly_spend=('amount', 'mean'),
            total_12m_spend=('amount', 'sum')
        ).reset_index()
        
        X = summary[['active_months', 'avg_monthly_spend']].fillna(0)
        y = summary['total_12m_spend'].fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        logger.info(f"CLTV Model MAE: £{mae:.2f}")

if __name__ == "__main__":
    ci = CustomerIntelligence()
    ci.update_rfm_segments()
    ci.predict_churn()
    ci.predict_cltv()
