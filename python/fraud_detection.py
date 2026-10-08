import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
try:
    from .config import setup_logging
    from .utils import get_engine, fetch_data, bulk_insert
except ImportError:
    from config import setup_logging
    from utils import get_engine, fetch_data, bulk_insert


logger = setup_logging("FraudDetection")

class FraudDetector:
    def __init__(self):
        logger.info("Initializing Fraud Detection Engine...")

    def extract_features(self):
        logger.info("Extracting features from data warehouse...")
        query = """
            SELECT 
                o.customer_id,
                COUNT(o.order_id) AS total_orders,
                AVG(o.total_amount) AS avg_order_amount,
                MAX(o.total_amount) AS max_order_amount,
                COALESCE(COUNT(DISTINCT substr(o.order_date, 1, 10)), 1) AS active_days
            FROM orders o
            WHERE o.status = 'Completed'
            GROUP BY o.customer_id
        """
        df = fetch_data(query)
        if df.empty:
            return pd.DataFrame()
            
        refunds_query = """
            SELECT o.customer_id, COUNT(r.refund_id) AS total_refunds
            FROM refunds r
            JOIN orders o ON r.order_id = o.order_id
            GROUP BY o.customer_id
        """
        refunds_df = fetch_data(refunds_query)
        df = df.merge(refunds_df, on='customer_id', how='left')
        df['total_refunds'] = df['total_refunds'].fillna(0)
        df['amount_stddev'] = (df['max_order_amount'] - df['avg_order_amount']).clip(lower=0)
        
        # Feature Engineering
        df['refund_ratio'] = df['total_refunds'] / df['total_orders']
        df['order_frequency'] = df['total_orders'] / df['active_days'].clip(lower=1)
        df.fillna(0, inplace=True)
        return df

    def detect_fraud(self):
        df = self.extract_features()
        if df.empty:
            logger.warning("No data available for fraud detection.")
            return

        features = ['avg_order_amount', 'max_order_amount', 'amount_stddev', 'refund_ratio', 'order_frequency']
        X = df[features].astype(float)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info("Running Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        df['iso_outlier'] = iso_forest.fit_predict(X_scaled)
        df['iso_score'] = iso_forest.decision_function(X_scaled) # lower is more abnormal

        logger.info("Running DBSCAN...")
        dbscan = DBSCAN(eps=2.5, min_samples=5)
        df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
        df['dbscan_outlier'] = np.where(df['dbscan_cluster'] == -1, -1, 1)

        logger.info("Running Z-Score Anomaly Detection...")
        mean_vals = X.mean()
        std_vals = X.std().replace(0, 1)
        z_scores = np.abs((X - mean_vals) / std_vals)
        df['zscore_outlier'] = np.where((z_scores > 3).any(axis=1), -1, 1)

        # Ensemble Voting: If 2 or more models flag as -1 (anomaly), then fraud
        df['anomaly_votes'] = (df['iso_outlier'] == -1).astype(int) + \
                              (df['dbscan_outlier'] == -1).astype(int) + \
                              (df['zscore_outlier'] == -1).astype(int)
                              
        fraud_cases = df[df['anomaly_votes'] >= 2].copy()
        
        logger.info(f"Detected {len(fraud_cases)} potential fraud cases.")
        
        if not fraud_cases.empty:
            # Prepare data for fraud_logs
            logs = []
            for _, row in fraud_cases.iterrows():
                # Determine primary reason
                fraud_type = "Complex Anomaly"
                if row['refund_ratio'] > 0.5:
                    fraud_type = "High Refund Abuse"
                elif row['order_frequency'] > 10:
                    fraud_type = "Velocity Fraud / Bot"
                elif row['zscore_outlier'] == -1:
                    fraud_type = "Statistical Outlier"

                # Normalize risk score 0-100 based on isolation forest score
                # iso_score is usually between -0.5 and 0.5. More negative = more abnormal.
                raw_score = row['iso_score']
                risk = min(100, max(0, int((-raw_score + 0.2) * 200))) 
                
                logs.append({
                    "customer_id": int(row['customer_id']),
                    "order_id": None, # Aggregated at customer level for now
                    "fraud_type": fraud_type,
                    "risk_score": risk,
                    "detection_method": "Ensemble ML",
                    "is_confirmed": False
                })
                
            bulk_insert(pd.DataFrame(logs), "fraud_logs")
            logger.info("Fraud logs successfully inserted.")

if __name__ == "__main__":
    detector = FraudDetector()
    detector.detect_fraud()
