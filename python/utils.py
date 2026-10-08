import pandas as pd
from sqlalchemy import text

try:
    from config import get_engine, setup_logging, is_sqlite
except ImportError:
    from .config import get_engine, setup_logging, is_sqlite

logger = setup_logging("Utils")

def execute_query(query, params=None):
    """Executes a SQL query without returning results (e.g., INSERT, UPDATE, DELETE)."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text(query), params or {})
            logger.debug("Query executed successfully.")
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise

def fetch_data(query, params=None):
    """Executes a SQL query and returns results as a pandas DataFrame."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
            return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

def bulk_insert(df, table_name, if_exists="append"):
    """Inserts a pandas DataFrame into a SQL table efficiently."""
    if df.empty:
        logger.warning(f"Empty DataFrame. Skipping insert for table {table_name}.")
        return
        
    engine = get_engine()
    try:
        if is_sqlite():
            df.to_sql(table_name, engine, if_exists=if_exists, index=False, method=None, chunksize=10000)
        else:
            df.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi', chunksize=1000)
        logger.info(f"Successfully inserted {len(df)} rows into {table_name}.")
    except Exception as e:
        logger.error(f"Error during bulk insert to {table_name}: {e}")
        raise


def refresh_materialized_views():
    """Refresh PostgreSQL materialized views used by Power BI dashboards."""
    if is_sqlite():
        logger.info("SQLite mode detected; skipping materialized view refresh.")
        return

    views = [
        "mv_daily_revenue",
        "mv_customer_kpis",
        "mv_product_performance",
        "mv_fraud_summary",
        "mv_inventory_status",
    ]

    for view in views:
        try:
            execute_query(f"REFRESH MATERIALIZED VIEW {view}")
            logger.info(f"Refreshed materialized view: {view}")
        except Exception as exc:
            logger.warning(f"Could not refresh {view}: {exc}")


def refresh_customer_segments():
    """Recalculate customer segmentation and refresh the customer_segments table."""
    logger.info("Refreshing customer segments...")
    customer_metrics = fetch_data(
        """
        SELECT
            c.customer_id,
            COUNT(o.order_id) AS total_orders,
            COALESCE(SUM(o.total_amount), 0) AS monetary,
            MAX(o.order_date) AS last_order_date
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'Completed'
        GROUP BY c.customer_id
        """
    )

    if customer_metrics.empty:
        logger.info("No customers found for segmentation refresh.")
        return

    customer_metrics["last_order_date"] = pd.to_datetime(
        customer_metrics["last_order_date"]
    ).fillna(pd.Timestamp.now() - pd.Timedelta(days=999))
    customer_metrics["recency"] = (
        pd.Timestamp.now() - customer_metrics["last_order_date"]
    ).dt.days.clip(lower=0)
    customer_metrics["frequency"] = customer_metrics["total_orders"].fillna(0).astype(int)
    customer_metrics["monetary"] = customer_metrics["monetary"].fillna(0.0)

    def choose_segment(row):
        if row["frequency"] >= 20 or row["monetary"] >= 5000:
            return "Platinum"
        if row["frequency"] >= 10 or row["monetary"] >= 2500:
            return "Gold"
        if row["frequency"] >= 4 or row["monetary"] >= 1000:
            return "Silver"
        return "Bronze"

    customer_metrics["segment_name"] = customer_metrics.apply(choose_segment, axis=1)
    customer_metrics["rfm_score"] = (
        customer_metrics["recency"].rank(method="dense", ascending=False).astype(int).astype(str)
        + "-"
        + customer_metrics["frequency"].rank(method="dense", ascending=True).astype(int).astype(str)
        + "-"
        + customer_metrics["monetary"].rank(method="dense", ascending=True).astype(int).astype(str)
    )

    execute_query("DELETE FROM customer_segments")
    bulk_insert(
        customer_metrics[
            ["customer_id", "segment_name", "rfm_score", "recency", "frequency", "monetary"]
        ],
        "customer_segments",
        if_exists="append",
    )
    logger.info("Customer segmentation refresh completed.")
