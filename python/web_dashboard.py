import os
import subprocess
import sys
import io
import csv
import threading
import time
import random
from datetime import datetime
from flask import Flask, render_template, jsonify, Response, request

try:
    from .config import setup_logging, test_db_connection
    from .utils import fetch_data
except ImportError:
    from config import setup_logging, test_db_connection
    from utils import fetch_data

try:
    from .analytics import get_analytics
except Exception:
    try:
        from analytics import get_analytics
    except Exception:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from analytics import get_analytics

logger = setup_logging("WebDashboard")

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=TEMPLATE_DIR)


def format_gbp(amount):
    """Format numeric amount into British Pound (£ GBP) representation."""
    if amount is None:
        return "£0.00"
    amount = float(amount)
    if amount >= 1_000_000:
        return f"£{amount / 1_000_000:.2f}M"
    elif amount >= 10_000:
        return f"£{amount / 1_000:.1f}k"
    else:
        return f"£{amount:,.2f}"


def fetch_dashboard_metrics():
    logger.info("Fetching operational metrics in GBP (£)...")
    try:
        total_cust = int(fetch_data("SELECT COUNT(*) AS c FROM customers").iloc[0]['c'])
        total_ord = int(fetch_data("SELECT COUNT(*) AS c FROM orders").iloc[0]['c'])
        rev_res = fetch_data("SELECT COALESCE(SUM(total_amount), 0) AS r FROM orders WHERE status = 'Completed'")
        raw_revenue = float(rev_res.iloc[0]['r']) if not rev_res.empty else 0.0
        total_revenue = raw_revenue
        avg_order_value = (total_revenue / total_ord) if total_ord > 0 else 0.0

        fraud_cases = int(fetch_data("SELECT COUNT(*) AS c FROM fraud_logs").iloc[0]['c'])

        top_prod_df = fetch_data(
            "SELECT p.product_name, SUM(oi.line_total) AS revenue, SUM(oi.quantity) AS units_sold "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
            "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 8"
        )
        top_products = []
        for _, r in top_prod_df.iterrows():
            rev = float(r['revenue'])
            top_products.append({
                "product_name": str(r['product_name']),
                "revenue": rev,
                "revenue_formatted": format_gbp(rev),
                "units_sold": int(r['units_sold'])
            })

        region_df = fetch_data(
            "SELECT region, COALESCE(SUM(total_amount), 0) AS revenue, COUNT(order_id) as order_count "
            "FROM orders WHERE status = 'Completed' GROUP BY region ORDER BY revenue DESC"
        )
        revenue_by_region = []
        for _, r in region_df.iterrows():
            rev = float(r['revenue'])
            revenue_by_region.append({
                "region": str(r['region']),
                "revenue": rev,
                "revenue_formatted": format_gbp(rev),
                "order_count": int(r['order_count'])
            })

        segment_df = fetch_data(
            "SELECT segment_name, COUNT(*) AS customers, COALESCE(AVG(monetary), 0) as avg_spend "
            "FROM customer_segments GROUP BY segment_name ORDER BY customers DESC"
        )
        segments = []
        for _, r in segment_df.iterrows():
            spend = float(r['avg_spend'])
            segments.append({
                "segment_name": str(r['segment_name']),
                "customers": int(r['customers']),
                "avg_spend": spend,
                "avg_spend_formatted": format_gbp(spend)
            })

        recent_df = fetch_data(
            "SELECT o.order_id, o.customer_id, o.order_date, o.status, o.total_amount, o.region, "
            "COALESCE(p.payment_method, 'Credit Card') as payment_method "
            "FROM orders o LEFT JOIN payments p ON o.order_id = p.order_id "
            "ORDER BY o.order_date DESC, o.order_id DESC LIMIT 12"
        )
        recent_orders = []
        for _, r in recent_df.iterrows():
            amt = float(r['total_amount'])
            recent_orders.append({
                "order_id": str(r['order_id']),
                "customer_id": int(r['customer_id']),
                "order_date": str(r['order_date']),
                "status": str(r['status']),
                "amount": amt,
                "amount_formatted": format_gbp(amt),
                "region": str(r['region']),
                "payment_method": str(r['payment_method'])
            })

        return {
            'total_customers': total_cust,
            'total_orders': total_ord,
            'total_revenue': total_revenue,
            'total_revenue_formatted': format_gbp(raw_revenue),
            'avg_order_value': avg_order_value,
            'avg_order_value_formatted': format_gbp(avg_order_value),
            'fraud_cases': fraud_cases,
            'currency': 'GBP',
            'currency_symbol': '£',
            'top_products': top_products,
            'revenue_by_region': revenue_by_region,
            'segments': segments,
            'recent_orders': recent_orders
        }
    except Exception as exc:
        logger.error(f"Error fetching dashboard metrics: {exc}")
        return {
            'total_customers': 0,
            'total_orders': 0,
            'total_revenue': 0.0,
            'total_revenue_formatted': "£0.00",
            'avg_order_value': 0.0,
            'avg_order_value_formatted': "£0.00",
            'fraud_cases': 0,
            'currency': 'GBP',
            'currency_symbol': '£',
            'top_products': [],
            'revenue_by_region': [],
            'segments': [],
            'recent_orders': []
        }


def get_scaled_analytics():
    """Fetch Machine Learning analytics and format values for GBP."""
    analytics = get_analytics()
    if not analytics or "error" in analytics:
        return analytics
    
    # Format fraud amounts
    if "fraud" in analytics and "top" in analytics["fraud"]:
        for item in analytics["fraud"]["top"]:
            if "avg" in item:
                item["avg_formatted"] = format_gbp(item["avg"])

    # Format forecast amounts
    if "forecast" in analytics:
        fc = analytics["forecast"]
        if "history" in fc:
            fc["history_formatted"] = [format_gbp(v) for v in fc.get("history", [])]
        if "forecast" in fc:
            fc["forecast_formatted"] = [format_gbp(v) for v in fc.get("forecast", [])]

    return analytics


@app.route('/health')
def health_check():
    """Health check endpoint for Render, Docker, and monitoring services."""
    return jsonify({
        "status": "healthy",
        "service": "QuantivaIQ",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/')
def index():
    try:
        db_ok, db_error = test_db_connection()
        if not db_ok:
            return render_template('index.html', db_ok=False, db_error=db_error)

        metrics = fetch_dashboard_metrics()
        analytics = get_scaled_analytics()

        return render_template(
            'index.html',
            db_ok=True,
            db_error="",
            pbi_available=False,
            metrics=metrics,
            analytics=analytics,
            **metrics
        )
    except Exception as exc:
        logger.error(f"Error in index route: {exc}")
        return render_template('index.html', db_ok=False, db_error=str(exc))


@app.route('/api/metrics')
def api_metrics():
    # In serverless environments (Vercel) or when tick requested, simulate incoming stream traffic
    if os.getenv("VERCEL") or request.args.get("tick") == "1" or random.random() < 0.35:
        try:
            try:
                from .live_data_generator import LiveSimulator
            except ImportError:
                from live_data_generator import LiveSimulator
            LiveSimulator().simulate_traffic()
        except Exception as e:
            logger.debug(f"Serverless simulation note: {e}")

    metrics = fetch_dashboard_metrics()
    analytics = get_scaled_analytics()
    return jsonify({**metrics, "analytics": analytics})


@app.route('/api/fraud')
def api_fraud():
    analytics = get_scaled_analytics()
    return jsonify(analytics.get("fraud", {}))


@app.route('/api/forecast')
def api_forecast():
    analytics = get_scaled_analytics()
    return jsonify(analytics.get("forecast", {}))


@app.route('/api/segments')
def api_segments():
    analytics = get_scaled_analytics()
    return jsonify(analytics.get("segments", {}))


@app.route('/api/recommendations')
def api_recommendations():
    analytics = get_scaled_analytics()
    return jsonify(analytics.get("recs", []))


@app.route('/api/recent-orders')
def api_recent_orders():
    metrics = fetch_dashboard_metrics()
    return jsonify(metrics.get("recent_orders", []))


@app.route('/api/simulator/trigger', methods=['POST'])
def api_trigger_simulator():
    """Triggers a single live simulated transaction batch for real-time demonstration."""
    try:
        from .live_data_generator import LiveSimulator
    except ImportError:
        from live_data_generator import LiveSimulator
    try:
        sim = LiveSimulator()
        sim.simulate_traffic()
        return jsonify({
            'status': 'success',
            'message': 'Simulated new transactions successfully injected into database'
        })
    except Exception as exc:
        logger.error(f"Simulation trigger failed: {exc}")
        return jsonify({
            'status': 'error',
            'message': str(exc)
        }), 500


@app.route('/api/export/csv')
def api_export_csv():
    """Streams a dynamic CSV report containing key executive KPI metrics."""
    metrics = fetch_dashboard_metrics()
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["=== QUANTIVAIQ RETAIL ANALYTICS EXECUTIVE REPORT ==="])
    writer.writerow(["Currency", "GBP (£)"])
    writer.writerow(["Total Customers", metrics["total_customers"]])
    writer.writerow(["Total Orders", metrics["total_orders"]])
    writer.writerow(["Total Revenue (GBP)", f"£{metrics['total_revenue']:,.2f}"])
    writer.writerow(["Avg Order Value (GBP)", f"£{metrics['avg_order_value']:,.2f}"])
    writer.writerow(["Flagged Anomaly Cases", metrics["fraud_cases"]])
    writer.writerow([])

    writer.writerow(["=== TOP PRODUCTS ==="])
    writer.writerow(["Product Name", "Units Sold", "Revenue (GBP)"])
    for p in metrics["top_products"]:
        writer.writerow([p["product_name"], p["units_sold"], f"£{p['revenue']:,.2f}"])
    writer.writerow([])

    writer.writerow(["=== REVENUE BY REGION ==="])
    writer.writerow(["Region", "Orders Count", "Revenue (GBP)"])
    for r in metrics["revenue_by_region"]:
        writer.writerow([r["region"], r["order_count"], f"£{r['revenue']:,.2f}"])
    writer.writerow([])

    writer.writerow(["=== CUSTOMER SEGMENTS (RFM) ==="])
    writer.writerow(["Segment Name", "Customer Count", "Avg Spend (GBP)"])
    for s in metrics["segments"]:
        writer.writerow([s["segment_name"], s["customers"], f"£{s['avg_spend']:,.2f}"])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=quantivaiq_retail_report.csv"}
    )


# ---------------------------------------------------------
# Background Live Simulator Thread for Continuous Streaming
# ---------------------------------------------------------
_simulator_started = False
_simulator_lock = threading.Lock()

def start_background_simulator():
    global _simulator_started
    with _simulator_lock:
        if _simulator_started:
            return
        _simulator_started = True

    def _worker():
        interval = int(os.getenv("SIMULATION_INTERVAL_SECONDS", 6))
        logger.info(f"Background live simulator daemon started (interval: {interval}s)")
        time.sleep(2)
        while True:
            try:
                try:
                    from .live_data_generator import LiveSimulator
                except ImportError:
                    from live_data_generator import LiveSimulator
                sim = LiveSimulator()
                sim.simulate_traffic()
            except Exception as e:
                logger.debug(f"Background simulation daemon step note: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_worker, daemon=True, name="LiveSimulatorDaemon")
    t.start()


if os.getenv("RUN_SIMULATOR", "1") == "1":
    try:
        start_background_simulator()
    except Exception as e:
        logger.warning(f"Could not start background simulator: {e}")


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
