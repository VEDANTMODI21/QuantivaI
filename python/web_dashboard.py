import os
import subprocess
import sys
import io
import csv
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



# Indian Rupee (INR) configuration
INR_RATE = float(os.getenv("INR_RATE", "1.0"))


def format_inr(amount):
    """Format numeric amount into Indian Rupee (INR) representation."""
    if amount is None:
        return "₹0.00"
    amount = float(amount) * INR_RATE
    # Format according to standard Indian numbering (Lakhs / Crores) for large numbers
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f} L"
    else:
        return f"₹{amount:,.2f}"


def fetch_dashboard_metrics():
    logger.info("Fetching operational metrics in INR...")
    try:
        total_cust = int(fetch_data("SELECT COUNT(*) AS c FROM customers").iloc[0]['c'])
        total_ord = int(fetch_data("SELECT COUNT(*) AS c FROM orders").iloc[0]['c'])
        rev_res = fetch_data("SELECT COALESCE(SUM(total_amount), 0) AS r FROM orders WHERE status = 'Completed'")
        raw_revenue = float(rev_res.iloc[0]['r']) if not rev_res.empty else 0.0
        total_revenue = raw_revenue * INR_RATE
        avg_order_value = (total_revenue / total_ord) if total_ord > 0 else 0.0

        fraud_cases = int(fetch_data("SELECT COUNT(*) AS c FROM fraud_logs").iloc[0]['c'])

        top_prod_df = fetch_data(
            "SELECT p.product_name, SUM(oi.line_total) AS revenue, SUM(oi.quantity) AS units_sold "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
            "GROUP BY p.product_name ORDER BY revenue DESC LIMIT 8"
        )
        top_products = []
        for _, r in top_prod_df.iterrows():
            rev = float(r['revenue']) * INR_RATE
            top_products.append({
                "product_name": str(r['product_name']),
                "revenue": rev,
                "revenue_formatted": format_inr(rev / INR_RATE),
                "units_sold": int(r['units_sold'])
            })

        region_df = fetch_data(
            "SELECT region, COALESCE(SUM(total_amount), 0) AS revenue, COUNT(order_id) as order_count "
            "FROM orders WHERE status = 'Completed' GROUP BY region ORDER BY revenue DESC"
        )
        revenue_by_region = []
        for _, r in region_df.iterrows():
            rev = float(r['revenue']) * INR_RATE
            revenue_by_region.append({
                "region": str(r['region']),
                "revenue": rev,
                "revenue_formatted": format_inr(rev / INR_RATE),
                "order_count": int(r['order_count'])
            })

        segment_df = fetch_data(
            "SELECT segment_name, COUNT(*) AS customers, COALESCE(AVG(monetary), 0) as avg_spend "
            "FROM customer_segments GROUP BY segment_name ORDER BY customers DESC"
        )
        segments = []
        for _, r in segment_df.iterrows():
            spend = float(r['avg_spend']) * INR_RATE
            segments.append({
                "segment_name": str(r['segment_name']),
                "customers": int(r['customers']),
                "avg_spend": spend,
                "avg_spend_formatted": format_inr(spend / INR_RATE)
            })

        recent_df = fetch_data(
            "SELECT o.order_id, o.customer_id, o.order_date, o.status, o.total_amount, o.region, "
            "COALESCE(p.payment_method, 'Credit Card') as payment_method "
            "FROM orders o LEFT JOIN payments p ON o.order_id = p.order_id "
            "ORDER BY o.order_date DESC LIMIT 10"
        )
        recent_orders = []
        for _, r in recent_df.iterrows():
            amt = float(r['total_amount']) * INR_RATE
            recent_orders.append({
                "order_id": str(r['order_id']),
                "customer_id": int(r['customer_id']),
                "order_date": str(r['order_date']),
                "status": str(r['status']),
                "amount": amt,
                "amount_formatted": format_inr(amt / INR_RATE),
                "region": str(r['region']),
                "payment_method": str(r['payment_method'])
            })

        return {
            'total_customers': total_cust,
            'total_orders': total_ord,
            'total_revenue': total_revenue,
            'total_revenue_formatted': format_inr(raw_revenue),
            'avg_order_value': avg_order_value,
            'avg_order_value_formatted': format_inr(avg_order_value / INR_RATE),
            'fraud_cases': fraud_cases,
            'currency': 'INR',
            'currency_symbol': '₹',
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
            'total_revenue_formatted': "₹0.00",
            'avg_order_value': 0.0,
            'avg_order_value_formatted': "₹0.00",
            'fraud_cases': 0,
            'currency': 'INR',
            'currency_symbol': '₹',
            'top_products': [],
            'revenue_by_region': [],
            'segments': [],
            'recent_orders': []
        }


def get_scaled_analytics():
    """Fetch Machine Learning analytics and format values for INR."""
    analytics = get_analytics()
    if not analytics or "error" in analytics:
        return analytics
    
    # Scale fraud amounts
    if "fraud" in analytics and "top" in analytics["fraud"]:
        for item in analytics["fraud"]["top"]:
            if "avg" in item:
                item["avg_formatted"] = format_inr(item["avg"])

    # Scale forecast amounts
    if "forecast" in analytics:
        fc = analytics["forecast"]
        if "history" in fc:
            fc["history_formatted"] = [format_inr(v) for v in fc.get("history", [])]
        if "forecast" in fc:
            fc["forecast_formatted"] = [format_inr(v) for v in fc.get("forecast", [])]

    return analytics


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
        sim = LiveSimulator()
        sim.simulate_traffic()
        return jsonify({"status": "success", "message": "Simulated new transactions successfully!"})
    except Exception as exc:
        logger.error(f"Simulator trigger error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/export/csv')
def api_export_csv():
    """Generates and streams a CSV summary report in INR currency."""
    metrics = fetch_dashboard_metrics()
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["QuantivaIQ Retail Intelligence Summary Report (INR Currency)"])
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Total Revenue (INR)", metrics.get("total_revenue_formatted")])
    writer.writerow(["Total Orders", metrics.get("total_orders")])
    writer.writerow(["Total Customers", metrics.get("total_customers")])
    writer.writerow(["Average Order Value (INR)", metrics.get("avg_order_value_formatted")])
    writer.writerow(["Fraud Cases Flagged", metrics.get("fraud_cases")])
    writer.writerow([])

    writer.writerow(["Top Products by Revenue (INR)"])
    writer.writerow(["Product Name", "Units Sold", "Revenue (INR)"])
    for p in metrics.get("top_products", []):
        writer.writerow([p["product_name"], p["units_sold"], p["revenue_formatted"]])
    writer.writerow([])

    writer.writerow(["Revenue by Region (INR)"])
    writer.writerow(["Region", "Orders", "Revenue (INR)"])
    for r in metrics.get("revenue_by_region", []):
        writer.writerow([r["region"], r["order_count"], r["revenue_formatted"]])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=quantivaiq_retail_report_inr.csv"}
    )


@app.route('/health')
def health():
    db_available, db_err = test_db_connection()
    return {'status': 'ok', 'db_available': db_available, 'error': db_err, 'currency': 'INR'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 8000)), debug=False)

