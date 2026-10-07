import os
import sys

# Ensure root directory and python directory are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from python.web_dashboard import app, fetch_dashboard_metrics, get_scaled_analytics

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    print(f"QuantivaIQ Retail Intelligence Platform running on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)


