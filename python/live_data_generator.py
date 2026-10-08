try:
    import schedule
except ImportError:
    schedule = None

import random
from faker import Faker
import pandas as pd
from datetime import datetime
import time

try:
    from .config import setup_logging, SIMULATION_INTERVAL, FRAUD_RATE, test_db_connection, is_sqlite
    from .utils import get_engine, fetch_data, bulk_insert, execute_query, refresh_materialized_views
except ImportError:
    from config import setup_logging, SIMULATION_INTERVAL, FRAUD_RATE, test_db_connection, is_sqlite
    from utils import get_engine, fetch_data, bulk_insert, execute_query, refresh_materialized_views


logger = setup_logging("LiveSimulator")
fake = Faker('en_GB')

class LiveSimulator:
    def __init__(self):
        logger.info("Initializing Live Simulator...")
        all_c = fetch_data("SELECT customer_id FROM customers")['customer_id'].tolist()
        self.cust_ids = [c for c in all_c if c != 99999]
        self.prod_df = fetch_data("SELECT product_id, price FROM products")
        
        if not self.cust_ids or self.prod_df.empty:
            logger.error("Database empty! Run ETL pipeline first.")
            raise ValueError("Empty database")

    def simulate_traffic(self):
        try:
            logger.info("Simulating live transaction cycle in GBP (£)...")
            
            # 1. New Sessions
            sessions = []
            for _ in range(random.randint(1, 5)):
                sessions.append({
                    "customer_id": random.choice(self.cust_ids),
                    "login_time": datetime.now(),
                    "logout_time": None,
                    "device_type": random.choice(["Mobile", "Desktop", "Tablet"]),
                    "ip_address": fake.ipv4(),
                    "pages_viewed": random.randint(1, 20)
                })
            bulk_insert(pd.DataFrame(sessions), "customer_sessions")
            
            # 2. Orders & Payments
            orders = []
            order_items_list = []
            payments = []
            num_orders = random.randint(1, 3)
            regions = ["United Kingdom", "Germany", "France", "EIRE", "Netherlands", "Australia", "Spain", "Switzerland"]

            for _ in range(num_orders):
                cid = random.choice(self.cust_ids)
                is_fraud = random.random() < FRAUD_RATE
                num_items = random.randint(1, 3)
                order_total = 0.0

                # Pre-calculate line items & total amount in GBP
                temp_items = []
                for _ in range(num_items):
                    p = self.prod_df.sample(1).iloc[0]
                    qty = random.randint(40, 100) if is_fraud else random.randint(1, 4)
                    unit_price = float(p['price'])
                    line_tot = qty * unit_price
                    order_total += line_tot
                    temp_items.append({
                        "product_id": int(p['product_id']),
                        "quantity": qty,
                        "unit_price": unit_price,
                        "discount": 0.0,
                        "line_total": line_tot
                    })

                orders.append({
                    "customer_id": cid,
                    "order_date": datetime.now(),
                    "status": "Completed",
                    "total_amount": round(order_total, 2),
                    "shipping_address": fake.address().replace('\n', ', '),
                    "region": random.choice(regions),
                    "_items": temp_items,
                    "_is_fraud": is_fraud
                })

            if orders:
                df_orders = pd.DataFrame([{k: v for k, v in o.items() if not k.startswith('_')} for o in orders])
                try:
                    engine = get_engine()
                    with engine.begin() as conn:
                        df_orders.to_sql('orders', conn, if_exists='append', index=False, method=None if is_sqlite() else 'multi')
                    recent_order_ids = fetch_data(f"SELECT order_id FROM orders ORDER BY order_id DESC LIMIT {num_orders}")['order_id'].tolist()
                    
                    for idx, oid in enumerate(reversed(recent_order_ids)):
                        if idx < len(orders):
                            ord_info = orders[idx]
                            for it in ord_info['_items']:
                                it['order_id'] = oid
                                order_items_list.append(it)
                            payments.append({
                                "order_id": oid,
                                "payment_method": "Credit Card" if ord_info['_is_fraud'] else random.choice(["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]),
                                "amount": ord_info['total_amount'],
                                "payment_date": datetime.now(),
                                "status": "Completed",
                                "transaction_ref": fake.uuid4()
                            })
                    if order_items_list:
                        bulk_insert(pd.DataFrame(order_items_list), "order_items")
                    if payments:
                        bulk_insert(pd.DataFrame(payments), "payments")
                except Exception as exc:
                    logger.warning(f"Live simulation DB write skipped (read-only mode active): {exc}")

                if not is_sqlite():
                    refresh_materialized_views()
                    logger.info("Power BI materialized views refreshed after live simulation cycle.")
                logger.info(f"Inserted {num_orders} live orders in GBP.")
                
        except Exception as e:
            logger.error(f"Error during simulation cycle: {e}")

def run_simulator():
    if not test_db_connection():
        logger.error("Unable to connect to the configured database. Run db_setup.py and verify .env configuration before running live_data_generator.py.")
        return

    simulator = LiveSimulator()
    
    if schedule:
        schedule.every(SIMULATION_INTERVAL).seconds.do(simulator.simulate_traffic)
        logger.info(f"Live Simulation started. Running every {SIMULATION_INTERVAL} seconds...")
        while True:
            schedule.run_pending()
            time.sleep(1)
    else:
        logger.info("Running single live simulation step...")
        simulator.simulate_traffic()

if __name__ == "__main__":
    run_simulator()
