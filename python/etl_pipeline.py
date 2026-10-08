import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from config import NUM_CUSTOMERS, NUM_PRODUCTS, NUM_ORDERS, setup_logging, test_db_connection, is_sqlite
from utils import get_engine, fetch_data, bulk_insert, execute_query, refresh_materialized_views

logger = setup_logging("ETL_Pipeline")
fake = Faker('en_GB')

REALISTIC_PRODUCTS = [
    ("WHITE HANGING HEART T-LIGHT HOLDER", 1, 1.25, 2.95),
    ("REGENCY CAKESTAND 3 TIER", 2, 4.50, 12.75),
    ("JUMBO BAG RED RETROSPOT", 4, 0.85, 2.08),
    ("PARTY BUNTING", 4, 2.10, 4.95),
    ("LUNCH BAG RED RETROSPOT", 2, 0.75, 1.65),
    ("SET OF 3 CAKE TINS PANTRY DESIGN", 2, 3.20, 8.95),
    ("HEART OF WICKER SMALL", 1, 0.90, 2.10),
    ("HEART OF WICKER LARGE", 1, 1.80, 4.25),
    ("ASSORTED COLOUR BIRD ORNAMENT", 1, 0.65, 1.69),
    ("PACK OF 72 RETROSPOT CAKE CASES", 2, 0.35, 0.95),
    ("VICTORIAN GLASS HANGING T-LIGHT", 1, 1.10, 2.45),
    ("ROSES REGENCY TEACUP AND SAUCER", 2, 2.25, 5.50),
    ("WOODEN PICTURE FRAME WHITE FINISH", 1, 1.60, 3.95),
    ("NATURAL SLATE HEART CHALKBOARD", 1, 1.20, 2.95),
    ("HAND WARMER OWL DESIGN", 3, 0.80, 2.10),
    ("VINTAGE SNAP CARDS", 3, 0.40, 1.25),
    ("POPCORN HOLDER", 4, 0.30, 0.85),
    ("RETROSPOT TEA SET CERAMIC 11 PC", 2, 4.80, 11.95),
    ("BAKING SET 9 PIECE RETROSPOT", 2, 2.50, 6.75),
    ("ANTIQUE SILVER T-LIGHT GLASS", 1, 0.75, 1.95),
    ("PAPER CHAIN KIT 50'S CHRISTMAS", 5, 1.10, 2.95),
    ("SET OF 4 POLKADOT COASTERS", 2, 0.60, 1.50),
    ("MINI PAINTED BIRD HOUSES", 1, 0.85, 2.25),
    ("DOORMAT UNION FLAG", 1, 3.10, 7.95),
    ("CHARLOTTE BAG SUKI DESIGN", 4, 0.45, 1.25),
    ("RED RETROSPOT CHARLOTTE BAG", 4, 0.45, 1.25),
    ("WOODEN STAR CHRISTMAS ORNAMENT", 5, 0.50, 1.45),
    ("VINTAGE HEADS AND TAILS CARD GAME", 3, 0.55, 1.65),
    ("ZINC METAL HEART DECORATION", 1, 0.60, 1.55),
    ("RABBIT NIGHT LIGHT", 1, 1.80, 4.50)
]

def generate_customers(num_customers):
    logger.info(f"Generating {num_customers} customers...")
    data = []
    countries = ["United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"]
    for _ in range(num_customers):
        data.append({
            "name": fake.name(),
            "email": fake.unique.email(),
            "phone": fake.phone_number()[:20],
            "city": fake.city(),
            "state": fake.county() if hasattr(fake, 'county') else "England",
            "country": random.choice(countries),
            "registration_date": fake.date_time_between(start_date='-2y', end_date='now'),
            "is_active": random.choices([True, False], weights=[0.92, 0.08])[0]
        })
    df = pd.DataFrame(data)
    bulk_insert(df, "customers", if_exists="append")
    logger.info("Customers inserted.")

def generate_categories_and_suppliers():
    logger.info("Generating Categories and Suppliers...")
    categories = [
        {"category_name": "Home & Decor", "parent_category_id": None},
        {"category_name": "Kitchenware", "parent_category_id": None},
        {"category_name": "Gifts & Novelties", "parent_category_id": None},
        {"category_name": "Party & Bags", "parent_category_id": None},
        {"category_name": "Seasonal & Holiday", "parent_category_id": None}
    ]
    df_cat = pd.DataFrame(categories)
    bulk_insert(df_cat, "categories", if_exists="append")

    suppliers = []
    uk_cities = ["London", "Manchester", "Birmingham", "Leeds", "Bristol", "Edinburgh"]
    for _ in range(30):
        suppliers.append({
            "supplier_name": fake.company() + " Ltd",
            "contact_email": fake.company_email(),
            "country": "United Kingdom",
            "reliability_score": round(random.uniform(85.0, 99.5), 2)
        })
    df_sup = pd.DataFrame(suppliers)
    bulk_insert(df_sup, "suppliers", if_exists="append")
    logger.info("Categories and Suppliers inserted.")

def generate_products(num_products):
    logger.info(f"Generating {num_products} products based on Online Retail catalog...")
    cat_ids = fetch_data("SELECT category_id FROM categories")['category_id'].tolist()
    sup_ids = fetch_data("SELECT supplier_id FROM suppliers")['supplier_id'].tolist()
    
    if not cat_ids or not sup_ids:
        logger.error("No categories or suppliers found. Cannot generate products.")
        return

    data = []
    # Seed with realistic Online Retail products first
    for name, cat_idx, cost, price in REALISTIC_PRODUCTS:
        actual_cat = cat_ids[min(cat_idx - 1, len(cat_ids) - 1)]
        data.append({
            "product_name": name,
            "category_id": actual_cat,
            "supplier_id": random.choice(sup_ids),
            "price": price,
            "cost_price": cost,
            "stock_quantity": random.randint(100, 3000),
            "reorder_level": random.randint(20, 80)
        })
    
    # Fill remaining products if needed
    for i in range(len(REALISTIC_PRODUCTS), num_products):
        cost_price = round(random.uniform(0.75, 28.0), 2)
        price = round(cost_price * random.uniform(1.4, 2.8), 2)
        data.append({
            "product_name": f"{fake.word().upper()} {random.choice(['DECORATION', 'BOX', 'HOLDER', 'MUG', 'BOWL', 'LIGHT', 'SET'])}",
            "category_id": random.choice(cat_ids),
            "supplier_id": random.choice(sup_ids),
            "price": price,
            "cost_price": cost_price,
            "stock_quantity": random.randint(50, 1500),
            "reorder_level": random.randint(10, 50)
        })
    
    df = pd.DataFrame(data[:num_products])
    bulk_insert(df, "products", if_exists="append")
    
    # Generate Inventory
    prod_ids = fetch_data("SELECT product_id FROM products")['product_id'].tolist()
    inv_data = []
    locations = ["London DC", "Midlands Hub", "North West Hub", "Scotland DC"]
    for pid in prod_ids:
        inv_data.append({
            "product_id": pid,
            "warehouse_location": random.choice(locations),
            "quantity_on_hand": random.randint(100, 2500),
            "quantity_reserved": 0,
            "last_restock_date": datetime.now()
        })
    bulk_insert(pd.DataFrame(inv_data), "inventory", if_exists="append")
    logger.info("Products and Inventory inserted.")

def generate_orders(num_orders):
    logger.info(f"Generating {num_orders} orders in GBP (£)...")
    cust_ids = fetch_data("SELECT customer_id FROM customers")['customer_id'].tolist()
    prod_df = fetch_data("SELECT product_id, price FROM products")
    
    if not cust_ids or prod_df.empty:
        logger.error("Missing customers or products.")
        return

    regions = ["London & South East", "Midlands", "North England", "Scotland & Wales", "International Europe"]
    prod_records = prod_df.to_dict('records')
    
    # Get initial order_id offset
    max_id_res = fetch_data("SELECT COALESCE(MAX(order_id), 0) as max_id FROM orders")
    current_order_id = int(max_id_res.iloc[0]['max_id'])
    
    orders = []
    items = []
    payments = []
    refunds = []
    
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_orders):
        current_order_id += 1
        oid = current_order_id
        
        # Fraud spike chance (2%)
        is_fraud = random.random() < 0.02
        cid = random.randint(1, 15) if is_fraud else random.choice(cust_ids)
        num_items = random.randint(1, 4)
        order_total = 0.0
        
        for _ in range(num_items):
            p = random.choice(prod_records)
            qty = random.randint(30, 80) if is_fraud else random.randint(1, 5)
            unit_price = float(p['price'])
            line_tot = qty * unit_price
            order_total += line_tot
            items.append({
                "order_id": oid,
                "product_id": int(p['product_id']),
                "quantity": qty,
                "unit_price": unit_price,
                "discount": 0.0,
                "line_total": line_tot
            })
            
        order_date = fake.date_time_between(start_date=start_date, end_date='now')
        status = random.choices(['Completed', 'Pending', 'Cancelled'], weights=[0.90, 0.07, 0.03])[0]
        
        orders.append({
            "order_id": oid,
            "customer_id": cid,
            "order_date": order_date,
            "status": status,
            "total_amount": round(order_total, 2),
            "shipping_address": fake.address().replace('\n', ', '),
            "region": random.choice(regions)
        })
        
        payments.append({
            "order_id": oid,
            "payment_method": "Credit Card" if is_fraud else random.choice(["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]),
            "amount": round(order_total, 2),
            "payment_date": order_date,
            "status": "Completed" if status == "Completed" else "Pending",
            "transaction_ref": fake.uuid4()
        })
        
        # Refunds (4% of completed orders, high for fraud)
        if status == "Completed" and (is_fraud or random.random() < 0.04):
            refunds.append({
                "order_id": oid,
                "payment_id": None,
                "refund_amount": round(order_total, 2),
                "reason": "Suspected velocity abuse" if is_fraud else random.choice(["Defective item", "Changed mind", "Damaged in transit"]),
                "refund_date": order_date + timedelta(days=random.randint(1, 14)),
                "status": "Processed"
            })
            
    logger.info(f"Writing {len(orders)} orders, {len(items)} items, {len(payments)} payments to database...")
    bulk_insert(pd.DataFrame(orders), "orders", if_exists="append")
    bulk_insert(pd.DataFrame(items), "order_items", if_exists="append")
    bulk_insert(pd.DataFrame(payments), "payments", if_exists="append")
    if refunds:
        bulk_insert(pd.DataFrame(refunds), "refunds", if_exists="append")
    logger.info("All orders and transactions inserted successfully.")


def generate_customer_segments():
    logger.info("Generating customer RFM segments in GBP...")
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
        logger.info("No customers found for segmentation generation.")
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
        if row["frequency"] >= 18 or row["monetary"] >= 1500:
            return "Platinum"
        if row["frequency"] >= 10 or row["monetary"] >= 800:
            return "Gold"
        if row["frequency"] >= 4 or row["monetary"] >= 300:
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
    logger.info("Customer segments generated.")


def generate_fraud_logs():
    logger.info("Generating ML fraud anomaly logs...")
    try:
        from fraud_detection import FraudDetector
        detector = FraudDetector()
        detector.detect_fraud()
    except Exception as e:
        logger.warning(f"Could not run real-time fraud detection during ETL: {e}")


def run_etl():
    if not test_db_connection():
        logger.error("Database connection failed. Please ensure PostgreSQL or SQLite is initialized.")
        return

    # Check if authentic UCI Online Retail II dataset is present
    import os
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "online_retail_II.xlsx")
    if os.path.exists(dataset_path):
        logger.info(f"Authentic UCI dataset detected at {dataset_path}. Ingesting real retail data warehouse...")
        from etl_real_data import load_and_ingest_real_retail
        load_and_ingest_real_retail()
        from customer_intelligence import CustomerIntelligence
        ci = CustomerIntelligence()
        ci.update_rfm_segments()
        generate_fraud_logs()
        if not is_sqlite():
            refresh_materialized_views()
        logger.info("Real Dataset ETL Pipeline execution completed successfully!")
        return

    logger.info("Starting Standard ETL Pipeline...")
    generate_categories_and_suppliers()
    generate_products(NUM_PRODUCTS)
    generate_customers(NUM_CUSTOMERS)
    generate_orders(NUM_ORDERS)
    generate_customer_segments()
    generate_fraud_logs()

    if not is_sqlite():
        refresh_materialized_views()

    logger.info("ETL Pipeline execution completed successfully!")

if __name__ == "__main__":
    run_etl()
