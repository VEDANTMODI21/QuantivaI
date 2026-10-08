import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from faker import Faker
from config import setup_logging, is_sqlite
from utils import get_engine, bulk_insert, execute_query
from db_setup import create_sqlite_schema

logger = setup_logging("RealDataETL")
fake = Faker('en_GB')

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
UCI_RETAIL_XLSX = os.path.join(DATASETS_DIR, "online_retail_II.xlsx")
FRAUD_CSV = os.path.join(DATASETS_DIR, "creditcard.csv")

def categorize_product(desc):
    desc = str(desc).upper()
    if any(k in desc for k in ['HEART', 'LIGHT', 'DECORATION', 'WALL', 'FRAME', 'CLOCK', 'MIRROR', 'CANDLE', 'LANTERN', 'SIGN', 'DOORMAT']):
        return 1 # Home & Decor
    elif any(k in desc for k in ['MUG', 'TEA', 'CAKE', 'BOWL', 'PLATE', 'CUP', 'KITCHEN', 'BAKING', 'TIN', 'COASTER', 'CUTLERY', 'JAR', 'BOTTLE']):
        return 2 # Kitchenware
    elif any(k in desc for k in ['CARD', 'GAME', 'TOY', 'PLUSH', 'DOLL', 'PEN', 'PENCIL', 'NOTEBOOK', 'BOOK', 'KEYRING', 'BADGE']):
        return 3 # Gifts & Novelties
    elif any(k in desc for k in ['BAG', 'BOX', 'WRAP', 'TISSUE', 'BUNTING', 'BALLOON', 'RIBBON', 'POPCORN', 'PARTY']):
        return 4 # Party & Bags
    elif any(k in desc for k in ['CHRISTMAS', 'XMAS', 'TREE', 'SANTA', 'SNOW', 'EASTER', 'HALLOWEEN', 'STAR', 'BELL']):
        return 5 # Seasonal & Holiday
    return 1

def load_and_ingest_real_retail(sample_limit=None):
    logger.info("Loading UCI Online Retail II dataset sheets...")
    
    # Read both sheets
    df1 = pd.read_excel(UCI_RETAIL_XLSX, sheet_name='Year 2009-2010')
    logger.info(f"Loaded Year 2009-2010: {len(df1):,} rows")
    
    df2 = pd.read_excel(UCI_RETAIL_XLSX, sheet_name='Year 2010-2011')
    logger.info(f"Loaded Year 2010-2011: {len(df2):,} rows")
    
    df = pd.concat([df1, df2], ignore_index=True)
    logger.info(f"Total Combined Raw Records: {len(df):,} rows")
    
    # Clean column names
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    # Standardize column names
    rename_dict = {
        'customer_id': 'customer_id',
        'invoicedate': 'invoice_date',
        'stockcode': 'stock_code',
        'description': 'description',
        'quantity': 'quantity',
        'price': 'price',
        'country': 'country',
        'invoice': 'invoice'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # Filter valid rows
    df = df.dropna(subset=['description'])
    df['description'] = df['description'].astype(str).str.strip()
    df = df[df['description'] != '']
    df = df[df['price'] > 0]
    
    logger.info(f"Cleaned valid records: {len(df):,} rows")
    
    # Reinitialize SQLite schema
    engine = get_engine()
    create_sqlite_schema(engine)
    
    # 1. Categories & Suppliers
    logger.info("Ingesting Categories and UK Suppliers...")
    categories = [
        {"category_name": "Home & Decor", "parent_category_id": None},
        {"category_name": "Kitchenware", "parent_category_id": None},
        {"category_name": "Gifts & Novelties", "parent_category_id": None},
        {"category_name": "Party & Bags", "parent_category_id": None},
        {"category_name": "Seasonal & Holiday", "parent_category_id": None}
    ]
    bulk_insert(pd.DataFrame(categories), "categories", if_exists="append")
    
    uk_suppliers = [
        {"supplier_name": "Albion Wholesale Imports Ltd", "contact_email": "orders@albionwholesale.co.uk", "country": "United Kingdom", "reliability_score": 98.5},
        {"supplier_name": "British Craft & Living Supplies", "contact_email": "b2b@britishcrafts.co.uk", "country": "United Kingdom", "reliability_score": 97.2},
        {"supplier_name": "Cotswold Giftware Logistics", "contact_email": "supply@cotswoldgiftware.co.uk", "country": "United Kingdom", "reliability_score": 99.1},
        {"supplier_name": "London Artisan Goods Co", "contact_email": "wholesale@londonartisangoods.co.uk", "country": "United Kingdom", "reliability_score": 96.8},
        {"supplier_name": "Midlands Retail Distribution", "contact_email": "info@midlandsretail.co.uk", "country": "United Kingdom", "reliability_score": 95.4},
        {"supplier_name": "Northern Homeware Traders", "contact_email": "trade@northernhomeware.co.uk", "country": "United Kingdom", "reliability_score": 98.0},
        {"supplier_name": "Thames Valley Paper & Party", "contact_email": "accounts@thamesvalleypaper.co.uk", "country": "United Kingdom", "reliability_score": 97.9},
        {"supplier_name": "Edinburgh Highland Goods Ltd", "contact_email": "sales@highlandgoods.co.uk", "country": "United Kingdom", "reliability_score": 96.5}
    ]
    bulk_insert(pd.DataFrame(uk_suppliers), "suppliers", if_exists="append")
    
    # 2. Customers
    logger.info("Ingesting Customers...")
    # Customers with real IDs + Guest Customers for unassigned
    known_customers = df[df['customer_id'].notna()].copy()
    known_customers['customer_id'] = known_customers['customer_id'].astype(int)
    
    cust_summary = known_customers.groupby('customer_id').agg(
        country=('country', lambda x: x.mode()[0] if not x.empty else 'United Kingdom'),
        first_order=('invoice_date', 'min')
    ).reset_index()
    
    cust_rows = []
    fake.seed_instance(42)
    for _, row in cust_summary.iterrows():
        cid = int(row['customer_id'])
        country = str(row['country'])
        name = fake.name()
        cust_rows.append({
            "customer_id": cid,
            "name": name,
            "email": f"customer_{cid}@quantiva-retail.co.uk",
            "phone": fake.phone_number()[:20],
            "city": fake.city() if country == "United Kingdom" else country,
            "state": fake.county() if country == "United Kingdom" and hasattr(fake, 'county') else country,
            "country": country,
            "registration_date": str(pd.to_datetime(row['first_order']) - pd.Timedelta(days=np.random.randint(5, 60))),
            "is_active": 1
        })
    
    # Add Guest customer ID 0 for unassigned invoices
    cust_rows.append({
        "customer_id": 99999,
        "name": "Guest Customer",
        "email": "guest@quantiva-retail.co.uk",
        "phone": "+44 20 7946 0991",
        "city": "London",
        "state": "Greater London",
        "country": "United Kingdom",
        "registration_date": "2009-11-01 00:00:00",
        "is_active": 1
    })
    
    df_customers = pd.DataFrame(cust_rows)
    bulk_insert(df_customers, "customers", if_exists="append")
    logger.info(f"Ingested {len(df_customers):,} genuine customers.")
    
    # 3. Products
    logger.info("Ingesting Products from real catalog...")
    # Group by stock_code to get unique products
    prod_summary = df.groupby('stock_code').agg(
        description=('description', lambda x: x.mode()[0] if not x.empty else 'Retail Item'),
        price=('price', 'median')
    ).reset_index()
    
    prod_rows = []
    inv_rows = []
    stock_to_pid = {}
    locations = ["London DC", "Midlands Hub", "North West Hub", "Scotland DC"]
    
    for idx, row in prod_summary.iterrows():
        pid = idx + 1
        scode = str(row['stock_code'])
        stock_to_pid[scode] = pid
        desc = str(row['description'])[:120]
        price = max(0.25, round(float(row['price']), 2))
        cost = max(0.10, round(price * np.random.uniform(0.40, 0.60), 2))
        cat_id = categorize_product(desc)
        sup_id = (pid % len(uk_suppliers)) + 1
        
        prod_rows.append({
            "product_id": pid,
            "product_name": desc,
            "category_id": cat_id,
            "supplier_id": sup_id,
            "price": price,
            "cost_price": cost,
            "stock_quantity": np.random.randint(50, 3500),
            "reorder_level": np.random.randint(15, 60)
        })
        
        inv_rows.append({
            "inventory_id": pid,
            "product_id": pid,
            "warehouse_location": locations[pid % len(locations)],
            "quantity_on_hand": np.random.randint(200, 4000),
            "quantity_reserved": 0,
            "last_restock_date": "2011-12-01 00:00:00"
        })
    
    df_products = pd.DataFrame(prod_rows)
    bulk_insert(df_products, "products", if_exists="append")
    bulk_insert(pd.DataFrame(inv_rows), "inventory", if_exists="append")
    logger.info(f"Ingested {len(df_products):,} unique genuine products and inventory records.")
    
    # 4. Orders & Order Items
    logger.info("Ingesting Orders and Line Items...")
    df['clean_cust_id'] = df['customer_id'].fillna(99999).astype(int)
    df['stock_code_str'] = df['stock_code'].astype(str)
    df['product_id'] = df['stock_code_str'].map(stock_to_pid).fillna(1).astype(int)
    df['line_total'] = df['quantity'] * df['price']
    df['invoice_str'] = df['invoice'].astype(str)
    
    # Aggregate orders
    order_groups = df.groupby('invoice_str').agg(
        customer_id=('clean_cust_id', 'first'),
        order_date=('invoice_date', 'first'),
        total_amount=('line_total', lambda x: max(0.0, round(x.sum(), 2))),
        country=('country', 'first')
    ).reset_index()
    
    order_groups['order_id'] = np.arange(1, len(order_groups) + 1)
    invoice_to_order_id = dict(zip(order_groups['invoice_str'], order_groups['order_id']))
    
    order_rows = []
    for _, row in order_groups.iterrows():
        inv = row['invoice_str']
        is_cancelled = inv.startswith('C') or inv.startswith('c')
        order_rows.append({
            "order_id": int(row['order_id']),
            "customer_id": int(row['customer_id']),
            "order_date": str(pd.to_datetime(row['order_date'])),
            "status": "Cancelled" if is_cancelled else "Completed",
            "total_amount": float(row['total_amount']),
            "shipping_address": f"{row['country']} Retail Hub",
            "region": str(row['country'])
        })
    
    df_orders = pd.DataFrame(order_rows)
    bulk_insert(df_orders, "orders", if_exists="append")
    logger.info(f"Ingested {len(df_orders):,} genuine orders.")
    
    # Prepare Order Items
    df['order_id'] = df['invoice_str'].map(invoice_to_order_id).astype(int)
    df_items = pd.DataFrame({
        "order_id": df['order_id'],
        "product_id": df['product_id'],
        "quantity": df['quantity'].abs().astype(int),
        "unit_price": df['price'].round(2),
        "discount": 0.00,
        "line_total": (df['quantity'].abs() * df['price']).round(2)
    })
    
    bulk_insert(df_items, "order_items", if_exists="append")
    logger.info(f"Ingested {len(df_items):,} order items.")
    
    # 5. Payments
    logger.info("Generating Payments for Completed Orders...")
    completed_orders = df_orders[df_orders['status'] == 'Completed']
    pay_methods = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
    pay_rows = []
    for _, ord_row in completed_orders.iterrows():
        oid = int(ord_row['order_id'])
        pay_rows.append({
            "order_id": oid,
            "payment_method": pay_methods[oid % len(pay_methods)],
            "amount": float(ord_row['total_amount']),
            "payment_date": ord_row['order_date'],
            "status": "Completed",
            "transaction_ref": f"TXN-UK-{oid:08d}"
        })
    bulk_insert(pd.DataFrame(pay_rows), "payments", if_exists="append")
    logger.info(f"Ingested {len(pay_rows):,} payment records.")
    
    logger.info("=== Real Retail Data ETL Pipeline Finished Successfully! ===")

if __name__ == "__main__":
    load_and_ingest_real_retail()
