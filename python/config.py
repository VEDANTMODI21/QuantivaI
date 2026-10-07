import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import logging

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Database Configuration
DB_DRIVER = os.getenv("DB_DRIVER", "sqlite").lower()

import shutil

SQLITE_DB_FULL_PATH = BASE_DIR / "quantivaiq.db"

def locate_sqlite_db():
    candidates = [
        BASE_DIR / "quantivaiq.db",
        Path("/var/task/quantivaiq.db"),
        Path(os.getcwd()) / "quantivaiq.db",
        Path("/tmp/quantivaiq.db")
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return BASE_DIR / "quantivaiq.db"

if DB_DRIVER == "sqlite":
    source_db = locate_sqlite_db()
    if os.getenv("VERCEL"):
        tmp_db = Path("/tmp/quantivaiq.db")
        if source_db.exists() and source_db != tmp_db:
            try:
                if not tmp_db.exists() or tmp_db.stat().st_size < source_db.stat().st_size:
                    shutil.copy2(source_db, tmp_db)
                target_db = tmp_db
            except Exception:
                target_db = source_db
        elif tmp_db.exists() and tmp_db.stat().st_size > 0:
            target_db = tmp_db
        else:
            target_db = source_db
        p = target_db.as_posix()
        DATABASE_URL = f"sqlite:///{p}"
    else:
        p = source_db.as_posix()
        DATABASE_URL = f"sqlite:///{p}"
else:
    DATABASE_URL = os.getenv("DATABASE_URL")




# Logging Configuration
def setup_logging(name="QuantivaIQ"):

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


# SQLAlchemy Engine
def get_engine():
    return create_engine(DATABASE_URL)


# Database Connection Test
def test_db_connection():

    try:
        engine = get_engine()

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return True, ""

    except OperationalError as exc:

        logger = setup_logging("DBConnectionCheck")

        logger.error(
            f"Unable to connect to the configured database. Error: {exc}"
        )

        return False, str(exc)


# App Settings
NUM_CUSTOMERS = int(os.getenv("NUM_CUSTOMERS", 500))
NUM_PRODUCTS = int(os.getenv("NUM_PRODUCTS", 50))
NUM_ORDERS = int(os.getenv("NUM_ORDERS", 5000))
FRAUD_RATE = float(
    os.getenv("FRAUD_CONTAMINATION_RATE", 0.02)
)
def is_sqlite():
    return DB_DRIVER == "sqlite"
SIMULATION_INTERVAL = int(os.getenv("SIMULATION_INTERVAL_SECONDS", 5))