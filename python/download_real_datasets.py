import os
import zipfile
import urllib.request
from config import setup_logging

logger = setup_logging("Dataset_Downloader")

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

UCI_RETAIL_ZIP = os.path.join(DATASETS_DIR, "online_retail_ii.zip")
UCI_RETAIL_XLSX = os.path.join(DATASETS_DIR, "online_retail_II.xlsx")
FRAUD_CSV = os.path.join(DATASETS_DIR, "creditcard.csv")

def download_file(url, target_path, desc):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        logger.info(f"{desc} already exists at {target_path} ({os.path.getsize(target_path)} bytes). Skipping download.")
        return
    logger.info(f"Downloading {desc} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as out_file:
        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rDownloading {desc}: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\rDownloading {desc}: {downloaded / (1024*1024):.1f}MB", end="", flush=True)
    print()
    logger.info(f"Downloaded {desc} successfully to {target_path}")

def prepare_datasets():
    # 1. UCI Online Retail II
    if not os.path.exists(UCI_RETAIL_XLSX):
        download_file("https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip", UCI_RETAIL_ZIP, "UCI Online Retail II Zip")
        logger.info("Extracting Online Retail II zip...")
        with zipfile.ZipFile(UCI_RETAIL_ZIP, 'r') as z:
            z.extractall(DATASETS_DIR)
        logger.info("Extraction complete.")
    else:
        logger.info(f"UCI Online Retail II already present: {UCI_RETAIL_XLSX}")

    # 2. Kaggle European Card Fraud
    if not os.path.exists(FRAUD_CSV):
        download_file("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv", FRAUD_CSV, "Credit Card Fraud CSV")
    else:
        logger.info(f"Credit Card Fraud dataset already present: {FRAUD_CSV}")

if __name__ == "__main__":
    prepare_datasets()
