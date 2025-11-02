import pymysql 
import os 
from dotenv import load_dotenv

# Load biến môi trường từ file .env 
load_dotenv()

# Đọc thông tin từ .env 
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'charset': os.getenv('DB_CHARSET'),
    'cursorclass': pymysql.cursors.DictCursor
}

def connectDB():
    return pymysql.connect(**DB_CONFIG)