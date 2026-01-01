import pymysql 
import os 
from app.core.config import settings

# Đọc thông tin từ .env 
DB_CONFIG = {
    'host': settings.DB_HOST,
    'port': settings.DB_PORT,
    'user': settings.DB_USER,
    'password': settings.DB_PASSWORD,
    'database': settings.DB_NAME,
    'charset': settings.DB_CHARSET,
    'cursorclass': pymysql.cursors.DictCursor
}

def connectDB():
    try: 
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except pymysql.MySQLError as e:
        print(f"[DB ERROR] Cannot connect to MySQL: {e}")
        return None 
