import sqlite3
import os

# Kết nối đến cơ sở dữ liệu
db_path = 'C:/Users/AD/my_database.db'
print(f"Connecting to database at: {os.path.abspath(db_path)}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Kiểm tra xem bảng có tồn tại hay không
try:
    cursor.execute("SELECT * FROM leave_requests")
    rows = cursor.fetchall()
    print(rows)
except sqlite3.OperationalError as e:
    print(f"Error: {e}")

conn.close()
