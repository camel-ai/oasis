import sqlite3


def print_db_contents(db_file):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # 获取并打印所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", [table[0] for table in tables])

    for table_name in tables:
        print(f"\nTable: {table_name[0]}")
        # 打印表结构
        cursor.execute(f"PRAGMA table_info({table_name[0]})")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")

        # 打印表内容
        cursor.execute(f"SELECT * FROM {table_name[0]}")
        rows = cursor.fetchall()
        print("Contents:")
        for row in rows:
            print(" ", row)
    # 关闭连接
    conn.close()
