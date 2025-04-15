import sqlite3
import numpy as np
import os
def save_feature(name, vector, db_path="feature_db.sqlite"):
    """保存特征向量到SQLite数据库，同名自动覆盖"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表（如果不存在）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            name TEXT PRIMARY KEY,
            vector BLOB
        )
    ''')

    # 插入或替换（自动处理同名覆盖）
    cursor.execute(
        "INSERT OR REPLACE INTO features (name, vector) VALUES (?, ?)",
        (name, np.array(vector).tobytes())  # 将向量转为二进制存储
    )
    conn.commit()
    conn.close()
def load_feature(db_path="feature_db.sqlite"):
    """从SQLite数据库加载所有特征，返回{name: vector}字典"""
    if not os.path.exists(db_path):
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, vector FROM features")
    rows = cursor.fetchall()
    conn.close()

    # 将二进制数据还原为NumPy数组
    return {name: np.frombuffer(vector, dtype=np.float32) for name, vector in rows}

def clear_feature_database(db_path="feature_db.sqlite"):
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM features;")  # 清空特征表
        conn.commit()
        conn.close()