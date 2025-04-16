import os
import json
import sqlite3
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

# 管理员用户名
ADMIN_USERNAME = "admin"


def create_user_db():
    """创建用户数据库"""
    conn = sqlite3.connect('user_database.db')
    c = conn.cursor()

    # 创建用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  is_admin INTEGER DEFAULT 0,
                  created_at TEXT,
                  last_login TEXT)''')

    # 检查并创建管理员账户
    c.execute("SELECT * FROM users WHERE username=?", (ADMIN_USERNAME,))
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES (?, ?, 1, ?, NULL)",
                  (ADMIN_USERNAME, "admin123", datetime.now().isoformat()))

    conn.commit()
    conn.close()


def check_user(username: str, password: str) -> bool:
    """检查用户是否存在且密码正确"""
    conn = sqlite3.connect('user_database.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    return result is not None and result[0] == password


def is_admin(username: str) -> bool:
    """检查用户是否是管理员"""
    conn = sqlite3.connect('user_database.db')
    c = conn.cursor()
    c.execute("SELECT is_admin FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    return result is not None and result[0] == 1


def add_user(username: str, password: str) -> bool:
    """添加新用户"""
    try:
        conn = sqlite3.connect('user_database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                  (username, password, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_user(username: str) -> bool:
    """删除用户"""
    if username == ADMIN_USERNAME:
        return False  # 不能删除管理员

    conn = sqlite3.connect('user_database.db')
    c = conn.cursor()
    try:
        c.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
        return c.rowcount > 0
    finally:
        conn.close()


def list_users() -> List[Dict[str, Any]]:
    """获取所有用户列表"""
    conn = sqlite3.connect('user_database.db')
    c = conn.cursor()
    c.execute("SELECT username, is_admin, created_at, last_login FROM users")
    users = []
    for row in c.fetchall():
        users.append({
            "username": row[0],
            "is_admin": bool(row[1]),
            "created_at": row[2],
            "last_login": row[3]
        })
    conn.close()
    return users


def update_last_login(username: str):
    """更新用户最后登录时间"""
    conn = sqlite3.connect('user_database.db')
    c = conn.cursor()
    c.execute("UPDATE users SET last_login=? WHERE username=?",
              (datetime.now().isoformat(), username))
    conn.commit()
    conn.close()


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


def clear_feature_database(db_path="feature_db.sqlite"):
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM features;")  # 清空特征表
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