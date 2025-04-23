import os
import json
import sqlite3
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

# 管理员用户名
ADMIN_USERNAME = "admin"



# 数据库初始化
def init_databases():
    """初始化所有数据库"""
    create_user_db()
    create_feature_db()


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


def create_feature_db():
    """创建特征数据库"""
    conn = sqlite3.connect('feature_database.db')
    c = conn.cursor()

    # 创建特征表，增加model_type字段
    c.execute('''
        CREATE TABLE IF NOT EXISTS features (
            username TEXT,
            model_type TEXT,
            vector BLOB,
            created_at TEXT,
            PRIMARY KEY (username, model_type)
        )
    ''')

    # 创建索引以提高查询性能
    c.execute('CREATE INDEX IF NOT EXISTS idx_model_type ON features(model_type)')

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


# 特征数据库操作
def save_feature(username: str, model_type: str, vector: np.ndarray):
    """
    保存特征向量到数据库
    :param username: 用户名
    :param model_type: 模型类型 ('resnet' 或 'swin')
    :param vector: 特征向量
    """
    conn = sqlite3.connect('feature_database.db')
    cursor = conn.cursor()

    # 插入或替换特征记录
    cursor.execute(
        "INSERT OR REPLACE INTO features (username, model_type, vector, created_at) VALUES (?, ?, ?, ?)",
        (username, model_type, vector.tobytes(), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def load_features_by_model(model_type: str) -> Dict[str, np.ndarray]:
    """
    加载指定模型类型的所有特征向量
    :param model_type: 模型类型 ('resnet' 或 'swin')
    :return: 字典 {username: feature_vector}
    """
    conn = sqlite3.connect('feature_database.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT username, vector FROM features WHERE model_type = ?",
        (model_type,)
    )

    features = {
        username: np.frombuffer(vector, dtype=np.float32)
        for username, vector in cursor.fetchall()
    }

    conn.close()
    return features


def get_user_features(username: str) -> Dict[str, np.ndarray]:
    """
    获取用户的所有特征向量（按模型类型分类）
    :param username: 用户名
    :return: 字典 {model_type: feature_vector}
    """
    conn = sqlite3.connect('feature_database.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT model_type, vector FROM features WHERE username = ?",
        (username,)
    )

    features = {
        model_type: np.frombuffer(vector, dtype=np.float32)
        for model_type, vector in cursor.fetchall()
    }

    conn.close()
    return features


def delete_user_features(username: str):
    """
    删除用户的所有特征记录
    :param username: 用户名
    """
    conn = sqlite3.connect('feature_database.db')
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM features WHERE username = ?",
        (username,)
    )

    conn.commit()
    conn.close()


def clear_feature_database():
    """清空特征数据库"""
    conn = sqlite3.connect('feature_database.db')
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='features'")
    if cursor.fetchone():
        cursor.execute("DELETE FROM features")

    conn.commit()
    conn.close()