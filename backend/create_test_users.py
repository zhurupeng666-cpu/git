import os
from getpass import getpass
import psycopg
from dotenv import load_dotenv
from pwdlib import PasswordHash

# 加载数据库配置并初始化密码哈希工具
load_dotenv()
password_hash = PasswordHash.recommended()

# 初始化两个测试账户：普通员工和工程师
users = [
    ("employee001", "普通员工测试", "employee"),
    ("engineer001", "工程师测试", "engineer")
]

# 将账户写入PostgreSQL，密码仅保存Argon2哈希
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    for username, display_name, role in users:
        password = getpass(f"请设置 {username} 的登录密码（至少12位）：")
        if len(password) < 12:
            raise ValueError("密码长度不能少于12位")
        result = conn.execute(
            "INSERT INTO app_users(username,password_hash,display_name,role) VALUES (%s,%s,%s,%s) ON CONFLICT(username) DO NOTHING RETURNING username",
            (username, password_hash.hash(password), display_name, role)
        ).fetchone()
        print(f"{username}：{'创建成功' if result else '账户已存在，未覆盖'}")