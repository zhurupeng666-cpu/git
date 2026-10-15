import os
from getpass import getpass
import psycopg
from dotenv import load_dotenv
from pwdlib import PasswordHash

# 读取数据库配置
load_dotenv()
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("缺少DATABASE_URL配置")

# 输入测试密码，不在代码中明文保存
password = getpass("请输入压测账号统一使用的密码：")
if len(password) < 12:
    raise ValueError("测试密码至少需要12位")

# 生成20个独立账号，使用Argon2保护密码
hasher = PasswordHash.recommended()
created = 0
with psycopg.connect(database_url) as conn:
    for i in range(1, 21):
        username = f"loadtest{i:03d}"
        password_hash = hasher.hash(password)
        result = conn.execute(
            """
            INSERT INTO app_users(username,password_hash,display_name,role)
            VALUES (%s,%s,%s,'employee')
            ON CONFLICT(username) DO NOTHING
            RETURNING id
            """,
            (username, password_hash, f"压测用户{i:03d}")
        ).fetchone()
        if result:
            created += 1
    print(f"成功创建：{created} 个压测账号")
    print(f"已存在并跳过：{20-created} 个账号")