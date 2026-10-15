"""
文件作用：
    项目配置管理。

核心功能：
    1. 加载环境变量
    2. 管理模型配置
    3. 管理数据库配置
"""

import os
from dotenv import load_dotenv


load_dotenv()


class Settings:

    def __init__(self):

        self.model_name = os.getenv(
            "MODEL_NAME",
            "deepseek-v4-pro"
        )

        self.base_url = os.getenv(
            "BASE_URL",
            "https://api.deepseek.com/v1"
        )

        self.api_key = os.getenv(
            "DEEPSEEK_API_KEY"
        )

        self.temperature = float(
            os.getenv(
                "TEMPERATURE",
                "0.7"
            )
        )


        self.postgres_url = os.getenv(
            "DATABASE_URL"
        )


settings = Settings()