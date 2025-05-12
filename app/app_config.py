from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

from starlette.config import Config

# # .env 파일 강제 로드
# load_dotenv(dotenv_path=".env")

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_path,  # 🔧 절대 경로로 변경
        env_file_encoding="utf-8",
    )

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URL: str

@lru_cache
def get_settings():
    s= Settings()
    print("✅ LOADED:", s.model_dump())  # 디버그용 출력
    return s


# 디버깅 코드
# if __name__ == "__main__":
#     settings = Settings()
#     print("POSTGRES_USER:", os.getenv("POSTGRES_USER"))
#     print("SQLALCHEMY_DATABASE_URL:", os.getenv("SQLALCHEMY_DATABASE_URL"))
#     print("Pydantic Settings:", settings.dict())
