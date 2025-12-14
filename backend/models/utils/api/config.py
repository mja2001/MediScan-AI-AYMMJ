from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:pass@localhost/db"
    MONGO_URL: str = "mongodb://localhost:27017"
    REDIS_URL: str = "redis://localhost:6379"
    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str
    ALLOWED_ORIGINS: list = ["*"]

settings = Settings()
