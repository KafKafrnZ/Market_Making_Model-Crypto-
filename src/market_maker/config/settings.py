from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    binance_api_key: str
    binance_api_secret: str
    symbol: str = "BTCUSDT"

    class Config:
        env_file = ".env"

settings = Settings()