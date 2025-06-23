from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    llm_api_key: str = Field(alias="LLM_API_KEY", default="super-secret")
    llm_base_url: str = Field(alias="LLM_BASE_URL", default="https://api.openai.com/v1")
    llm_model_id: str = Field(alias="LLM_MODEL_ID", default="gpt-4o-mini")

    # Logging
    lite_logging_base_url: Optional[str] = Field(alias="LITE_LOGGING_BASE_URL", default=None)
    lite_logging_channel: Optional[str] = Field(alias="LITE_LOGGING_CHANNEL", default=f'room-{os.urandom(16).hex()}')
    telegram_post_url: Optional[str] = Field(alias="TELEGRAM_POST_URL", default=None)

    # app state
    app_env: str = Field(alias="APP_ENV", default="development")

    # Server
    host: str = Field(alias="HOST", default="0.0.0.0")
    port: int = Field(alias="PORT", default=80)

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings() 