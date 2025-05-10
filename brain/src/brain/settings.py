from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic for validation and .env file loading.
    """
    temperature: float = 0.9

    # OpenAI API settings
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt4.1", description="OpenAI model to use")

    # Optional additional settings
    max_tokens: int = Field(1000, description="Maximum tokens for completions")

    # Configure the settings to load from .env file
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    URL: str = "http://192.168.24.82:8081"



# Create a global instance of settings
settings = Settings()

def get_settings() -> Settings:
    """
    Returns the settings instance.
    This function is useful for dependency injection in larger applications.
    """
    return settings
