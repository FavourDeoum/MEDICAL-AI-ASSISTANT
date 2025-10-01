import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""

    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "DUMMY_TAVILY_KEY")

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    temperature: float = float(os.getenv("TEMPERATURE", 0.7))
    max_tokens: int = int(os.getenv("MAX_TOKENS", 2048))

    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB

    @property
    def cors_origins_list(self):
        """Convert comma-separated CORS origins to list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()
print(f"google  api key is: {settings.google_api_key} and port is {settings.port}")


@lru_cache()
def load_google_llm():
    """
    Load Google Gemini LLM with LangChain
    Cached to avoid recreating on every request
    """
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_tokens,
        convert_system_message_to_human=True, 
    )


@lru_cache()
def load_google_vision_llm():
    """
    Load Google Gemini with vision capabilities
    """
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.5,  
        max_output_tokens=settings.max_tokens,
        convert_system_message_to_human=True,
    )
