"""Конфигурация приложения и параметры инференса."""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List, Set


class Settings(BaseSettings):
    """Настройки приложения (путь к модели, пороги, CORS и пр.)."""
    # Настройки модели и инференса
    MODEL_PATH: str = Field(default="./models/model.onnx", description="Path to ONNX model file (.onnx)")
    DEVICE: str = Field(default="cpu", description="Device for inference: cpu or cuda:0")
    CONF_THRESH: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold")
    IOU_THRESH: float = Field(default=0.5, ge=0.0, le=1.0, description="IoU threshold for NMS")

    # Валидация изображений
    MAX_IMAGE_MB: int = Field(default=100, ge=1, description="Max image size in megabytes")
    ALLOWED_FORMATS: Set[str] = Field(default_factory=lambda: {"JPEG", "PNG", "BMP", "WEBP"})

    # Параметры CORS / HTTP
    CORS_ALLOW_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default_factory=lambda: ["*"])

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()