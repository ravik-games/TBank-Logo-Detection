"""Точка входа FastAPI-приложения для детекции логотипов."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.api.routes import router
from app.core.config import settings
from app.core.exceptions import (
    APIException,
    api_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)
from app.services.inference import InferenceService


def create_app() -> FastAPI:
    """Создание и настройка экземпляра FastAPI (CORS, роуты, обработчики ошибок)."""
    app = FastAPI(title="TBank Logo Detection API")

    # Настройка CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # Инициализация сервисов
    inference_service = InferenceService()
    app.state.inference_service = inference_service

    # Подключение роутеров
    app.include_router(router)

    # Обработчики исключений
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    @app.get("/")
    async def root() -> dict:
        """Короткая справка по сервису и доступным эндпойнтам."""
        return {"name": "TBank Logo Detection API", "version": "1.0", "endpoints": ["/health", "/detect"]}

    return app


app = create_app()