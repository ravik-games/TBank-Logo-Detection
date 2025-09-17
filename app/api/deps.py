"""Зависимости FastAPI для доступа к сервисам приложения."""
from fastapi import Request

from app.services.inference import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    """Возвращает singleton InferenceService, хранящийся в состоянии приложения."""
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        # Ленивая инициализация если приложение поднято без create_app()
        service = InferenceService()
        request.app.state.inference_service = service
    return service
