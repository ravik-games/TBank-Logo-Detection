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
    app = FastAPI(title="TBank Logo Detection API")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # Services
    inference_service = InferenceService()
    setattr(router, "inference_service", inference_service)

    # Routers
    app.include_router(router)

    # Exceptions
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    @app.get("/")
    async def root() -> dict:
        return {"name": "TBank Logo Detection API", "version": "1.0", "endpoints": ["/health", "/detect"]}

    return app


app = create_app()