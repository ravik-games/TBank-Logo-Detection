from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Optional
import logging

from app.schemas import ErrorResponse

logger = logging.getLogger("uvicorn.error")


class APIException(HTTPException):
    """Класс исключений API для унифицированной обработки ошибок"""

    def __init__(self, status_code: int, error: str, detail: Optional[str] = None):
        payload = ErrorResponse(error=error, detail=detail).model_dump()
        super().__init__(status_code=status_code, detail=payload)
        self.error = error
        self.detail = detail


async def api_exception_handler(request, exc: APIException):
    if 400 <= exc.status_code < 500:
        logger.warning(f"{exc.status_code} {exc.error}: {exc.detail}")
    else:
        logger.error(f"{exc.status_code} {exc.error}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.error, detail=exc.detail).model_dump(),
    )


async def validation_exception_handler(request, exc: RequestValidationError):
    logger.warning(f"422 Validation Error: {exc}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="Validation Error", detail=str(exc)).model_dump(),
    )


async def general_exception_handler(request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error", detail="An unexpected error occurred"
        ).model_dump(),
    )
