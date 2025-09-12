from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.schemas import DetectionResponse, ErrorResponse
from typing import Optional
import io
from PIL import Image, UnidentifiedImageError
import logging

MAX_IMAGE_BYTES = 100 * 1024 * 1024  # 100 MiB
SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}

# Логирование
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="TBank Logo Detection API")


class APIException(HTTPException):
    """Класс исключений API для унифицированной обработки ошибок"""
    def __init__(self, status_code: int, error: str, detail: Optional[str] = None):
        # В HTTPException.detail положим уже готовую структуру, чтобы сохранить контекст
        payload = ErrorResponse(error=error, detail=detail).model_dump()
        super().__init__(status_code=status_code, detail=payload)
        self.error = error
        self.detail = detail


@app.exception_handler(APIException)
async def api_exception_handler(request, exc: APIException):
    if 400 <= exc.status_code < 500:
        logger.warning(f"{exc.status_code} {exc.error}: {exc.detail}")
    else:
        logger.error(f"{exc.status_code} {exc.error}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.error,
            detail=exc.detail
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    logger.warning(f"422 Validation Error: {exc}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Глобальная обработка неожиданных ошибок с унифицированным ответом.
    """
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred"
        ).model_dump()
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile):
    """
    Детекция логотипа Т-банка на изображении.
    Ожидает файл изображения через multipart/form-data.
    """
    if not file:
        raise APIException(status_code=400, error="Empty Body", detail="No file uploaded")

    # Чтение файла с ограничением размера
    buf = io.BytesIO()
    size = 0
    while content := await file.read(8192):
        size += len(content)
        if size > MAX_IMAGE_BYTES:
            raise APIException(
                status_code=413,
                error="Payload Too Large",
                detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
            )
        buf.write(content)

    if buf.tell() == 0:
        raise APIException(status_code=400, error="Empty Body", detail="Uploaded file is empty")

    # Проверка и открытие изображения через Pillow
    try:
        # Проверка целостности
        tmp = io.BytesIO(buf.getvalue())
        with Image.open(tmp) as im_verify:
            im_verify.verify()

        # Второе открытие для работы
        img_io = io.BytesIO(buf.getvalue())
        with Image.open(img_io) as image:
            image_format = (image.format or "").upper()
            if image_format not in SUPPORTED_FORMATS:
                raise APIException(
                    status_code=400,
                    error="Invalid Image Format",
                    detail=f"Unsupported format: {image_format or 'unknown'}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
                )

            # TODO: Model prediction

    except UnidentifiedImageError as e:
        raise APIException(status_code=400, error="Invalid Image", detail="Failed to identify image") from e
    except APIException:
        raise
    except Exception as e:
        logger.exception("Failed to process image")
        raise APIException(status_code=500, error="Processing Error", detail="Failed to process image") from e

    return DetectionResponse(detections=[])