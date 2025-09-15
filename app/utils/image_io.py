import io
from typing import Tuple
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.exceptions import APIException


def read_image_from_upload(file) -> Tuple[Image.Image, str, int, int]:
    """
    Чтение и валидация изображения из UploadFile FastAPI.
    Возвращает (PIL.Image, image_format, width, height)
    """
    if not file:
        raise APIException(status_code=400, error="Empty Body", detail="No file uploaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise APIException(
            status_code=400,
            error="Invalid Content Type",
            detail=f"Expected image/*, got {file.content_type or 'unknown'}",
        )

    max_bytes = settings.MAX_IMAGE_MB * 1024 * 1024
    buf = io.BytesIO()
    size = 0
    while True:
        chunk = file.file.read(8192) if hasattr(file, 'file') else None
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise APIException(
                status_code=413,
                error="Payload Too Large",
                detail=f"Image exceeds {settings.MAX_IMAGE_MB} MB limit",
            )
        buf.write(chunk)

    if buf.tell() == 0:
        raise APIException(status_code=400, error="Empty Body", detail="Uploaded file is empty")

    try:
        # verify
        tmp = io.BytesIO(buf.getvalue())
        with Image.open(tmp) as im_verify:
            im_verify.verify()

        # open for work
        img_io = io.BytesIO(buf.getvalue())
        image = Image.open(img_io)
        image_format = (image.format or "").upper()
        if image_format not in settings.ALLOWED_FORMATS:
            raise APIException(
                status_code=400,
                error="Invalid Image Format",
                detail=f"Unsupported format: {image_format or 'unknown'}. Supported: {', '.join(sorted(settings.ALLOWED_FORMATS))}",
            )
        width, height = image.size
        return image, image_format, width, height

    except UnidentifiedImageError as e:
        raise APIException(status_code=400, error="Invalid Image", detail="Failed to identify image") from e
    except APIException:
        raise
    except Exception as e:
        raise APIException(status_code=500, error="Processing Error", detail="Failed to process image") from e
