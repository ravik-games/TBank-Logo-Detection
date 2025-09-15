from fastapi import APIRouter, UploadFile

from app.core.exceptions import APIException
from app.schemas import DetectionResponse
from app.utils.image_io import read_image_from_upload

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile):
    image, _fmt, width, height = read_image_from_upload(file)

    # Инференс
    try:
        detections = router.inference_service.predict(image)  # type: ignore[attr-defined]
    except Exception as e:
        raise APIException(status_code=500, error="Model Error", detail=str(e))

    return DetectionResponse(detections=detections)
