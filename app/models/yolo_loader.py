import threading

from app.core.config import settings

_yolo_model = None
_lock = threading.Lock()


def get_yolo_model():
    """Ленивая загрузка модели YOLO (Ultralytics)."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    with _lock:
        if _yolo_model is None:
            try:
                # Импортируем здесь, чтобы не тянуть зависимость, если инференс не используется
                from ultralytics import YOLO  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "Ultralytics is not installed. Add 'ultralytics' to requirements and reinstall."
                ) from e

            try:
                _yolo_model = YOLO(settings.MODEL_PATH)
                _yolo_model.to(settings.DEVICE)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load YOLO model from '{settings.MODEL_PATH}' on device '{settings.DEVICE}': {e}"
                ) from e
    return _yolo_model
