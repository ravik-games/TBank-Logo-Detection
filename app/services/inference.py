from typing import List
import numpy as np
from PIL import Image

from app.schemas import BoundingBox, Detection
from app.core.config import settings
from app.models.yolo_loader import get_yolo_model


class InferenceService:
    def __init__(self):
        # Ленивая загрузка при первом обращении
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_yolo_model()
        return self._model

    def predict(self, image: Image.Image) -> List[Detection]:
        # Выполняем предсказание
        try:
            results = self.model.predict(
                source=image,
                conf=settings.CONF_THRESH,
                iou=settings.IOU_THRESH,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            # Оборачиваем в RuntimeError, выше перехватится в APIException
            raise RuntimeError(f"YOLO inference failed: {e}") from e

        detections: List[Detection] = []
        if not results:
            return detections

        # Берём первый результат (одно изображение на запрос)
        res = results[0]
        # В ultralytics 8/11 у res.boxes.xyxy — тензор Nx4, res.boxes.conf — Nx1
        try:
            xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, 'cpu') else np.array(res.boxes.xyxy)
        except Exception:
            # Совместимость на случай иной структуры
            xyxy = np.array(getattr(res.boxes, 'xyxy', []))

        h, w = getattr(res.orig_img, 'shape', (None, None))[:2]
        if h is None or w is None:
            w, h = image.size

        for box in xyxy:
            x_min = int(max(0, np.floor(box[0]).item()))
            y_min = int(max(0, np.floor(box[1]).item()))
            x_max = int(min(w - 1, np.ceil(box[2]).item()))
            y_max = int(min(h - 1, np.ceil(box[3]).item()))
            if x_max < x_min or y_max < y_min:
                continue
            detections.append(Detection(bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)))

        return detections
