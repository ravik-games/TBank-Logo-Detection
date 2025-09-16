from typing import List, Tuple
import numpy as np
from PIL import Image

from app.schemas import BoundingBox, Detection
from app.core.config import settings
from app.models.onnx_loader import get_onnx_session


class InferenceService:
    """Сервис инференса ONNX-модели (YOLO-подобный выход)."""
    DEFAULT_INPUT_HW: Tuple[int, int] = (800, 800)  # (высота, ширина)
    PAD_COLOR: Tuple[int, int, int] = (114, 114, 114)
    NMS_EPS: float = 1e-6

    def __init__(self):
        # Ленивая загрузка при первом обращении
        self._session = None
        self._input_name = None
        self._input_hw: Tuple[int, int] | None = None  # (высота, ширина)

    @property
    def session(self):
        if self._session is None:
            self._session = get_onnx_session()
            # Кэширование имени входа и статического размера
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            shape = inp.shape  # [N,C,H,W] или динамический
            try:
                h = int(shape[2]) if isinstance(shape[2], (int,)) else None
                w = int(shape[3]) if isinstance(shape[3], (int,)) else None
                if h and w:
                    self._input_hw = (h, w)
            except Exception:
                self._input_hw = None
        return self._session

    def _get_input_hw(self) -> Tuple[int, int]:
        """Возвращает размеры входа модели (H, W), либо дефолтные при динамическом входе."""
        return self._input_hw if self._input_hw is not None else self.DEFAULT_INPUT_HW

    def _letterbox(self, img: Image.Image, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Масштабирование с сохранением пропорций и паддинг (letterbox).

        Возвращает массив CHW в диапазоне [0,1], коэффициент масштаба и сдвиги (pad_left, pad_top).
        """
        # Масштабирование с сохранением сторон (letterbox)
        ih, iw = img.height, img.width
        new_h, new_w = target_hw
        scale = min(new_w / iw, new_h / ih)
        nw, nh = int(round(iw * scale)), int(round(ih * scale))
        img_resized = img.resize((nw, nh))
        pad_w, pad_h = new_w - nw, new_h - nh
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        canvas = Image.new("RGB", (new_w, new_h), self.PAD_COLOR)
        canvas.paste(img_resized, (pad_left, pad_top))
        arr = np.asarray(canvas).astype(np.float32) / 255.0
        # Преобразование к формату NCHW
        arr = np.transpose(arr, (2, 0, 1))
        return arr, scale, (pad_left, pad_top)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        """Простая NMS: возвращает индексы боксов, оставшихся после подавления по IoU."""
        idxs = scores.argsort()[::-1]
        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            rest = idxs[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_rest - inter + self.NMS_EPS)
            idxs = rest[iou <= iou_thr]
        return keep

    def _parse_yolo_output(self, output: np.ndarray) -> np.ndarray:
        """Нормализация различных форм выхода YOLO ONNX к виду (N, D).

        Поддерживаемые типичные формы:
        - (1, N, D)
        - (N, D)
        - (1, D, N)
        - (D, N)
        Возвращает двумерный массив (N, D), где D >= 5.
        """
        arr = output
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2:
            n, d = arr.shape
            # Ожидается что N > D
            # Если строк меньше, чем столбцов, вероятно это (D, N)
            if n < d:
                arr = arr.T
        return arr

    def _map_boxes_to_original(
            self,
            boxes_xyxy: np.ndarray,
            pad_left: int,
            pad_top: int,
            scale: float,
            orig_w: int,
            orig_h: int,
    ) -> np.ndarray:
        """Переводит координаты из letterbox-пространства в оригинальные координаты изображения и клипует."""
        # Отменяем letterbox
        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top
        boxes_xyxy /= scale
        # Ограничение координат границами изображения
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)
        return boxes_xyxy

    def predict(self, image: Image.Image) -> List[Detection]:
        """Основной метод инференса: принимает PIL.Image и возвращает список детекций в координатах исходного изображения."""
        try:
            # Инициализация сессии при первом обращении
            sess = self.session
            # Подготовка входных данных
            input_hw = self._get_input_hw()
            arr, scale, (pad_left, pad_top) = self._letterbox(image.convert("RGB"), input_hw)
            inp = np.expand_dims(arr, 0)  # формат NCHW
            # Запуск модели
            input_feed = {self._input_name: inp}
            outputs = sess.run(None, input_feed)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"ONNX inference failed: {e}") from e

        detections: List[Detection] = []
        if not outputs:
            return detections

        out = outputs[0]
        out = self._parse_yolo_output(out)
        if out.size == 0:
            return detections

        # Формирование списка боксов
        boxes_xyxy = []
        scores = []
        for row in out:
            if row.shape[0] < 5:
                continue
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            # Два возможных формата:
            # [cx, cy, w, h, obj, cls1, ...]
            # [cx, cy, w, h, cls1, cls2, ...] (без objness)
            score_a = 0.0
            if row.shape[0] >= 6:
                obj = float(row[4])
                cls_scores_a = row[5:]
                cls_conf_a = float(cls_scores_a.max()) if cls_scores_a.size > 0 else obj
                score_a = obj * cls_conf_a
            cls_scores_b = row[4:]
            score_b = float(cls_scores_b.max()) if cls_scores_b.size > 0 else 0.0
            score = max(score_a, score_b)
            if score < settings.CONF_THRESH:
                continue
            # Координаты xyxy в пространстве изображения после letterbox
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(score)
        if not boxes_xyxy:
            return detections
        boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)

        # Перенос координат обратно
        boxes_xyxy = self._map_boxes_to_original(
            boxes_xyxy, pad_left=pad_left, pad_top=pad_top, scale=scale, orig_w=image.width, orig_h=image.height
        )

        # Постобработка NMS
        keep = self._nms(boxes_xyxy, scores_np, settings.IOU_THRESH)
        for i in keep:
            x_min = int(np.floor(boxes_xyxy[i, 0]).item())
            y_min = int(np.floor(boxes_xyxy[i, 1]).item())
            x_max = int(np.ceil(boxes_xyxy[i, 2]).item())
            y_max = int(np.ceil(boxes_xyxy[i, 3]).item())
            if x_max <= x_min or y_max <= y_min:
                continue
            detections.append(Detection(bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)))

        return detections
