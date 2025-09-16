"""Загрузка и кэширование ONNX Runtime с выбором провайдеров (CPU/CUDA)."""
import threading
from typing import List, Optional

import onnxruntime as ort

from app.core.config import settings

_session: Optional[ort.InferenceSession] = None
_lock = threading.Lock()


def _select_providers(device: str) -> List[str]:
    """Возвращает список execution providers в зависимости от устройства (cpu/cuda)."""
    d = device.lower()
    if d.startswith("cuda"):
        # Для CUDAExecutionProvider требуется пакет onnxruntime-gpu
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # По умолчанию — CPU
    return ["CPUExecutionProvider"]


def get_onnx_session() -> ort.InferenceSession:
    """Ленивая загрузка onnxruntime.InferenceSession для модели из settings.MODEL_PATH.

    Поддерживает CPU и CUDA в зависимости от доступных провайдеров.
    """
    global _session
    if _session is not None:
        return _session

    with _lock:
        if _session is None:
            try:
                providers = _select_providers(settings.DEVICE)
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                _session = ort.InferenceSession(settings.MODEL_PATH, sess_options=sess_options, providers=providers)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load ONNX model from '{settings.MODEL_PATH}' with providers {providers}: {e}"
                ) from e
    return _session
