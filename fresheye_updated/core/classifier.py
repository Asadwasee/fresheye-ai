"""
core/classifier.py
===================
Wraps the user's trained Ultralytics YOLO CLASSIFICATION model
(models/best.pt) and exposes the prediction surfaces used by the Flask
routes:

  * predict_image(path)   -> single image classification
  * predict_pil(image)    -> classify an already-loaded PIL image (used
                              internally, and reusable for future input
                              sources without touching disk)

The model is loaded once (singleton) and reused across requests, which is
the standard production pattern for serving a torch/ultralytics model
inside a Flask worker - this matters even more once you're classifying a
batch of images back-to-back in one request.
"""
import threading

import numpy as np
from PIL import Image

from config import Config


class ClassificationResult:
    """Plain, JSON-serialisable prediction result."""

    def __init__(self, label: str, confidence: float, all_probs: dict, is_fresh: bool | None):
        self.label = label
        self.confidence = confidence
        self.all_probs = all_probs
        self.is_fresh = is_fresh

    def to_dict(self):
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "all_probs": {k: round(v, 4) for k, v in self.all_probs.items()},
            "is_fresh": self.is_fresh,
        }


def classify_label_freshness(label: str) -> bool | None:
    """Heuristically decide whether a predicted class name means
    'fresh' or 'rotten' based on Config.FRESH_KEYWORDS / ROTTEN_KEYWORDS.
    Returns True (fresh), False (rotten), or None (unknown / can't tell).
    """
    lower = label.lower()
    if any(k in lower for k in Config.ROTTEN_KEYWORDS):
        return False
    if any(k in lower for k in Config.FRESH_KEYWORDS):
        return True
    return None


def is_non_fruit_label(label: str) -> bool:
    """True if the predicted class represents 'this isn't a fruit at all'
    (e.g. a negative/catch-all class like 'Non Fruits') rather than a
    fresh/rotten fruit reading. Used to skip freshness advice generation
    and show an honest 'can't classify this' response instead."""
    normalized = label.lower().replace("_", " ").replace("-", " ")
    return any(k in normalized for k in Config.NON_FRUIT_KEYWORDS)


class FruitFreshnessClassifier:
    """Thread-safe singleton wrapper around the Ultralytics YOLO model."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, weights_path: str | None = None):
        if self._initialized:
            return
        from ultralytics import YOLO  # imported lazily so app boot doesn't need torch until first use

        self.weights_path = weights_path or Config.YOLO_WEIGHTS_PATH
        self.model = YOLO(self.weights_path)
        self.class_names: dict[int, str] = self.model.names
        self._initialized = True
        print(f"[classifier] Loaded YOLO classification model from "
              f"{self.weights_path} with classes: {self.class_names}")

    # ------------------------------------------------------------------ #
    # Core inference helpers
    # ------------------------------------------------------------------ #
    def _result_from_yolo_output(self, result) -> ClassificationResult:
        probs = result.probs  # ultralytics Probs object for classification models
        top1_idx = int(probs.top1)
        confidence = float(probs.top1conf)
        label = self.class_names[top1_idx]
        all_probs = {
            self.class_names[i]: float(probs.data[i]) for i in range(len(self.class_names))
        }
        is_fresh = classify_label_freshness(label)
        return ClassificationResult(label, confidence, all_probs, is_fresh)

    def predict_image(self, image_path: str) -> ClassificationResult:
        results = self.model.predict(source=image_path, verbose=False)
        return self._result_from_yolo_output(results[0])

    def predict_pil(self, pil_image: Image.Image) -> ClassificationResult:
        results = self.model.predict(source=np.array(pil_image.convert("RGB")), verbose=False)
        return self._result_from_yolo_output(results[0])


def get_classifier() -> FruitFreshnessClassifier:
    """Convenience accessor - triggers lazy singleton load."""
    return FruitFreshnessClassifier()
