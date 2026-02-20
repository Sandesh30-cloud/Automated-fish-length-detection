import os
from functools import lru_cache

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "fish_presence_model.h5")


@lru_cache(maxsize=1)
def load_fish_model(model_path: str = DEFAULT_MODEL_PATH):
    return tf.keras.models.load_model(model_path)


def predict_fish(image_path: str, model=None, threshold: float = 0.5) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    model = model if model is not None else load_fish_model()
    prob = float(model.predict(img, verbose=0)[0][0])

    # Fish -> class 0 -> lower values
    fish_present = prob < threshold
    confidence = (1 - prob) if fish_present else prob

    return {
        "fish_present": fish_present,
        "confidence": float(confidence),
        "raw_probability": prob,
    }


if __name__ == "__main__":
    sample_image = os.path.join("fish_dataset", "val", "Fish", "fish.jpg")
    try:
        result = predict_fish(sample_image)
        print(result)
    except Exception as exc:
        print(f"Inference failed: {exc}")
