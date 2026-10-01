from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from PIL import Image


def _severity(conf: float) -> str:
    if conf >= 0.90:
        return "high"
    if conf >= 0.75:
        return "medium"
    return "low"


def predict(
    image_path: Optional[str],
    model_path: str | Path,
    class_names_path: str | Path,
    image_array: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(Path(class_names_path).read_text(encoding="utf-8"))

    if image_array is None:
        if image_path is None:
            raise ValueError("Provide image_path or image_array")
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        image_array = np.array(img)

    x = tf.convert_to_tensor(image_array, dtype=tf.float32)
    x = tf.image.resize(x, (224, 224))
    x = tf.expand_dims(x, 0)

    prob = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(prob))
    conf = float(prob[idx])
    disease = class_names[idx]

    should_consult_expert = conf < 0.65
    return {
        "disease": disease,
        "confidence": conf,
        "severity": _severity(conf),
        "should_consult_expert": bool(should_consult_expert),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--model", type=str, default=str(Path(__file__).parent / "model" / "disease_model.h5"))
    ap.add_argument("--classes", type=str, default=str(Path(__file__).parent / "model" / "class_names.json"))
    args = ap.parse_args()

    out = predict(args.image, args.model, args.classes)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

