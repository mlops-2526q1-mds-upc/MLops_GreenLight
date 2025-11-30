import os
import json
from io import BytesIO

import numpy as np
import torch

try:
    from PIL import Image as _Image
    if not hasattr(_Image, "LINEAR") and hasattr(_Image, "BILINEAR"):
        _Image.LINEAR = _Image.BILINEAR
except Exception:
    pass

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


print(1)


app = FastAPI(title="Red Light Detection API")


def load_model_and_classes():
    """
    Load Detectron2 config + trained weights + class mapping
    WITHOUT reading the auto-generated config.yaml.
    """
    models_dir = "models"

    weights_path = os.path.join(models_dir, "model_final.pth")
    classes_path = os.path.join(models_dir, "classes.json")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file not found: {classes_path}")



    cfg = get_cfg()

    # 1) Start from the same base config used for training
    #    (in your YAML: COCO-Detection/faster_rcnn_R_50_FPN_3x) :contentReference[oaicite:4]{index=4}
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    # 2) Apply your custom training settings that matter for inference
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"            # t3.micro -> CPU only
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5     # or whatever you like

    predictor = DefaultPredictor(cfg)

    # Load id_to_class mapping (keys are strings in JSON)
    with open(classes_path, "r") as f:
        mapping = json.load(f)
    id_to_class = mapping.get("id_to_class", {})

    return predictor, id_to_class


print(2)
predictor, id_to_class = load_model_and_classes()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file and return detections + a 'red_light_present' flag.
    """
    # Read image bytes
    image_bytes = await file.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Detectron2 expects BGR numpy array
    image = np.array(pil_image)[:, :, ::-1]

    with torch.no_grad():
        outputs = predictor(image)

    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
    scores = instances.scores.tolist() if instances.has("scores") else []
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []

    predictions = []
    for box, score, cls_id in zip(boxes, scores, classes):
        # JSON keys are strings -> try both int and string
        label = (
            id_to_class.get(str(cls_id))
            or id_to_class.get(cls_id)
            or "unknown"
        )
        predictions.append(
            {
                "box": box,                  # [x1, y1, x2, y2]
                "score": float(score),       # confidence
                "class_id": int(cls_id),
                "label": label,
            }
        )

    # Simple rule: red_light_present if any 'Red' is detected above threshold
    red_light_present = any(
        p["label"] == "Red" and p["score"] >= 0.5 for p in predictions
    )

    return JSONResponse(
        {
            "red_light_present": red_light_present,
            "num_detections": len(predictions),
            "predictions": predictions,
        }
    )

print(3)