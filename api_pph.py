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

from fastapi import FastAPI, UploadFile, File,HTTPException
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

        # Load id_to_class mapping (keys are strings in JSON)
    with open(classes_path, "r") as f:
        mapping = json.load(f)
    
    if "class_to_id" in mapping:
        id_to_class = {v: k for k, v in mapping["class_to_id"].items()}
    else:
        id_to_class = mapping.get("id_to_class", {})
    
    num_classes = len(id_to_class) 

    # 2) Apply your custom training settings that matter for inference
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"      
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes       # t3.micro -> CPU only
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5     # or whatever you like

    predictor = DefaultPredictor(cfg)



    return predictor, id_to_class



predictor, id_to_class = load_model_and_classes()
print(2)


from PIL import Image , UnidentifiedImageError
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file and return detections + a 'red_light_present' flag.
    Includes validation and error handling.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image."
        )

    try:
        # Read and decode image
        image_bytes = await file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Unable to read image. Unsupported or corrupted file."
        )

    try:
        # Convert to BGR numpy (Detectron2 format)
        image = np.array(pil_image)[:, :, ::-1]

        with torch.no_grad():
            outputs = predictor(image)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
    scores = instances.scores.tolist() if instances.has("scores") else []
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []

    predictions = []
    for box, score, cls_id in zip(boxes, scores, classes):
        label = (
            id_to_class.get(str(cls_id))
            or id_to_class.get(cls_id)
            or "unknown"
        )
        predictions.append(
            {
                "box": box,
                "score": float(score),
                "class_id": int(cls_id),
                "label": label,
            }
        )

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