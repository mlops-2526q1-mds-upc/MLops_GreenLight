import os
from datetime import datetime
import torch
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torchvision.ops import nms
import yaml
from dotenv import load_dotenv
import dagshub
import mlflow
from codecarbon import EmissionsTracker
import pandas as pd

# ------------------------------------------------------
# Environment setup (CPU-only)
# ------------------------------------------------------
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False
print("[INFO] Forcing CPU-only mode for Detectron2 and PyTorch.")

# ------------------------------------------------------
# DagsHub / MLflow setup
# ------------------------------------------------------
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_experiment("GreenLight Inference")

# ------------------------------------------------------
# Dataset loading
# ------------------------------------------------------
test_root = os.path.join("data", "raw", "dataset_test_rgb")
test_yaml = os.path.join(test_root, "test.yaml")
test_images_dir = os.path.join(test_root, "rgb", "test")

with open(test_yaml, "r") as f:
    test_entries = yaml.safe_load(f)

test_images = {
    os.path.join(test_images_dir, os.path.basename(e["path"])): e["boxes"]
    for e in test_entries
}
print(f"[DEBUG] Loaded {len(test_images)} entries from {test_yaml}")

# ------------------------------------------------------
# Config & predictor (CPU)
# ------------------------------------------------------
state_colors = {"Red": (0, 0, 255), "Yellow": (0, 255, 255), "Green": (0, 255, 0), "off": (128, 128, 128)}
id_to_label = {0: "Red", 1: "Green", 2: "Yellow", 3: "off"}

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("models", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(id_to_label)
cfg.MODEL.DEVICE = "cpu"  # ✅ Force CPU

print("[INFO] Running inference on CPU...")

predictor = DefaultPredictor(cfg)

# ------------------------------------------------------
# Run inference and save predictions
# ------------------------------------------------------
pred_dir = os.path.join("models", "predictions")
os.makedirs(pred_dir, exist_ok=True)

run_name = f"inference_frcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.start_run(run_name=run_name)
mlflow.log_params(
    {"weights_path": cfg.MODEL.WEIGHTS, "device": cfg.MODEL.DEVICE, "score_thresh_test": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}
)

tracker = EmissionsTracker()
tracker.start()

num_images_processed = 0
for img_path, _ in test_images.items():
    if not os.path.exists(img_path):
        print(f"[WARNING] Image not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outputs = predictor(img_rgb)
    instances = outputs["instances"].to("cpu")

    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        keep = nms(boxes, scores, iou_threshold=0.3)
        instances = instances[keep]

    for i in range(len(instances)):
        box = instances.pred_boxes[i].tensor.numpy()[0]
        cls_id = int(instances.pred_classes[i])
        score = float(instances.scores[i])
        label = id_to_label.get(cls_id, "off")
        color = state_colors.get(label, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_rgb, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = os.path.join(pred_dir, f"pred_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    num_images_processed += 1
    print(f"[INFO] Saved prediction: {out_path}")

tracker.stop()
mlflow.log_metric("num_images_processed", num_images_processed)
mlflow.log_artifacts(pred_dir)
mlflow.end_run()
