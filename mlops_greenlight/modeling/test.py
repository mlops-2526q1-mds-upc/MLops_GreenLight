import os
import sys
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

# =====================================================
# 0. Environment setup (CPU/GPU selection from .env)
# =====================================================
load_dotenv()

device_mode = os.getenv("DEVICE_MODE")
if device_mode is None:
    print(
        "ERROR: 'DEVICE_MODE' not found in .env file. Please add one of the following lines:"
    )
    print("DEVICE_MODE=cpu")
    print("or")
    print("DEVICE_MODE=gpu")
    sys.exit(1)

device_mode = device_mode.strip().lower()
if device_mode not in {"cpu", "gpu"}:
    print(f"ERROR: Invalid DEVICE_MODE='{device_mode}'. Must be 'cpu' or 'gpu'.")
    sys.exit(1)

# Force correct mode
if device_mode == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.is_available = lambda: False
    device_str = "cpu"
    print("[INFO] Forcing CPU mode for Detectron2 and PyTorch.")
else:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device_str.upper()}")

# =====================================================
# 1. DagsHub / MLflow setup
# =====================================================
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")

if not repo_owner or not repo_name:
    print("ERROR: DagsHub configuration missing!")
    print("Please create a .env file with the following variables:")
    print("DAGSHUB_REPO_OWNER=your_username")
    print("DAGSHUB_REPO_NAME=your_repo_name")
    sys.exit(1)

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_experiment("GreenLight Inference")

# =====================================================
# 2. Dataset loading
# =====================================================
test_root = os.path.join("data", "raw", "dataset_test_rgb")
test_yaml = os.path.join(test_root, "test.yaml")
test_images_dir = os.path.join(test_root, "rgb", "test")

if not os.path.exists(test_yaml):
    print(f"ERROR: test.yaml not found at {test_yaml}")
    sys.exit(1)

with open(test_yaml, "r") as f:
    test_entries = yaml.safe_load(f)

test_images = {
    os.path.join(test_images_dir, os.path.basename(e["path"])): e["boxes"]
    for e in test_entries
}

print(f"[DEBUG] Loaded {len(test_images)} entries from {test_yaml}")

# =====================================================
# 3. Color mapping and label setup
# =====================================================
state_colors = {
    "Red": (255, 0, 0),
    "Yellow": (255, 0, 255),
    "Green": (0, 255, 0),
    "off": (128, 128, 128),
}

id_to_label = {
    0: "Green",
    1: "Red",
    2: "Yellow",
    3: "off",
}  # must match training

# =====================================================
# 4. Config & predictor
# =====================================================
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = os.path.join("models", "model_final.pth")

# Check model file
if not os.path.exists(cfg.MODEL.WEIGHTS):
    print(f"ERROR: Trained model not found at {cfg.MODEL.WEIGHTS}")
    sys.exit(1)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(id_to_label)
cfg.MODEL.DEVICE = device_str  # Dynamic CPU/GPU setting

print(f"[INFO] Running inference on {cfg.MODEL.DEVICE.upper()}...")
predictor = DefaultPredictor(cfg)

# =====================================================
# 5. Run inference & save predictions
# =====================================================
pred_dir = os.path.join("models", "predictions")
os.makedirs(pred_dir, exist_ok=True)

run_name = f"inference_frcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.start_run(run_name=run_name)
mlflow.log_param("device_mode", device_str)

mlflow.log_params(
    {
        "weights_path": cfg.MODEL.WEIGHTS,
        "score_thresh_test": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
    }
)

tracker = EmissionsTracker()
tracker.start()

num_images_processed = 0
for img_path, _ in test_images.items():
    if not os.path.exists(img_path):
        print(f"[WARNING] Image not found locally: {img_path}")
        continue

    print(f"[DEBUG] Processing {img_path}")
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    outputs = predictor(img_rgb)
    instances = outputs["instances"].to("cpu")

    # Apply NMS (optional)
    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        keep = nms(boxes, scores, iou_threshold=0.3)
        instances = instances[keep]

    # Draw predictions
    for i in range(len(instances)):
        box = instances.pred_boxes[i].tensor.numpy()[0]
        cls_id = int(instances.pred_classes[i])
        score = float(instances.scores[i])
        label = id_to_label.get(cls_id, "off")
        color = state_colors.get(label, (255, 255, 255))

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_rgb,
            f"{label} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        print(f"[DEBUG] Predicted: {label} at [{x1},{y1},{x2},{y2}] score={score:.2f}")

    # Save prediction
    out_path = os.path.join(pred_dir, f"pred_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    num_images_processed += 1
    print(f"[DEBUG] Saved predictions to {out_path}")

tracker.stop()

# =====================================================
# 6. Log results to MLflow
# =====================================================
if os.path.exists("emissions.csv"):
    emissions = pd.read_csv("emissions.csv")
    last = emissions.iloc[-1]
    emissions_metrics = last.iloc[4:13].to_dict()
    emissions_params = last.iloc[13:].to_dict()
    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

mlflow.log_metric("num_images_processed", num_images_processed)
mlflow.log_artifacts(pred_dir)
mlflow.end_run()
