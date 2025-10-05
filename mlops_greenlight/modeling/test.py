# test.py
# CPU-only inference script aligned with train.py config.
# Loads class mapping from models/classes.json and runs on a small subset.

import os
import json
import yaml
import cv2
import random
from pathlib import Path

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from torchvision.ops import nms

# ----------------------
# Paths
# ----------------------
TEST_ROOT = os.path.join("data", "raw", "dataset_test_rgb")
TEST_YAML = os.path.join(TEST_ROOT, "test.yaml")
TEST_IMAGES_DIR = os.path.join(TEST_ROOT, "rgb", "test")

OUTPUT_DIR = "./models"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
CLASSES_JSON = os.path.join(OUTPUT_DIR, "classes.json")
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "model_final.pth")

# ----------------------
# Colors by state (visualization only)
# ----------------------
STATE_COLORS = {
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "off": (128, 128, 128),
}

# ----------------------
# Load test.yaml
# ----------------------
if not os.path.exists(TEST_YAML):
    raise FileNotFoundError(f"Missing test.yaml at {TEST_YAML}")

with open(TEST_YAML, "r") as f:
    test_entries = yaml.safe_load(f)

# Map YAML paths to local image files
test_images = {}
for entry in test_entries:
    fname = os.path.basename(entry["path"])
    local_path = os.path.join(TEST_IMAGES_DIR, fname)
    test_images[local_path] = entry.get("boxes", [])

print(f"[DEBUG] Loaded {len(test_images)} entries from {TEST_YAML}")

# ----------------------
# Load class mapping from training
# ----------------------
if not os.path.exists(CLASSES_JSON):
    raise FileNotFoundError(
        f"Missing {CLASSES_JSON}. Run training first so it writes classes.json."
    )

with open(CLASSES_JSON, "r") as f:
    cls_data = json.load(f)

# id_to_label as produced by train.py
id_to_label = {int(k): v for k, v in cls_data["id_to_class"].items()}
num_classes = len(id_to_label)
print(f"[DEBUG] num_classes={num_classes}, id_to_label={id_to_label}")

# ----------------------
# Subset for quick pipeline checks
# ----------------------
MAX_TEST = int(os.getenv("MAX_TEST", "100"))
random.seed(int(os.getenv("TEST_SAMPLE_SEED", "42")))
items = list(test_images.items())
if len(items) > MAX_TEST:
    items = random.sample(items, MAX_TEST)
print(f"[DEBUG] Testing on {len(items)} images (sampled from {len(test_images)})")

# ----------------------
# Build config & predictor (CPU)
# ----------------------
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"Missing weights at {WEIGHTS_PATH}. Train first to produce model_final.pth."
    )

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
cfg.MODEL.WEIGHTS = WEIGHTS_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(os.getenv("SCORE_THRESH_TEST", "0.50"))

predictor = DefaultPredictor(cfg)

# ----------------------
# Inference & save predictions
# ----------------------
Path(PRED_DIR).mkdir(parents=True, exist_ok=True)

processed = 0
for idx, (img_path, gt_boxes) in enumerate(items, 1):
    if not os.path.exists(img_path):
        print(f"[WARN] Image not found: {img_path}")
        continue

    print(f"[DEBUG] ({idx}/{len(items)}) Processing {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[WARN] Failed to read image: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    outputs = predictor(img_rgb)
    instances = outputs["instances"].to("cpu")

    # Optional NMS to remove overlaps
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
        color = STATE_COLORS.get(label, (255, 255, 255))

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_rgb, f"{label} {score:.2f}", (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    out_path = os.path.join(PRED_DIR, f"pred_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    processed += 1

print(f"[INFO] Saved {processed} prediction images to {PRED_DIR}")
