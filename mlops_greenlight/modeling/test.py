import os

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torchvision.ops import nms
import yaml

# ----------------------
# Paths for test data
# ----------------------
test_root = os.path.join("data", "raw", "dataset_test_rgb")
test_yaml = os.path.join(test_root, "test.yaml")
test_images_dir = os.path.join(test_root, "rgb", "test")

# Load test.yaml
with open(test_yaml, "r") as f:
    test_entries = yaml.safe_load(f)

# Map YAML paths to local images
test_images = {}
for entry in test_entries:
    fname = os.path.basename(entry["path"])  # e.g. "24070.png"
    local_path = os.path.join(test_images_dir, fname)
    test_images[local_path] = entry["boxes"]

print(f"[DEBUG] Loaded {len(test_images)} entries from {test_yaml}")

# ----------------------
# Traffic light state colors
# ----------------------
state_colors = {
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "off": (128, 128, 128),
}

# Class ID to label mapping (must match training)
id_to_label = {
    0: "Red",
    1: "Green",
    2: "Yellow",
    3: "off",
}  # match training class order

# ----------------------
# Config & predictor
# ----------------------
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)

# Update with trained model path
cfg.MODEL.WEIGHTS = os.path.join("models", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(id_to_label)

predictor = DefaultPredictor(cfg)

# ----------------------
# Run inference & save predictions
# ----------------------
pred_dir = os.path.join("models", "predictions")
os.makedirs(pred_dir, exist_ok=True)

for idx, (img_path, gt_boxes) in enumerate(test_images.items()):
    if not os.path.exists(img_path):
        print(f"[WARNING] Image not found locally: {img_path}")
        continue

    print(f"[DEBUG] Processing {img_path}")
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ensure RGB

    outputs = predictor(img_rgb)
    instances = outputs["instances"].to("cpu")

    # Optional: NMS to remove duplicate overlapping predictions
    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        keep = nms(boxes, scores, iou_threshold=0.3)
        instances = instances[keep]

    # Draw predictions with color-coded boxes
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

    # Save RGB prediction image
    out_path = os.path.join(pred_dir, f"pred_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Saved predictions to {out_path}")
