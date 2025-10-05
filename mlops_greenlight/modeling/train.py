# train.py
# CPU-only Detectron2 training script, import-safe for pytest.
# Works in Docker on macOS (no CUDA/MPS inside Linux containers).

import os
from datetime import datetime
from dotenv import load_dotenv

# --- Load env early; optionally force CPU (before importing torch/detectron2) ---
load_dotenv()
if os.getenv("FORCE_CPU", "").lower() in {"1", "true", "yes"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import yaml
import cv2
import pandas as pd

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from codecarbon import EmissionsTracker


# -----------------------------
# Label aliasing / parsing
# -----------------------------
LABEL_ALIAS = {
    "Red": "Red",
    "RedLeft": "Red",
    "RedRight": "Red",
    "RedStraight": "Red",
    "RedStraightLeft": "Red",
    "Green": "Green",
    "GreenLeft": "Green",
    "GreenRight": "Green",
    "GreenStraight": "Green",
    "GreenStraightLeft": "Green",
    "GreenStraightRight": "Green",
    "Yellow": "Yellow",
    "off": "off",
}

def get_classes_from_yaml(yaml_files):
    labels = set()
    for yfile in yaml_files:
        if not yfile:
            continue
        if yfile and os.path.exists(yfile):
            with open(yfile, "r") as f:
                data = yaml.safe_load(f)
            for item in data:
                for box in item.get("boxes", []):
                    aliased = LABEL_ALIAS.get(box["label"], box["label"])
                    labels.add(aliased)
    return sorted(labels)

def load_yaml_annotations(yaml_file, dataset_root, class_name_to_id):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    dataset_dicts = []
    for idx, item in enumerate(data):
        record = {}

        img_path = item["path"]
        if img_path.startswith("./"):
            img_path = img_path[2:]
        abs_path = os.path.join(dataset_root, img_path)

        record["file_name"] = abs_path
        record["image_id"] = idx

        objs = []
        for box in item.get("boxes", []):
            aliased = LABEL_ALIAS.get(box["label"], "off")
            if aliased not in class_name_to_id:
                continue
            x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            objs.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": 0,  # BoxMode.XYXY_ABS (avoid import for simplicity)
                "category_id": class_name_to_id[aliased],
                "iscrowd": 0,
            })

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


# -----------------------------
# Dataset registration
# -----------------------------
def register_dataset(name_prefix, dataset_root, yaml_train, yaml_val, class_name_to_id):
    DatasetCatalog.register(
        f"{name_prefix}_train",
        lambda: load_yaml_annotations(yaml_train, dataset_root, class_name_to_id)
    )
    MetadataCatalog.get(f"{name_prefix}_train").set(
        thing_classes=list(class_name_to_id.keys())
    )

    if yaml_val:
        DatasetCatalog.register(
            f"{name_prefix}_val",
            lambda: load_yaml_annotations(yaml_val, dataset_root, class_name_to_id)
        )
        MetadataCatalog.get(f"{name_prefix}_val").set(
            thing_classes=list(class_name_to_id.keys())
        )


# -----------------------------
# Trainer
# -----------------------------
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)


# -----------------------------
# Config builder (CPU + fast)
# -----------------------------
def build_cfg(num_classes: int, use_cpu: bool = True, fast_cpu: bool = True):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))

    # Datasets will be set by caller; keep defaults here
    # caller should set cfg.DATASETS.TRAIN / TEST if needed
    # For this project:
    cfg.DATASETS.TRAIN = ("traffic_train",)
    # Will only evaluate if a val set was registered:
    cfg.DATASETS.TEST = ("traffic_val",)

    if use_cpu:
        cfg.MODEL.DEVICE = "cpu"
        cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Fast, CPU-friendly defaults
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = int(os.getenv("MAX_ITER", "5"))
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.STEPS = []               # avoid warnings (no LR steps)
    cfg.SOLVER.CHECKPOINT_PERIOD = 10**9
    cfg.TEST.EVAL_PERIOD = 0

    if fast_cpu:
        cfg.INPUT.MIN_SIZE_TRAIN = (480,)   # smaller images => less RAM/CPU
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
        cfg.INPUT.MAX_SIZE_TRAIN = 800
        cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
        cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 500
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

    cfg.OUTPUT_DIR = "./models"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


# -----------------------------
# Tracking (DagsHub/MLflow) - import safe
# -----------------------------
USE_DAGSHUB = os.getenv("MLOPS_DISABLE_DAGSHUB", "").lower() not in {"1", "true", "yes"}

def setup_tracking():
    """
    Initialize DagsHub/MLflow only when explicitly running the script,
    never at import time (pytest-safe).
    Returns (use_mlflow: bool, mlflow_module_or_None)
    """
    if not USE_DAGSHUB:
        print("[INFO] DagsHub disabled via MLOPS_DISABLE_DAGSHUB.")
        return False, None

    owner = os.getenv("DAGSHUB_REPO_OWNER")
    name  = os.getenv("DAGSHUB_REPO_NAME")
    if not owner or not name:
        print("[WARN] DagsHub config missing; skipping DagsHub/MLflow init.")
        return False, None

    import dagshub, mlflow
    dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
    mlflow.set_experiment("GreenLight Fine-Tuning")
    return True, mlflow


# -----------------------------
# Utils
# -----------------------------
def safe_read_emissions():
    candidates = [
        os.path.join(os.getcwd(), "emissions.csv"),
        "/workspace/emissions.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Paths (adapt to your layout if needed)
    train_root = os.path.join("data", "raw", "dataset_train_rgb")
    test_root  = os.path.join("data", "raw", "dataset_test_rgb")

    yaml_train = os.path.join(train_root, "train.yaml")
    yaml_val   = os.path.join(test_root,  "test.yaml") if os.path.exists(
        os.path.join(test_root, "test.yaml")
    ) else None

    # Classes
    all_classes = get_classes_from_yaml([yaml_train, yaml_val])
    class_name_to_id = {cls: i for i, cls in enumerate(all_classes)}

    # Register datasets
    register_dataset("traffic", train_root, yaml_train, yaml_val, class_name_to_id)

    # Build config
    cfg = build_cfg(num_classes=len(class_name_to_id), use_cpu=True, fast_cpu=True)

    # Optional: further overrides from env
    # e.g. export MAX_ITER=20 to change iterations quickly

    # Start tracking (optional)
    use_ml, mlflow = setup_tracking()

    print("[INFO] Starting training...")
    tracker = EmissionsTracker()  # disabled if CODECARBON_DISABLED=1
    tracker.start()

    if use_ml:
        mlflow.start_run(run_name=f"train_frcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Train
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    tracker.stop()

    # Artifacts
    # 1) CO2 metrics (if available)
    emissions_csv = safe_read_emissions()
    if use_ml and emissions_csv:
        try:
            emissions = pd.read_csv(emissions_csv)
            # Defensive slice: align with CodeCarbon version
            # (columns typically: timestamp.. + power/energy + metadata)
            # Here we pick the last row and split metrics/params roughly:
            last = emissions.iloc[-1]
            # Example heuristic:
            metrics = {}
            params = {}
            for k, v in last.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
                else:
                    params[k] = str(v)
            if metrics:
                mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params(params)
        except Exception as e:
            print(f"[WARN] Failed to log emissions.csv: {e}")

    # 2) Persist & log class mappings
    classes_path = os.path.join(cfg.OUTPUT_DIR, "classes.json")
    id_to_class = {v: k for k, v in class_name_to_id.items()}
    with open(classes_path, "w") as f:
        json.dump({"class_to_id": class_name_to_id, "id_to_class": id_to_class}, f, indent=2)
    if use_ml:
        mlflow.log_artifact(classes_path)

    # 3) Dump config
    cfg_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(cfg.dump())
    if use_ml:
        mlflow.log_artifact(cfg_path)

    # 4) Log final weights (if present)
    weights_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if os.path.exists(weights_path) and use_ml:
        mlflow.log_artifact(weights_path)

    # 5) Log detectron2 metrics.json (last record)
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
    if os.path.exists(metrics_path) and use_ml:
        last_metrics = None
        with open(metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    last_metrics = record
                except json.JSONDecodeError:
                    pass
        if isinstance(last_metrics, dict):
            numeric = {k: v for k, v in last_metrics.items() if isinstance(v, (int, float))}
            if numeric:
                mlflow.log_metrics(numeric)
        mlflow.log_artifact(metrics_path)

    if use_ml:
        mlflow.end_run()

    print("[INFO] Training finished.")
