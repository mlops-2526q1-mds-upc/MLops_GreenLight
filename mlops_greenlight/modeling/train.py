from datetime import datetime
import json
import os
import sys
import torch
import yaml
import dagshub
import mlflow
import pandas as pd
from dotenv import load_dotenv
from codecarbon import EmissionsTracker
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer
from prometheus_client import Gauge, start_http_server

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
# Prometheus metrics (training-only; exported when run as a script)
# =====================================================
TRAIN_DURATION_SECONDS = Gauge(
    "mlops_train_duration_seconds",
    "Wall-clock duration of the last training run in seconds.",
)
TRAIN_EMISSIONS_KG = Gauge(
    "mlops_train_emissions_kg",
    "CO2 emissions in kilograms for the last training run (CodeCarbon).",
)
TRAIN_LAST_ITERATION = Gauge(
    "mlops_train_last_iteration",
    "Last training iteration number reported by Detectron2 metrics.json.",
)
TRAIN_LAST_TOTAL_LOSS = Gauge(
    "mlops_train_last_total_loss",
    "Last total_loss value reported by Detectron2 metrics.json.",
)
TRAIN_STATUS = Gauge(
    "mlops_train_status",
    "Status of the most recent training run (1=success, 0=failure).",
)

# =====================================================
# 1. DagsHub / MLflow setup
#    (can be disabled for local/CI runs with MLOPS_DISABLE_DAGSHUB=1)
# =====================================================
disable_dagshub = os.getenv("MLOPS_DISABLE_DAGSHUB", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

if disable_dagshub:
    # Local file-based MLflow tracking, no network calls
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("GreenLight Fine-Tuning")
else:
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")

if not repo_owner or not repo_name:
    print("ERROR: DagsHub configuration missing!")
    print("Please create a .env file with:")
    print("DAGSHUB_REPO_OWNER=your_username")
    print("DAGSHUB_REPO_NAME=your_repo_name")
    sys.exit(1)

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_experiment("GreenLight Fine-Tuning")

# ======================
# TEMPORARY PLACEHOLDER
# Inline copies of helpers from features.py while package imports are stabilized.
# ======================
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
        if yfile and os.path.exists(yfile):
            with open(yfile, "r") as f:
                data = yaml.safe_load(f)
            for item in data:
                for box in item.get("boxes", []):
                    aliased_label = LABEL_ALIAS.get(box["label"], box["label"])
                    labels.add(aliased_label)
    return sorted(labels)


def load_yaml_annotations(yaml_file, dataset_root, class_name_to_id):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    dataset_dicts = []
    for idx, item in enumerate(data):
        record = {}
        img_path = item["path"].lstrip("./")
        abs_path = os.path.join(dataset_root, img_path)
        record["file_name"] = abs_path
        record["image_id"] = idx

        objs = []
        for box in item.get("boxes", []):
            aliased_label = LABEL_ALIAS.get(box["label"], "off")
            if aliased_label not in class_name_to_id:
                continue
            obj = {
                "bbox": [box["x_min"], box["y_min"], box["x_max"], box["y_max"]],
                "bbox_mode": 0,
                "category_id": class_name_to_id[aliased_label],
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# ======================
# 3. Register dataset
# ======================
def register_dataset(name_prefix, dataset_root, yaml_train, yaml_val, class_name_to_id):
    DatasetCatalog.register(
        f"{name_prefix}_train",
        lambda: load_yaml_annotations(yaml_train, dataset_root, class_name_to_id),
    )
    MetadataCatalog.get(f"{name_prefix}_train").set(
        thing_classes=list(class_name_to_id.keys())
    )
    if yaml_val:
        DatasetCatalog.register(
            f"{name_prefix}_val",
            lambda: load_yaml_annotations(yaml_val, dataset_root, class_name_to_id),
        )
        MetadataCatalog.get(f"{name_prefix}_val").set(
            thing_classes=list(class_name_to_id.keys())
        )


# ======================
# 4. Trainer
# ======================
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)


# ======================
# 5. Main
# ======================
if __name__ == "__main__":
    # Start Prometheus metrics HTTP server (for live scraping by Prometheus/Grafana)
    metrics_port = int(os.getenv("TRAIN_METRICS_PORT", "8001"))
    start_http_server(metrics_port)
    print(f"[INFO] Prometheus metrics server started on port {metrics_port}")

    # New dataset paths
    train_root = os.path.join("data", "raw", "dataset_train_rgb")
    test_root = os.path.join("data", "raw", "dataset_test_rgb")

    yaml_train = os.path.join(train_root, "train.yaml")
    yaml_val = (
        os.path.join(test_root, "test.yaml")
        if os.path.exists(os.path.join(test_root, "test.yaml"))
        else None
    )

    # Collect all aliased labels from both train + val
    all_classes = get_classes_from_yaml([yaml_train, yaml_val])
    class_name_to_id = {cls: i for i, cls in enumerate(all_classes)}

    # Register datasets
    register_dataset("traffic", train_root, yaml_train, yaml_val, class_name_to_id)

    # Config
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("traffic_train",)
    cfg.DATASETS.TEST = ("traffic_val",) if yaml_val else ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500  # increase later
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_name_to_id)
    cfg.OUTPUT_DIR = "./models"
    cfg.MODEL.DEVICE = device_str  # ✅ Dynamic CPU/GPU setting

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Starting training on {device_str.upper()}...")
    mlflow.start_run(run_name=f"train_frcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    mlflow.log_param("device_mode", device_str)

    tracker = EmissionsTracker()
    tracker.start()

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    tracker.stop()

    # Log the CO2 emissions to MLflow
    emissions = pd.read_csv("emissions.csv")
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()
    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

    # Update Prometheus metrics related to emissions / duration
    duration = float(emissions_metrics.get("duration", 0.0))
    emissions_kg = float(emissions_metrics.get("emissions", 0.0))
    TRAIN_DURATION_SECONDS.set(duration)
    TRAIN_EMISSIONS_KG.set(emissions_kg)

    # Persist and log class mappings
    classes_path = os.path.join(cfg.OUTPUT_DIR, "classes.json")
    id_to_class = {v: k for k, v in class_name_to_id.items()}
    with open(classes_path, "w") as f:
        json.dump(
            {"class_to_id": class_name_to_id, "id_to_class": id_to_class}, f, indent=2
        )
    mlflow.log_artifact(classes_path)

    # Dump Detectron2 config and log to MLflow
    cfg_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(cfg.dump())
    mlflow.log_artifact(cfg_path)

    # Log trained weights if present
    weights_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if os.path.exists(weights_path):
        mlflow.log_artifact(weights_path)

    # Log training metrics from Detectron2's metrics.json (last record)
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
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
            numeric_metrics = {
                k: v for k, v in last_metrics.items() if isinstance(v, (int, float))
            }
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)
                # Update Prometheus metrics with a few high-signal values
                if "iteration" in numeric_metrics:
                    TRAIN_LAST_ITERATION.set(float(numeric_metrics["iteration"]))
                if "total_loss" in numeric_metrics:
                    TRAIN_LAST_TOTAL_LOSS.set(float(numeric_metrics["total_loss"]))
        mlflow.log_artifact(metrics_path)

    # Mark the run as successful in Prometheus
    TRAIN_STATUS.set(1.0)

    mlflow.end_run()
