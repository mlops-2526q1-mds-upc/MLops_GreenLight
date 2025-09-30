import os
import yaml
import cv2
import random

import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()

tracker.start()
# ======================
# 0. Label aliasing
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
    "off": "off"
}

# ======================
# 1. Collect classes dynamically from YAML
# ======================
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


# ======================
# 2. YAML Loader
# ======================
def load_yaml_annotations(yaml_file, dataset_root, class_name_to_id):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    dataset_dicts = []
    for idx, item in enumerate(data):
        record = {}

        # Fix relative paths
        img_path = item["path"]
        if img_path.startswith("./"):
            img_path = img_path[2:]

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
                "bbox_mode": 0,  # XYXY_ABS
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
        lambda: load_yaml_annotations(yaml_train, dataset_root, class_name_to_id)
    )
    MetadataCatalog.get(f"{name_prefix}_train").set(thing_classes=list(class_name_to_id.keys()))

    if yaml_val:
        DatasetCatalog.register(
            f"{name_prefix}_val",
            lambda: load_yaml_annotations(yaml_val, dataset_root, class_name_to_id)
        )
        MetadataCatalog.get(f"{name_prefix}_val").set(thing_classes=list(class_name_to_id.keys()))


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
    # New dataset paths
    train_root = os.path.join("data", "raw", "dataset_train_rgb")
    test_root = os.path.join("data", "raw", "dataset_test_rgb")

    yaml_train = os.path.join(train_root, "train.yaml")
    yaml_val = os.path.join(test_root, "test.yaml") if os.path.exists(os.path.join(test_root, "test.yaml")) else None

    # Collect all aliased labels from both train + val
    all_classes = get_classes_from_yaml([yaml_train, yaml_val])
    class_name_to_id = {cls: i for i, cls in enumerate(all_classes)}

    # Register datasets
    register_dataset("traffic", train_root, yaml_train, yaml_val, class_name_to_id)

    # Config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("traffic_train",)
    cfg.DATASETS.TEST = ("traffic_val",) if yaml_val else ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500  # increase later
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_name_to_id)
    cfg.OUTPUT_DIR = "./models"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("[INFO] Starting training...")
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


tracker.stop()
