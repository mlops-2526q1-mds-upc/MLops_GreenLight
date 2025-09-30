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
from features import get_classes_from_yaml,load_yaml_annotations

tracker = EmissionsTracker()

tracker.start()

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
