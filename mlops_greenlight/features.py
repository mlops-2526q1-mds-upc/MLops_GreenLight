"""Helpers to build dataset dicts from YAML annotations for Detectron2."""

import os

import yaml

# cv2 is not used directly in this module; remove unused import

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
    "off": "off",
}


# ======================
# 1. Collect classes dynamically from YAML
# ======================
def get_classes_from_yaml(yaml_files):
    """Return sorted, aliased class names found across provided YAML files."""
    labels = set()
    for yfile in yaml_files:
        if yfile and os.path.exists(yfile):
            with open(yfile, "r", encoding="utf-8") as f:
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
    """Load annotations from a YAML file and return Detectron2-style records."""
    with open(yaml_file, "r", encoding="utf-8") as f:
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
