---
license: apache-2.0
tags:
  - object-detection
  - faster-rcnn
  - ResNet-50
  - FPN
  - COCO
library_name: detectron2
paper_name: “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”
paper_arxiv_id: “1506.01497”
base_model_architecture: Faster R-CNN (ResNet-50 + FPN)
datasets:
  - name: COCO 2017
    type: object detection (bounding boxes)
    size: 80 classes, ~118k train images / 5k validation images / 20k test images
metrics:
  # These are published baseline numbers from Detectron2 Model Zoo for this model
  # See Detectron2 “COCO Object Detection Baselines” table.
  box_AP (IoU=0.50-0.95): ~ 40.2  
  box_AP50 (IoU=0.50): ~ 61.2-62 (approx)  
  box_AP75 (IoU=0.75): ~ 43-45  
  # small / medium / large object APs:
  box_AP_S: (small objects) ~ 22-25  
  box_AP_M: (medium objects) ~ 43-45  
  box_AP_L: (large objects) ~ 51-55  
commits: 137849458 # model id in Detectron2 model zoo for R50-FPN 3x
inference_time: ~ 0.209 s / image (on GPU, batch size = 1) :contentReference[oaicite:2]{index=2}
model_size: ~ 170-180 MB (checkpoint size) # approximate

---

## Model description

This model is **Faster R-CNN with a ResNet-50 backbone + Feature Pyramid Network**, trained on the COCO 2017 dataset using the “3× schedule” (i.e. approximately 3× the number of epochs/iterations relative to the “1× schedule”).  

It is designed for general object detection over 80 classes.  

---

## How to use

You can use this model via Detectron2’s model zoo:

```python
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
```

