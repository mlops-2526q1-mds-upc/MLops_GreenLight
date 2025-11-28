import pytest
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

pytestmark = pytest.mark.integration


def test_inference_cpu_predictor_builds():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    DefaultPredictor(cfg)
