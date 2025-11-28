from detectron2.data import DatasetCatalog, MetadataCatalog
from mlops_greenlight.modeling import train as trn


def test_dataset_registration(tiny_yaml):
    y, root = tiny_yaml
    classes = trn.get_classes_from_yaml([str(y)])
    mapping = {c: i for i, c in enumerate(classes)}
    trn.register_dataset("traffic_unit", str(root), str(y), None, mapping)
    dataset = DatasetCatalog.get("traffic_unit_train")
    assert len(dataset) == 2
    meta = MetadataCatalog.get("traffic_unit_train")
    assert list(meta.thing_classes) == list(mapping.keys())
