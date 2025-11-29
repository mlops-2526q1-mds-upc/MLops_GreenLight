from mlops_greenlight.modeling import train as trn


def test_get_classes_from_yaml(tiny_yaml):
    y, root = tiny_yaml
    classes = trn.get_classes_from_yaml([str(y)])
    assert "Red" in classes and "Green" in classes


def test_load_yaml_annotations(tiny_yaml):
    y, root = tiny_yaml
    classes = trn.get_classes_from_yaml([str(y)])
    mapping = {c: i for i, c in enumerate(classes)}
    data = trn.load_yaml_annotations(str(y), str(root), mapping)
    assert len(data) == 2
    assert all("file_name" in d for d in data)
