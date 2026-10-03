import pytest

from ocr import registry
from ocr.base import Detector


def test_register_and_build(isolated_registry):
    @registry.register("detector", "fake")
    class Fake(Detector):
        def __init__(self, a=1):
            self.a = a

    inst = registry.build("detector", "fake", a=5)
    assert isinstance(inst, Fake)
    assert inst.a == 5


def test_unknown_name_lists_valid_options():
    with pytest.raises(ValueError, match=r"Unknown detector 'nope'.*yolo"):
        registry.build("detector", "nope")


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown kind 'widget'"):
        registry.build("widget", "x")


def test_register_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown kind 'widget'"):
        registry.register("widget", "x")


def test_unknown_param_rejected_and_lists_accepted(isolated_registry):
    @registry.register("detector", "fake2")
    class Fake(Detector):
        def __init__(self, a=1):
            self.a = a

    with pytest.raises(ValueError, match=r"Unknown params \['b'\] for detector 'fake2'.*\['a'\]"):
        registry.build("detector", "fake2", b=2)


def test_component_with_var_kwargs_skips_param_check(isolated_registry):
    @registry.register("detector", "fake3")
    class Fake(Detector):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    assert registry.build("detector", "fake3", anything=1).kwargs == {"anything": 1}


def test_builtin_names_are_listed():
    assert registry.available("detector") == ["yolo"]
    assert registry.available("recognizer") == ["paddleocr_vl"]
    assert registry.available("spotter") == ["paddleocr_vl"]
