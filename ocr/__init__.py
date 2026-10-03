from ocr import registry
from ocr.grouping import make_grouper
from ocr.pipelines import DetectRecognize, Spot

PIPELINES = ("detect_recognize", "spot")
_BLOCKS = {
    "detect_recognize": {"detector", "recognizer"},
    "spot": {"spotter"},
}


def _parse_component(kind: str, block):
    """Split a config block into (name, params); raises ValueError for malformed blocks."""
    if not isinstance(block, dict):
        raise ValueError(f"ocr.{kind} must be an object with a 'name'")
    params = dict(block)
    name = params.pop("name", None)
    if name is None:
        raise ValueError(f"ocr.{kind} needs a 'name'. Valid options: {registry.available(kind)}")
    return name, params


def build_pipeline(ocr_config):
    """Build an OCR pipeline from the config's "ocr" block. All validation happens before any model loads."""
    if not isinstance(ocr_config, dict):
        raise ValueError("config.json must contain an 'ocr' block (see README, 'OCR pipelines')")
    config = dict(ocr_config)

    pipeline = config.pop("pipeline", None)
    if pipeline not in PIPELINES:
        raise ValueError(f"Invalid ocr.pipeline '{pipeline}'. Valid options: {list(PIPELINES)}")

    grouping_config = config.pop("grouping", None)
    required = _BLOCKS[pipeline]
    unexpected = set(config) - required
    if unexpected:
        raise ValueError(
            f"Unexpected ocr blocks {sorted(unexpected)} for pipeline '{pipeline}'. "
            f"Valid blocks: {sorted(required | {'grouping'})}"
        )
    missing = required - set(config)
    if missing:
        raise ValueError(f"ocr.pipeline '{pipeline}' requires blocks: {sorted(missing)}")

    grouper = make_grouper(grouping_config)

    if pipeline == "spot":
        name, params = _parse_component("spotter", config["spotter"])
        registry.validate("spotter", name, **params)
        return Spot(registry.build("spotter", name, **params), grouper)

    det_name, det_params = _parse_component("detector", config["detector"])
    rec_name, rec_params = _parse_component("recognizer", config["recognizer"])
    crop_padding = rec_params.pop("crop_padding", 10)
    registry.validate("detector", det_name, **det_params)
    registry.validate("recognizer", rec_name, **rec_params)

    detector = registry.build("detector", det_name, **det_params)
    try:
        recognizer = registry.build("recognizer", rec_name, **rec_params)
    except BaseException:
        detector.close()
        raise
    return DetectRecognize(detector, recognizer, grouper, crop_padding=crop_padding)
