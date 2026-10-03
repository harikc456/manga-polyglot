import importlib
import inspect

KINDS = ("detector", "recognizer", "spotter")

_REGISTRY: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}

# Built-in components: modules are imported lazily on first build, so a config
# that doesn't use a component never imports its (possibly heavy) dependencies.
_BUILTIN_MODULES = {
    ("detector", "yolo"): "ocr.detectors.yolo",
    ("recognizer", "paddleocr_vl"): "ocr.recognizers.paddleocr_vl",
    ("spotter", "paddleocr_vl"): "ocr.spotters.paddleocr_vl",
}


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Valid kinds: {list(KINDS)}")


def register(kind: str, name: str):
    _check_kind(kind)

    def decorator(cls):
        _REGISTRY[kind][name] = cls
        return cls

    return decorator


def available(kind: str) -> list[str]:
    _check_kind(kind)
    names = set(_REGISTRY[kind]) | {n for (k, n) in _BUILTIN_MODULES if k == kind}
    return sorted(names)


def _check_params(kind: str, name: str, cls: type, params: dict) -> None:
    signature = inspect.signature(cls.__init__)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return  # the component validates its own params
    accepted = {
        n
        for n, p in signature.parameters.items()
        if n != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(
            f"Unknown params {sorted(unknown)} for {kind} '{name}'. Valid: {sorted(accepted)}"
        )


def _resolve(kind: str, name: str) -> type:
    _check_kind(kind)
    if name not in _REGISTRY[kind]:
        module = _BUILTIN_MODULES.get((kind, name))
        if module is None:
            raise ValueError(f"Unknown {kind} '{name}'. Valid options: {available(kind)}")
        importlib.import_module(module)
        if name not in _REGISTRY[kind]:
            raise ValueError(f"Module '{module}' did not register {kind} '{name}'")
    return _REGISTRY[kind][name]


def validate(kind: str, name: str, **params) -> None:
    """Check name and params exactly as build() would, without instantiating."""
    _check_params(kind, name, _resolve(kind, name), params)


def build(kind: str, name: str, **params):
    cls = _resolve(kind, name)
    _check_params(kind, name, cls, params)
    return cls(**params)
