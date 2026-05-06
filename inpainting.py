from pathlib import Path
import urllib.request
from tqdm import tqdm

MODEL_URL = "https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx"
MODEL_PATH = Path.home() / ".manga-polyglot" / "models" / "lama-manga.onnx"


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tqdm(unit="B", unit_scale=True, miniters=1, desc="Downloading lama-manga.onnx") as t:
            def _reporthook(count, block_size, total_size):
                if t.total is None and total_size > 0:
                    t.total = total_size
                t.update(block_size)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _reporthook)
    return MODEL_PATH
