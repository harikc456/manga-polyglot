import argparse
import io
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel


class ReviewBody(BaseModel):
    status: str  # "approved" | "flagged" | "unseen"
    notes: str


def create_app(input_dir: Path, temp_dir: Path, output_dir: Path) -> FastAPI:
    app = FastAPI()

    review_log_path = temp_dir / "review_log.json"
    review_log: dict = json.loads(review_log_path.read_text()) if review_log_path.exists() else {}

    def _save_review_log():
        review_log_path.write_text(json.dumps(review_log, indent=2))

    def _read_cache(name: str) -> dict:
        cache_path = temp_dir / f"{name}.ocr.json"
        if not cache_path.exists():
            raise HTTPException(status_code=404, detail=f"Cache not found for {name}")
        return json.loads(cache_path.read_text())

    def _page_names() -> list[str]:
        return sorted(p.name.removesuffix(".ocr.json") for p in temp_dir.glob("*.ocr.json"))

    @app.get("/api/pages")
    def list_pages():
        return [
            {
                "name": name,
                "status": review_log.get(name, {}).get("status", "unseen"),
            }
            for name in _page_names()
        ]

    frontend_dir = Path(__file__).parent / "review_frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


def main():
    parser = argparse.ArgumentParser(description="Manga translation review UI")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--temp-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    app = create_app(args.input_dir, args.temp_dir, args.output_dir)
    webbrowser.open(f"http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
