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

    def _serve_image(path: Path) -> Response:
        if not path.exists():
            raise HTTPException(status_code=404, detail=str(path.name))
        suffix = path.suffix.lower()
        media = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        return Response(content=path.read_bytes(), media_type=media)

    @app.get("/image/original/{name}")
    def image_original(name: str):
        return _serve_image(input_dir / name)

    @app.get("/image/cleaned/{name}")
    def image_cleaned(name: str):
        return _serve_image(temp_dir / name)

    @app.get("/image/output/{name}")
    def image_output(name: str):
        return _serve_image(output_dir / name)

    @app.get("/image/thumbnail/{name}")
    def image_thumbnail(name: str):
        return _serve_image(input_dir / name)

    @app.get("/image/detection/{name}")
    def image_detection(name: str):
        orig_path = input_dir / name
        if not orig_path.exists():
            raise HTTPException(status_code=404, detail=name)
        cache = _read_cache(name)
        img = Image.open(orig_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in cache.get("boxes", []):
            color = "red" if box.get("type") == "fixed" else "orange"
            x1, y1, x2, y2 = box["original_text_box"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            conf = box.get("confidence", 0)
            draw.text((x1, max(0, y1 - 12)), f"{conf:.2f}", fill=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.get("/api/page/{name}")
    def page_detail(name: str):
        cache = _read_cache(name)
        return {
            "name": name,
            "texts": cache.get("texts", []),
            "text_boxes": cache.get("text_boxes", []),
            "boxes": cache.get("boxes", []),
            "translations": cache.get("translations", []),
            "translated": cache.get("translated", False),
            "review": review_log.get(name, {"status": "unseen", "notes": "", "timestamp": None}),
        }

    @app.post("/api/review/{name}")
    def save_review(name: str, body: ReviewBody):
        review_log[name] = {
            "status": body.status,
            "notes": body.notes,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        _save_review_log()
        return {"ok": True}

    @app.get("/api/export")
    def export_log():
        return JSONResponse(content=review_log)

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
