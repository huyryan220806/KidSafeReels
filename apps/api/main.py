import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi import FastAPI, HTTPException, UploadFile

from kidsafe.pipeline import ModelNotReadyError, analyze_video
from kidsafe.schemas import AnalysisResult
from kidsafe.errors import InvalidMediaError

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}

app = FastAPI(title="KidSafeReels", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "backend": os.getenv("KIDSAFE_BACKEND", "mock")}


@app.post("/v1/analyze", response_model=AnalysisResult)
def analyze(file: UploadFile):
    started = perf_counter()
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, "Chỉ nhận file .mp4, .mov hoặc .webm.")
        # Do not use the uploaded filename as a storage path.
        with TemporaryDirectory(prefix="kidsafe-") as directory:
            path = Path(directory) / f"video{suffix}"
            total = 0
            with path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Video vượt giới hạn 50 MiB.")
                    destination.write(chunk)
            if total == 0:
                raise HTTPException(400, "File video rỗng.")
            try:
                result = analyze_video(path, backend=os.getenv("KIDSAFE_BACKEND", "mock"))
            except ModelNotReadyError as exc:
                raise HTTPException(503, str(exc)) from exc
            except InvalidMediaError as exc:
                raise HTTPException(422, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(503, "Cấu hình pipeline không hợp lệ.") from exc
            result.processing_seconds = round(perf_counter() - started, 4)
            return result
    finally:
        file.file.close()
