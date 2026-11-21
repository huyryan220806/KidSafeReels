import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, UploadFile

from apps.api.worker import AnalysisError, AnalyzerWorker
from kidsafe.schemas import AnalysisResult

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}


@asynccontextmanager
async def lifespan(app):
    app.state.analyzer = AnalyzerWorker()
    try:
        yield
    finally:
        app.state.analyzer.close()


app = FastAPI(title="KidSafeReels", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "backend": os.getenv("KIDSAFE_BACKEND", "mock")}


@app.post("/v1/analyze", response_model=AnalysisResult)
def analyze(request: Request, file: UploadFile):
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
                result = request.app.state.analyzer.analyze(path, backend=os.getenv("KIDSAFE_BACKEND", "mock"))
            except AnalysisError as exc:
                headers = {"Retry-After": "5"} if exc.status_code == 429 else None
                raise HTTPException(exc.status_code, str(exc), headers=headers) from exc
            result.processing_seconds = round(perf_counter() - started, 4)
            return result
    finally:
        file.file.close()
