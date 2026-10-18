"""Download RLVS version 1 to ignored local storage; never redistribute media."""
import hashlib
import json
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw/rlvs"
URL = "https://www.kaggle.com/api/v1/datasets/download/mohamedmustafa/real-life-violence-situations-dataset?datasetVersionNumber=1"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "rlvs-v1.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with requests.get(URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            size = 0
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
                    size += len(chunk)
                    if size // (100 * 1024 * 1024) != (size - len(chunk)) // (100 * 1024 * 1024):
                        print(f"Downloaded {size / 1024**2:.0f} MiB", flush=True)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Endpoint did not return a ZIP; check Kaggle access manually")
        temporary.replace(archive)
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (DEST / member.filename).resolve()
            if not target.is_relative_to(DEST.resolve()):
                raise ValueError("Unsafe archive member")
        zipped.extractall(DEST)
    info = {"url": URL, "version": 1, "archive_sha256": checksum,
            "license_metadata": "Data files © Original Authors",
            "citation": "Soliman et al., Violence Recognition from Videos using Deep Learning Techniques, ICICIS 2019",
            "video_count": len(list(DEST.rglob("*.mp4")))}
    (DEST / "provenance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
