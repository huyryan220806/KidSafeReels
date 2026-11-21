"""Create a local inference bundle without uploading weights or raw datasets."""
import json
from pathlib import Path
import zipfile

from ml.toxic_baseline import ROOT, sha256


def main():
    source = ROOT / "models/checkpoints"
    if (source / "inference-config.json").exists():
        raise ValueError("An improved bundle is configured; use ml.package_improved instead of overwriting the active v1 manifest.")
    names = ["toxic-tfidf-v1.joblib", "violence-resnet18-v1.joblib", "resnet18-imagenet1k-v1.pth",
             "asr_provenance.json", "faster-whisper-base/config.json", "faster-whisper-base/model.bin",
             "faster-whisper-base/tokenizer.json", "faster-whisper-base/vocabulary.txt"]
    manifest = {"version": "kidsafe-cpu-v1", "files": []}
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}; prepare/train models first")
        manifest["files"].append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest["asr"] = json.loads((source / "asr_provenance.json").read_text())
    for key, file in (("toxic", "toxic_baseline_metrics.json"), ("violence", "visual_baseline_metrics.json")):
        report = json.loads((ROOT / "reports" / file).read_text(encoding="utf-8"))
        manifest[key] = {"report": f"reports/{file}", "threshold": report["threshold"]}
    content = json.dumps(manifest, ensure_ascii=False, indent=2)
    (ROOT / "models/artifacts.json").write_text(content, encoding="utf-8")
    output = source / "bundle"
    output.mkdir(exist_ok=True)
    archive = output / "kidsafe-cpu-v1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for name in names:
            zipped.write(source / name, name)
        zipped.writestr("bundle-manifest.json", content)
    checksum = sha256(archive)
    (output / "kidsafe-cpu-v1.sha256.txt").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(f"Bundle: {archive}\nSHA256: {checksum}")


if __name__ == "__main__":
    main()
