"""Explicitly download and verify the pinned public PhoBERT training checkpoint."""
from ml.toxic_baseline import ROOT, sha256
from ml.experiment_metrics import write_json

REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
FILES = {
    "README.md": "2cb34fea651ea41128c58e25ea3ba9d22f839f7ffd7a1f5d5c09d1b6956182ae",
    "bpe.codes": "9dfe53fb8abbddd3387da99641c36e00bee757a3679aad9cf0a7cdc4b30f6ef8",
    "config.json": "71df79a67a65b1ee472938d0dff77297c0d77e20fef1af41e0013b1d384efb0d",
    "pytorch_model.bin": "a0b0f0912c710147fbaac015b0a4011216a0061a56c03b840b639e40d3bb49cc",
    "vocab.txt": "dffd7543c401fb6567253ae7c3313004532c33d062525250049c616544bad777",
}


def prepare():
    from huggingface_hub import hf_hub_download
    target = ROOT / "models/checkpoints/pretrained/phobert-base"
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, digest in FILES.items():
        path = target / name
        if not path.is_file():
            hf_hub_download("vinai/phobert-base", name, revision=REVISION, local_dir=target)
        if sha256(path) != digest:
            raise ValueError(f"Pinned PhoBERT checksum mismatch: {name}")
        entries.append(dict(path=name, bytes=path.stat().st_size, sha256=digest))
    write_json(target / "provenance.json", dict(repo="vinai/phobert-base", revision=REVISION, license="MIT", files=entries))
    print("Verified five pinned PhoBERT files.")


if __name__ == "__main__":
    prepare()
