"""Explicit asset preparation; inference never downloads models."""
import json

from huggingface_hub import snapshot_download

from ml.visual import CHECKPOINTS, prepare_backbone


def main():
    prepare_backbone()
    repo = "Systran/faster-whisper-base"
    revision = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
    destination = CHECKPOINTS / "faster-whisper-base"
    snapshot_download(repo, revision=revision, local_dir=destination,
                      allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.*"])
    (CHECKPOINTS / "asr_provenance.json").write_text(json.dumps({"repo": repo, "revision": revision}, indent=2))
    print(f"Prepared visual backbone and multilingual ASR in {CHECKPOINTS}")


if __name__ == "__main__":
    main()
