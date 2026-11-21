"""Versioned local inference choices; older CPU bundles keep their original behavior."""
import json
from pathlib import Path

from kidsafe.errors import ModelNotReadyError

DEFAULT = {
    "version": "kidsafe-cpu-v1",
    "toxic": "toxic-tfidf-v1.joblib",
    "violence": "violence-resnet18-v1.joblib",
    "asr_directory": "faster-whisper-base",
    "asr_version": "whisper-base",
    "asr_options": {"beam_size": 1},
}


def load_config(directory):
    directory = Path(directory).resolve()
    path = directory / "inference-config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(DEFAULT)
        if set(config) != set(DEFAULT):
            raise ValueError("Unexpected inference config keys")
        for key in ("toxic", "violence", "asr_directory"):
            value = config[key]
            if not isinstance(value, str) or not value or Path(value).is_absolute() or not (directory / value).resolve().is_relative_to(directory):
                raise ValueError("Artifact must stay inside the local bundle")
        options = config["asr_options"]
        if not isinstance(options, dict) or set(options) - {"beam_size", "temperature"}:
            raise ValueError("Unsupported ASR options")
        if options.get("beam_size") not in (1, 5) or ("temperature" in options and options["temperature"] != 0):
            raise ValueError("Unsupported ASR decoding configuration")
        for key in ("version", "asr_version"):
            if not isinstance(config[key], str) or not config[key]:
                raise ValueError("Missing version")
        return config
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ModelNotReadyError("Cấu hình model local không hợp lệ; kiểm tra bundle/checksum.") from exc
