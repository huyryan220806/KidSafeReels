"""Offline, CPU-only inference for the frozen Vietnamese text checkpoint."""
import json
from pathlib import Path

from ml.toxic_baseline import normalize


class PhoBertClassifier:
    def __init__(self, config_path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        config_path = Path(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config.get("segmentation") != "pyvi-0.1.1"
                or config.get("normalization") != "NFC-casefold-whitespace-v1"
                or config.get("max_length") != 192
                or not 0 < config["threshold"] < 1):
            raise ValueError("Unsupported PhoBERT preprocessing configuration")
        self.threshold = config["threshold"]
        self.version = config["version"]
        self.max_length = config["max_length"]
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(config_path.parent, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config_path.parent, local_files_only=True, use_safetensors=True).to("cpu").eval()

    def predict(self, text):
        import torch
        from pyvi import ViTokenizer
        if not isinstance(text, str) or not normalize(text):
            raise ValueError("Text must be a non-empty string")
        segmented = ViTokenizer.tokenize(normalize(text))
        inputs = self.tokenizer(segmented, max_length=self.max_length, truncation=True,
                                padding=True, return_tensors="pt")
        with torch.inference_mode():
            score = float(torch.softmax(self.model(**inputs).logits.float(), dim=-1)[0, 1])
        return {"label": "toxic_language", "score": score, "flagged": score >= self.threshold,
                "threshold": self.threshold, "model_version": self.version, "scope": "text_only"}


def load_text_classifier(path):
    if Path(path).name == "kidsafe-config.json":
        return PhoBertClassifier(path)
    from ml.toxic_baseline import ToxicClassifier
    return ToxicClassifier(path)
