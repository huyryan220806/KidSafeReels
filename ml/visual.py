"""Frozen ResNet18 features and a video-level violence classifier."""
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from torchvision.models import ResNet18_Weights, resnet18

from kidsafe.errors import ModelNotReadyError

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "models/checkpoints"
BACKBONE = CHECKPOINTS / "resnet18-imagenet1k-v1.pth"
CLASSIFIER = CHECKPOINTS / "violence-resnet18-v1.joblib"
SAMPLING_VERSION = "uniform8-mean-max-v1"


def prepare_backbone():
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if BACKBONE.exists():
        return
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), BACKBONE)


class VisualEncoder:
    def __init__(self, path=BACKBONE):
        if not path.is_file():
            raise ModelNotReadyError("Thiếu ResNet18 local. Chạy python -m ml.video_assets.")
        torch.set_num_threads(4)
        self.model = resnet18(weights=None)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def encode(self, frames):
        features = self.encode_frames(frames)
        return np.concatenate([features.mean(axis=0), features.max(axis=0)])

    def encode_frames(self, frames):
        batch = torch.stack([self.transform(frame) for frame in frames])
        with torch.inference_mode():
            features = self.model(batch).numpy()
        return features


def aggregate_features(features, mode):
    """Use identical, label-independent aggregation during fitting and inference."""
    features = np.asarray(features)
    base = [features.mean(axis=0), features.max(axis=0)]
    if mode == "mean-max-std-diff":
        delta = np.abs(np.diff(features, axis=0)).mean(axis=0) if len(features) > 1 else np.zeros(features.shape[1])
        base.extend([features.std(axis=0), delta])
    elif mode != "mean-max":
        raise ValueError(f"Unknown aggregation: {mode}")
    return np.concatenate(base)


class ViolenceClassifier:
    def __init__(self, path=CLASSIFIER):
        if not path.is_file():
            raise ModelNotReadyError("Thiếu classifier Violence. Chạy python -m ml.train_visual.")
        artifact = joblib.load(path)
        if artifact.get("temporal_backbone"):
            from ml.temporal_visual import SAMPLING, TemporalEncoder
            from ml.toxic_baseline import sha256
            mode = artifact.get("configuration", {}).get("features")
            if (artifact["sklearn_version"] != sklearn.__version__
                    or artifact["sampling"] != SAMPLING or mode not in {"temporal", "static-temporal"}
                    or artifact["temporal_backbone"] != "r3d18-kinetics400.pth"):
                raise ModelNotReadyError("Cấu hình chuyển động không tương thích.")
            backbone = path.parent / artifact["temporal_backbone"]
            if not backbone.is_file() or sha256(backbone) != artifact["temporal_backbone_sha256"]:
                raise ModelNotReadyError("Backbone chuyển động không khớp classifier.")
            torch.set_num_threads(4)
            self.temporal_encoder = TemporalEncoder(backbone, device="cpu")
            self.encoder = None
            if mode == "static-temporal":
                static = path.parent / BACKBONE.name
                if not static.is_file() or sha256(static) != artifact["backbone_sha256"]:
                    raise ModelNotReadyError("Backbone hình ảnh không khớp classifier.")
                self.encoder = VisualEncoder(static)
            self.model, self.threshold, self.version = artifact["model"], artifact["threshold"], artifact["version"]
            self.frame_count = 80 if mode == "temporal" else 88
            self.is_temporal = True
            return
        dense_modes = {f"uniform24-{mode}-v2": mode for mode in ("mean-max", "mean-max-std-diff")}
        if artifact["sklearn_version"] != sklearn.__version__ or artifact["sampling"] not in {SAMPLING_VERSION, *dense_modes}:
            raise ModelNotReadyError("Phiên bản artifact Violence không tương thích.")
        self.frame_count = 8 if artifact["sampling"] == SAMPLING_VERSION else 24
        self.aggregation = dense_modes.get(artifact["sampling"])
        if self.aggregation and (artifact.get("frame_count") != 24 or artifact.get("aggregation") != self.aggregation):
            raise ModelNotReadyError("Cấu hình lấy mẫu không khớp artifact Violence.")
        self.model = artifact["model"]
        self.threshold = artifact["threshold"]
        self.version = artifact["version"]
        backbone = path.parent / BACKBONE.name
        from ml.toxic_baseline import sha256
        if not backbone.is_file() or sha256(backbone) != artifact["backbone_sha256"]:
            raise ModelNotReadyError("Backbone không khớp classifier Violence.")
        self.encoder = VisualEncoder(backbone)

    def score(self, frames):
        feature = (aggregate_features(self.encoder.encode_frames(frames), self.aggregation)
                   if self.aggregation else self.encoder.encode(frames))
        return float(self.model.predict_proba(feature[None, :])[0, 1])

    def score_video(self, path):
        from ml.temporal_visual import sample_windows
        from ml.media import sample_frames
        feature = self.temporal_encoder.encode(sample_windows(path))
        if self.encoder is not None:
            feature = np.concatenate([self.encoder.encode(sample_frames(path)), feature])
        return float(self.model.predict_proba(feature[None, :])[0, 1])
