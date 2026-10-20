import json

import pytest

from kidsafe.errors import ModelNotReadyError
from ml.inference_config import DEFAULT, load_config


def test_existing_bundle_preserves_legacy_decoding(tmp_path):
    assert load_config(tmp_path) == DEFAULT


def test_config_can_select_small_without_forcing_non_vietnamese_audio(tmp_path):
    config = dict(DEFAULT, version="kidsafe-cpu-v2", asr_directory="faster-whisper-small",
                  asr_version="whisper-small-beam5", asr_options={"beam_size": 5, "temperature": 0})
    (tmp_path / "inference-config.json").write_text(json.dumps(config))
    assert load_config(tmp_path) == config


@pytest.mark.parametrize("change", [
    {"toxic": "../outside.joblib"}, {"asr_directory": "../outside"},
    {"asr_options": {"beam_size": 5, "language": "vi"}},
    {"asr_options": {"beam_size": 200}}, {"asr_version": None},
])
def test_invalid_bundle_configuration_fails_explicitly(tmp_path, change):
    (tmp_path / "inference-config.json").write_text(json.dumps(dict(DEFAULT, **change)))
    with pytest.raises(ModelNotReadyError):
        load_config(tmp_path)


@pytest.mark.parametrize("aggregation", ["mean-max", "mean-max-std-diff"])
def test_dense_inference_uses_same_feature_aggregation_as_training(tmp_path, monkeypatch, aggregation):
    import joblib
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from ml import visual
    from ml.toxic_baseline import sha256

    embeddings = np.array([[1., 2.], [3., 5.]])
    features = visual.aggregate_features(embeddings, aggregation)
    model = LogisticRegression().fit(np.array([features * i for i in (0, .1, 1, 2)]), [0, 0, 1, 1])
    backbone = tmp_path / visual.BACKBONE.name
    backbone.write_bytes(b"fixture backbone")

    class Encoder:
        def __init__(self, path):
            assert path == backbone

        def encode_frames(self, frames):
            return embeddings

    monkeypatch.setattr(visual, "VisualEncoder", Encoder)
    path = tmp_path / "classifier.joblib"
    joblib.dump(dict(model=model, threshold=.5, version="test", sklearn_version=sklearn.__version__,
                backbone_sha256=sha256(backbone), sampling=f"uniform24-{aggregation}-v2",
                frame_count=24, aggregation=aggregation), path)
    classifier = visual.ViolenceClassifier(path)
    assert classifier.frame_count == 24
    assert classifier.score([None] * 24) == pytest.approx(model.predict_proba(features[None])[0, 1])


def test_restoring_old_bundle_removes_unverified_new_inference_config(tmp_path):
    import hashlib
    import zipfile
    from ml.release_verify import install_bundle, verify_artifacts
    payload = b"legacy fixture"
    manifest = {"version": "v1-test", "files": [{"path": "model.bin", "bytes": len(payload),
                 "sha256": hashlib.sha256(payload).hexdigest()}]}
    archive = tmp_path / "old.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("model.bin", payload)
        zipped.writestr("bundle-manifest.json", json.dumps(manifest))
    target = tmp_path / "models"
    target.mkdir()
    (target / "inference-config.json").write_text(json.dumps(dict(DEFAULT, version="v2")))
    with pytest.raises(ValueError, match="unverified inference-config"):
        verify_artifacts(target, manifest)
    assert install_bundle(archive, target, manifest)["verified_files"] == 1
    assert load_config(target) == DEFAULT
