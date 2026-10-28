import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("cv2")

from ml.temporal_visual import sample_windows


def test_short_temporal_clip_keeps_order_and_repeats_end_without_fake_frames(tmp_path):
    path=tmp_path/'ramp.mp4'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=30)
        stream.width=64; stream.height=48; stream.pix_fmt='yuv420p'
        for i in range(15):
            frame=av.VideoFrame.from_ndarray(np.full((48,64,3),i*12,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    windows=sample_windows(path)
    assert windows.shape==(5,16,112,112,3) and windows.dtype==np.uint8
    assert np.array_equal(windows[0],windows[-1])
    luminance=windows[0].mean(axis=(1,2,3))
    assert np.all(np.diff(luminance)>=-1)
    assert luminance[0]<5 and luminance[-1]>155
    assert np.array_equal(windows[0,-1],windows[0,-2])


def test_corrupt_temporal_input_fails_instead_of_inventing_frames(tmp_path):
    from kidsafe.errors import InvalidMediaError
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(InvalidMediaError):
        sample_windows(path)


def test_temporal_runtime_uses_frozen_feature_order_and_rejects_wrong_backbone(tmp_path, monkeypatch):
    import joblib
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from ml import temporal_visual, visual
    from ml.toxic_baseline import sha256
    from kidsafe.errors import ModelNotReadyError
    feature = np.array([1., 2., 3., 4.])
    model = LogisticRegression().fit(np.array([feature*i for i in (0, .1, 1, 2)]), [0, 0, 1, 1])
    backbone = tmp_path / "r3d18-kinetics400.pth"
    backbone.write_bytes(b"test backbone bytes")
    artifact = dict(model=model, threshold=.4, version="fixture", sklearn_version=sklearn.__version__,
        sampling=temporal_visual.SAMPLING, temporal_backbone=backbone.name,
        temporal_backbone_sha256=sha256(backbone), configuration={"features": "temporal"})
    path = tmp_path / "classifier.joblib"
    joblib.dump(artifact, path)

    class Encoder:
        def __init__(self, path, device):
            assert path == backbone and device == "cpu"

        def encode(self, windows):
            assert windows == "decoded windows"
            return feature

    monkeypatch.setattr(temporal_visual, "TemporalEncoder", Encoder)
    monkeypatch.setattr(temporal_visual, "sample_windows", lambda path: "decoded windows")
    classifier = visual.ViolenceClassifier(path)
    assert classifier.is_temporal and classifier.frame_count == 80
    assert classifier.score_video("fixture.mp4") == pytest.approx(model.predict_proba(feature[None])[0, 1])
    backbone.write_bytes(b"changed backbone")
    with pytest.raises(ModelNotReadyError):
        visual.ViolenceClassifier(path)
