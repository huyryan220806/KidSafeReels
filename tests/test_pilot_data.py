"""Prevent source leakage, false gold labels and lost annotation assignments."""
from copy import deepcopy
import json

import pytest

from ml.pilot_data import MANIFEST, cache_key, load_manifest, sha, validate_manifest
from ml.pilot_annotation import annotation_rows, validate_annotations


@pytest.fixture
def manifest():
    return load_manifest()


def test_frozen_dataset_is_unlabeled_and_source_disjoint(manifest):
    assert len(manifest["clips"]) == 25
    assert len(manifest["sources"]) == 8
    assert all(v is None for clip in manifest["clips"] for v in clip["labels"].values())
    groups = [{c["group_id"] for c in manifest["clips"] if c["split"] == split}
              for split in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_rejects_related_sources_in_different_splits(manifest):
    manifest["sources"][0]["split"] = "test"
    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest(manifest)


def test_rejects_identical_source_with_new_id_across_splits(manifest):
    manifest["sources"][-1]["sha1"] = manifest["sources"][0]["sha1"]
    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest(manifest)


@pytest.mark.parametrize("kind", ["overlap", "out_of_bounds", "nan", "path", "predicted_gold"])
def test_rejects_unusable_manifest_before_extraction(manifest, kind):
    if kind == "overlap":
        manifest["clips"][1]["start_seconds"] = 10
    elif kind == "out_of_bounds":
        manifest["clips"][0]["end_seconds"] = 1000
    elif kind == "nan":
        manifest["clips"][0]["start_seconds"] = float("nan")
    elif kind == "path":
        manifest["clips"][0]["video_path"] = "../outside.mp4"
    else:
        manifest["clips"][0]["labels"]["violence"] = 0
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_manifest_hash_detects_silent_split_change(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes(MANIFEST.read_bytes())
    path.with_suffix(".sha256").write_text(sha(path), encoding="ascii")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["clips"][0]["split"] = "train"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(path)


def test_cache_changes_with_media_manifest_model_and_settings():
    original = cache_key("manifest", "media", "model", {"version": 1})
    variants = [cache_key("new", "media", "model", {"version": 1}),
                cache_key("manifest", "new", "model", {"version": 1}),
                cache_key("manifest", "media", "new", {"version": 1}),
                cache_key("manifest", "media", "model", {"version": 2})]
    assert all(key != original for key in variants)


def payload(manifest, annotator="huy"):
    return {"manifest_sha256": sha(MANIFEST), "annotator": annotator,
            "rows": annotation_rows(manifest, annotator)}


def test_blank_annotations_do_not_count_as_negatives_or_completed(manifest):
    p = payload(manifest)
    assert validate_annotations(p, manifest, "huy") == 0
    assert all(row["violence"] is None and row["toxic_language"] is None for row in p["rows"])
    ids = {row["video_id"] for row in annotation_rows(manifest, "app")}
    assert len(ids) >= len(manifest["clips"]) * 0.2
    selected = [c for c in manifest["clips"] if c["video_id"] in ids]
    assert {c["split"] for c in selected} == {"train", "validation", "test"}
    assert len({c["group_id"] for c in selected}) == 5


@pytest.mark.parametrize("kind", ["not_watched", "positive_no_evidence", "invalid_time", "bool_label", "duplicate_id"])
def test_invalid_annotations_cannot_become_gold(manifest, kind):
    p = payload(manifest)
    row = p["rows"][0]
    row.update(watched_full=True, language="vi", violence=0, toxic_language=0)
    if kind == "not_watched":
        row["watched_full"] = False
    elif kind == "positive_no_evidence":
        row["violence"] = 1
    elif kind == "invalid_time":
        row["violence"] = 1
        row["violence_evidence"] = [{"start_seconds": 24, "end_seconds": 30, "reason": "example"}]
    elif kind == "bool_label":
        row["violence"] = True
    else:
        p["rows"][1] = deepcopy(row)
    with pytest.raises(ValueError):
        validate_annotations(p, manifest, "huy")


def test_completed_annotation_count_preserves_uncertainty(manifest):
    p = payload(manifest)
    p["rows"][0].update(watched_full=True, language="uncertain", violence=0, toxic_language="uncertain")
    assert validate_annotations(p, manifest, "huy") == 1
    assert p["rows"][0]["toxic_language"] == "uncertain"


def test_frames_use_real_pts_and_source_offset(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from fractions import Fraction
    from ml.pilot_data import extract_frames
    path = tmp_path / "variable.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 10)
        for index in (0, 3, 11, 17, 23):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 10, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    records = extract_frames(path, tmp_path / "frames", 45)
    assert [r["clip_seconds"] for r in records] == pytest.approx([0, 1.1, 2.3])
    assert [r["source_seconds"] for r in records] == pytest.approx([45, 46.1, 47.3])


def test_asr_overshoot_is_bounded_with_raw_timestamp_preserved():
    from ml.pilot_data import bound_asr_interval
    result = bound_asr_interval(20.97, 25.43, 25)
    assert result["end_seconds"] == 25
    assert result["raw_end_seconds"] == 25.43
    assert result["timing_clipped"] is True
    assert bound_asr_interval(26, 28, 25) is None
    with pytest.raises(ValueError):
        bound_asr_interval(float("nan"), 2, 25)


def test_annotation_browser_script_parses(manifest, tmp_path):
    import shutil
    import subprocess
    from ml.pilot_annotation import review_html
    node = shutil.which("node")
    if not node:
        pytest.skip("Optional Node syntax check; browser QA is also documented")
    html = review_html(manifest, "huy", annotation_rows(manifest, "huy"))
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    path = tmp_path / "annotation.js"
    path.write_text(script, encoding="utf-8")
    subprocess.run([node, "--check", str(path)], check=True, capture_output=True, timeout=15)
