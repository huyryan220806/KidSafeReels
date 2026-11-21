"""Keep reported labels, human judgments and private source information distinct."""
from copy import deepcopy
import json
from pathlib import Path
import re
import zipfile

import pytest

from ml import speech_intake as intake


@pytest.fixture
def manifest():
    rows = [dict(video_id=f"review-{i:03d}", source_group_id=f"source-{i//2}",
                 user_reported_clip_label=int(i >= 100), duration_seconds=20.) for i in range(200)]
    return dict(rows=rows, assignments={a: intake.reviewer_ids(rows, a) for a in ("huy", "app")})


def payload(manifest, annotator="huy"):
    return dict(manifest_sha256="digest", annotator=annotator,
                rows=[intake.blank_row(sid, annotator) for sid in manifest["assignments"][annotator]])


def test_blank_review_does_not_inherit_reported_class(manifest):
    result = intake.validate(payload(manifest), manifest, "huy", "digest")
    assert result["missing"] == 200
    assert result["completed"] == result["positive"] == result["negative"] == 0


def test_cross_review_covers_both_cohorts_without_duplicate_sources(manifest):
    chosen = [r for r in manifest["rows"] if r["video_id"] in manifest["assignments"]["app"]]
    assert len(chosen) == len({r["source_group_id"] for r in chosen}) == 40
    assert sum(r["user_reported_clip_label"] for r in chosen) == 20
    assert intake.reviewer_ids(manifest["rows"], "app") == manifest["assignments"]["app"]


def test_review_preserves_partial_and_uncertain_labels(manifest):
    p = payload(manifest)
    p["rows"][0].update(watched_full=True, language="vi", toxic_language=0)
    p["rows"][1].update(watched_full=True, language="uncertain", toxic_language="uncertain")
    p["rows"][2].update(watched_full=True, language="vi", toxic_language=1,
                         evidence=[dict(start_seconds=0, end_seconds=20, reason="Test fixture only")])
    result = intake.validate(p, manifest, "huy", "digest")
    assert (result["positive"], result["negative"], result["uncertain"], result["missing"]) == (1, 1, 1, 197)


@pytest.mark.parametrize("changes", [
    dict(watched_full=False), dict(language=None), dict(toxic_language=True),
    dict(language="uncertain"), dict(evidence=[dict(start_seconds=1, end_seconds=2, reason="Conflict")]),
    dict(transcript_verified=True), dict(notes=[]), dict(annotator="app"), dict(evidence="oops"),
])
def test_rejects_invalid_or_conflicting_review(manifest, changes):
    p = payload(manifest)
    p["rows"][0].update(watched_full=True, language="vi", toxic_language=0)
    p["rows"][0].update(changes)
    with pytest.raises(ValueError):
        intake.validate(p, manifest, "huy", "digest")


@pytest.mark.parametrize("interval", [None, {}, dict(start_seconds=-1, end_seconds=2, reason="x"),
    dict(start_seconds=1, end_seconds=21, reason="x"), dict(start_seconds=2, end_seconds=2, reason="x"),
    dict(start_seconds=True, end_seconds=2, reason="x"), dict(start_seconds=0, end_seconds=float("nan"), reason="x"),
    dict(start_seconds=0, end_seconds=float("inf"), reason="x"), dict(start_seconds=0, end_seconds=2, reason=" ")])
def test_positive_needs_valid_timestamped_evidence(manifest, interval):
    p = payload(manifest)
    p["rows"][0].update(watched_full=True, language="vi", toxic_language=1,
                         evidence=[] if interval is None else [interval])
    with pytest.raises(ValueError):
        intake.validate(p, manifest, "huy", "digest")


@pytest.mark.parametrize("rows", [None, {}, [None]*200, [{}]*200, [dict(video_id=[])]*200])
def test_malformed_import_is_a_validation_error(manifest, rows):
    p = payload(manifest)
    p["rows"] = rows
    with pytest.raises(ValueError):
        intake.validate(p, manifest, "huy", "digest")


def test_wrong_cohort_or_duplicate_ids_rejected(manifest):
    p = payload(manifest)
    with pytest.raises(ValueError):
        intake.validate(p, manifest, "huy", "other-digest")
    p["rows"][1] = deepcopy(p["rows"][0])
    with pytest.raises(ValueError):
        intake.validate(p, manifest, "huy", "digest")


@pytest.mark.parametrize("path", ["../outside.mp4", "movie.txt"])
def test_source_media_cannot_escape_pack(tmp_path, path):
    with pytest.raises(ValueError):
        intake.checked_media_path(tmp_path, path)


def test_near_duplicate_queue_requires_multiple_nonflat_hashes():
    rows = [dict(video_id=v, source_group_id=v, user_reported_clip_label=i % 2)
            for i, v in enumerate(("a", "b", "c"))]
    hashes = ["aaaaaaaaaaaaaaaa", "5555555555555555", "ffffffff00000000"]
    fps = {v: [dict(dhash=h, gray_stddev=30 if v != "c" else 0) for h in hashes] for v in ("a", "b", "c")}
    pairs = intake.near_duplicate_candidates(rows, fps)
    assert len(pairs) == 1
    assert pairs[0]["left"] == "a" and pairs[0]["right"] == "b"
    assert pairs[0]["cross_reported_label"] and pairs[0]["human_duplicate_verdict"] is None


def test_pack_hides_sources_preserves_submission_and_checks_media(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    private = root / "private"
    private.mkdir(parents=True)
    (root / "ml/templates").mkdir(parents=True)
    (root / "docs").mkdir()
    template = (intake.ROOT / "ml/templates/speech_review.html").read_text(encoding="utf-8")
    (root / "ml/templates/speech_review.html").write_text(template, encoding="utf-8")
    for name in ("SPEECH200_ANNOTATION.md", "ANNOTATION_GUIDE.md"):
        (root / "docs" / name).write_text("Fixture", encoding="utf-8")
    media = root / "secret-toxic.mp4"
    media.write_bytes(b"fixture; not real video")
    m = dict(rows=[dict(video_id="review-001", original_video_id="vi-normal-001", path=media.name,
                        sha256=intake.sha(media), duration_seconds=20, user_reported_clip_label=0)],
             assignments={a: ["review-001"] for a in ("huy", "app")})
    intake.write(private / "manifest.json", m)
    submission = private / "annotations-huy.blank.json"
    submission.write_text("existing draft", encoding="utf-8")
    (private / "annotations-app.blank.json").write_text("existing app draft", encoding="utf-8")
    monkeypatch.setattr(intake, "ROOT", root)
    monkeypatch.setattr(intake, "PRIVATE", private)
    intake.pack()
    assert submission.read_text(encoding="utf-8") == "existing draft"
    html = (private / "review-huy.html").read_text(encoding="utf-8")
    config = json.loads(re.search(r"const config=(.*);", html)[1])
    assert config["initial"]["rows"][0]["toxic_language"] is None
    assert not any(word in json.dumps(config) for word in ("secret-toxic", "vi-normal", "user_reported", "source_group"))
    with zipfile.ZipFile(private / "speech200-app-review.zip") as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("annotations-app.blank.json"))["rows"][0]["toxic_language"] is None
        assert set(archive.namelist()) == {"review-app.html", "annotations-app.blank.json", "HUONG_DAN.md",
                                           "ANNOTATION_GUIDE.md", "media/review-001.mp4"}
    assert (private / "annotations-app.blank.json").read_text(encoding="utf-8") == "existing app draft"
    media.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after audit"):
        intake.pack()
