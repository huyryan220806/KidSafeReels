"""Protect annotation independence, frozen splits and portable handoff integrity."""
import json
from pathlib import Path
import re
import zipfile

import pytest

from ml import toxic_app_review as review
from ml.speech_intake import read, sha, write


def cohort():
    rows = [dict(video_id=f"review-{i:03d}", training_group=f"group-{i}",
        source_group_id=f"source-{i}", split="valid" if i < 2 else "train",
        training_label=i % 2) for i in range(10)]
    rows.append(dict(rows[2], video_id="test-negative", split="test", training_group="test", source_group_id="test"))
    predictions = {r["video_id"]: dict(flagged=not bool(r["training_label"]), score=.5) for r in rows}
    return dict(speech=rows), predictions


def test_selection_keeps_overlap_and_excludes_test_and_recovered_cases():
    manifest, predictions = cohort()
    predictions["review-002"]["flagged"] = False  # Already fixed, not a remaining disagreement.
    selected = review.select_rows(manifest, ["review-000", "review-001"], predictions, per_label=2)
    ids = {r["video_id"] for r in selected}
    assert len(ids) == 6
    assert {"review-000", "review-001"} <= ids
    assert not {"test-negative", "review-002"} & ids
    assert all(r["split"] == "train" for r in selected if r["video_id"] not in ("review-000", "review-001"))
    assert selected == review.select_rows(manifest, ["review-000", "review-001"], predictions, per_label=2)


def test_selection_rejects_test_overlap_duplicate_ids_and_insufficient_sources():
    manifest, predictions = cohort()
    for overlap in (["test-negative"], ["review-000", "review-000"]):
        with pytest.raises(ValueError):
            review.select_rows(manifest, overlap, predictions, per_label=2)
    for row in manifest["speech"]:
        row["source_group_id"] = "same-original-video"
    with pytest.raises(ValueError, match="Not enough"):
        review.select_rows(manifest, ["review-000"], predictions, per_label=2)


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "repo"
    (root / "ml/templates").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "ml/templates/speech_review.html").write_text(
        (review.ROOT / "ml/templates/speech_review.html").read_text(encoding="utf-8"), encoding="utf-8")
    (root / "docs/APP_TOXIC_ANNOTATION.md").write_text("Fixture instructions", encoding="utf-8")
    (root / "secret-normal.mp4").write_bytes(b"Synthetic test bytes, not a real video")
    row = dict(video_id="review-001", path="secret-normal.mp4", sha256=sha(root / "secret-normal.mp4"),
               duration_seconds=20, training_label=0, annotation={"toxic_language": 1},
               source_group_id="private-source", corrected_transcript="PRIVATE TRANSCRIPT")
    destination = root / "private/app30"
    archive = review.build_pack(root, destination, [row], "test-dataset")
    return root, destination, archive, row


def test_pack_is_portable_blind_and_checksum_complete(pack):
    root, destination, archive, _ = pack
    html = (destination / "review-app.html").read_text(encoding="utf-8")
    config = json.loads(re.search(r"const config=(.*);", html)[1])
    assert config["initial"]["rows"][0]["toxic_language"] is None
    assert config["initial"]["manifest_sha256"] == sha(destination / "review-manifest.json")
    assert review.check_submission(destination, destination / "annotations-app.blank.json")["missing"] == 1
    for forbidden in ("private-source", "secret-normal", "training_label", "PRIVATE TRANSCRIPT"):
        assert forbidden not in html
        assert forbidden not in (destination / "review-manifest.json").read_text(encoding="utf-8")
    checksums = read(destination / "CHECKSUMS.json")
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert set(zipped.namelist()) == set(checksums) | {"CHECKSUMS.json"}
        for name, digest in checksums.items():
            assert sha(destination / name) == digest
        # The reviewer can extract somewhere else and check without private source manifests.
        relocated = root / "different-machine"
        zipped.extractall(relocated)
    assert review.check_submission(relocated, relocated / "annotations-app.blank.json")["completed"] == 0


def test_pack_will_not_overwrite_existing_work_or_accept_changed_media(pack):
    root, destination, _, row = pack
    (destination / "annotations-app.blank.json").write_text("existing user work", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        review.build_pack(root, destination, [row], "test-dataset")
    assert (destination / "annotations-app.blank.json").read_text(encoding="utf-8") == "existing user work"
    (root / row["path"]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Source media changed"):
        review.build_pack(root, root / "another-pack", [row], "test-dataset2")


def test_submission_requires_matching_manifest_and_retains_uncertainty(pack):
    _, destination, _, _ = pack
    submission = destination / "draft.json"
    payload = read(destination / "annotations-app.blank.json")
    write(submission, payload)
    with pytest.raises(ValueError, match="missing"):
        review.check_submission(destination, submission, require_complete=True)
    payload["rows"][0].update(watched_full=True, language="uncertain", toxic_language="uncertain")
    write(submission, payload)
    with pytest.raises(ValueError, match="explain uncertainty"):
        review.check_submission(destination, submission)
    payload["rows"][0]["notes"] = "Cannot hear the masked phrase; synthetic fixture."
    write(submission, payload)
    status = review.check_submission(destination, submission, require_complete=True)
    assert status["uncertain"] == 1 and status["binary_labels"] == 0
    payload["manifest_sha256"] = "other pack"
    write(submission, payload)
    with pytest.raises(ValueError, match="Wrong manifest"):
        review.check_submission(destination, submission)
