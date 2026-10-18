from copy import deepcopy
import hashlib
import json
import zipfile

import pytest

from ml.pilot_annotation import annotation_rows
from ml.pilot_data import MANIFEST, load_manifest, sha
from ml.pilot_evaluation import agreement, binary_metrics, compare, evaluate_predictions, make_gold, risk_union
from ml.release_verify import install_bundle, verify_artifacts


def test_evaluation_rejects_incomplete_gold_dirty_code_and_changed_media(tmp_path, monkeypatch):
    import ml.pilot_evaluation as module
    data, huy, app = submissions()
    gold = make_gold(data, huy, app, compare(data, huy, app))
    path = tmp_path / "gold.json"
    invalid = deepcopy(gold)
    invalid["records"][0]["labels"].pop("violence")
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="non-final labels"):
        module.run_evaluation(data, path, "test", tmp_path / "report.json")
    path.write_text(json.dumps(gold), encoding="utf-8")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: " M ml/inference.py")
    with pytest.raises(ValueError, match="Commit code"):
        module.run_evaluation(data, path, "test", tmp_path / "report.json")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(module, "sha", lambda p: "changed" if p.suffix == ".mp4" else sha(p))
    with pytest.raises(ValueError, match="media changed"):
        module.run_evaluation(data, path, "test", tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def submissions():
    data = load_manifest()
    values = []
    for name in ("huy", "app"):
        p = {"manifest_sha256": sha(MANIFEST), "annotator": name, "rows": annotation_rows(data, name)}
        for row in p["rows"]:
            row.update(watched_full=True, language="vi", violence=0, toxic_language=0)
        values.append(p)
    return data, *values


def test_blank_human_forms_cannot_trigger_evaluation():
    data, huy, app = submissions()
    huy["rows"][0]["toxic_language"] = None
    with pytest.raises(ValueError, match="incomplete"):
        compare(data, huy, app)


def test_disagreement_needs_explicit_adjudication_and_preserves_inputs():
    data, huy, app = submissions()
    app["rows"][0]["violence"] = 1
    app["rows"][0]["violence_evidence"] = [{"start_seconds": 1, "end_seconds": 2, "reason": "test fixture"}]
    original = deepcopy(app)
    review = compare(data, huy, app)
    assert review["status"] == "needs_adjudication"
    with pytest.raises(ValueError, match="human review"):
        make_gold(data, huy, app, review)
    review["decisions"][0].update(value=None, reviewer="test-reviewer", reason="fixture uncertain", reviewed=True)
    gold = make_gold(data, huy, app, review)
    assert gold["records"][0]["labels"]["violence"] is None
    assert app == original


def test_all_negative_agreement_does_not_invent_kappa_or_positive_f1():
    assert agreement([0, 0], [0, 0])["cohen_kappa"] is None
    result = binary_metrics([0, 0], [0, 0])
    assert result["recall_conditional_on_scored"] is None
    assert result["f1_conditional_on_scored"] is None


def test_missing_scores_are_abstentions_not_negatives():
    result = binary_metrics([1, 1, 0], [1, None, 0])
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["abstained_positive"] == 1
    assert result["confusion_matrix_scored"] == [[1, 0], [0, 1]]
    assert risk_union([0, None]) is None
    assert risk_union([1, None]) == 1


def test_ablation_uses_same_ids_and_target_for_every_mode():
    rows = [{"video_id": "a", "labels": {"violence": 0, "toxic_language": 1}},
            {"video_id": "b", "labels": {"violence": 1, "toxic_language": 0}},
            {"video_id": "c", "labels": {"violence": 0, "toxic_language": 0}}]
    scores = {"a": {"violence": .1, "toxic_language": .9}, "b": {"violence": .9, "toxic_language": None},
              "c": {"violence": .1, "toxic_language": .1}}
    result = evaluate_predictions(rows, scores, {"violence": .43, "toxic_language": .59})
    assert result["ablation_video_ids"] == ["a", "c"]
    assert all(v["support"] == 2 for v in result["ablation"].values())
    assert result["ablation"]["visual_only"]["recall_conditional_on_scored"] == 0
    assert result["ablation"]["fusion"]["recall_conditional_on_scored"] == 1
    assert result["per_label"]["toxic_language"]["abstained"] == 1
    with pytest.raises(ValueError, match="cohort mismatch"):
        evaluate_predictions(rows, {"a": scores["a"]}, {"violence": .43, "toxic_language": .59})


def test_model_bundle_verification_rejects_changed_bytes_and_extra_paths(tmp_path):
    content = b"small test fixture, not a checkpoint"
    manifest = {"version": "fixture", "files": [{"path": "model.bin", "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()}]}
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("model.bin", content)
        z.writestr("bundle-manifest.json", "{}")
    assert install_bundle(archive, tmp_path / "models", manifest)["verified_files"] == 1
    (tmp_path / "models/model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_artifacts(tmp_path / "models", manifest)
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="entries"):
        install_bundle(archive, tmp_path / "models", manifest)
    assert not (tmp_path / "escape").exists()
