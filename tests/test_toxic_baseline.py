import pytest

pytest.importorskip("sklearn")
from ml.toxic_baseline import clean_splits, normalize, read_split, choose_threshold


def test_split_cleaning_removes_leakage_and_conflicts():
    splits = {
        "train": [("clean", 0), ("bad", 1), ("clean", 0), ("conflict", 0), ("conflict", 1)],
        "valid": [("clean", 0), ("vclean", 0), ("vbad", 1)],
        "test": [("vbad", 1), ("tclean", 0), ("tbad", 1)],
    }
    result, audit = clean_splits(splits)
    assert result["train"] == [("clean", 0), ("bad", 1)]
    assert audit["train"]["conflicting_texts"] == 1
    assert audit["test"]["overlap_with_earlier_split"] == 1
    assert not set(t for t, _ in result["train"]) & set(t for t, _ in result["valid"])


def test_unicode_and_whitespace_normalization():
    assert normalize("  CHÀO\n bạn  ") == normalize("cha\u0300o bạn")


def test_missing_labels_are_not_negative(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("Comment,Toxicity\nhello,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_split(path)


def test_threshold_separates_validation_example():
    assert 0.2 < choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]) <= 0.7


def test_saved_classifier_round_trip_and_empty_input(tmp_path):
    import joblib
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from ml.toxic_baseline import ToxicClassifier

    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(["good friendly", "nice friendly", "bad rude", "bad hostile"], [0, 0, 1, 1])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "threshold": 0.5, "version": "test-only",
                 "sklearn_version": sklearn.__version__}, path)
    classifier = ToxicClassifier(path)
    result = classifier.predict(" BAD rude ")
    assert result["score"] == pytest.approx(model.predict_proba(["bad rude"])[0, 1])
    assert result["scope"] == "text_only"
    with pytest.raises(ValueError):
        classifier.predict("  ")
