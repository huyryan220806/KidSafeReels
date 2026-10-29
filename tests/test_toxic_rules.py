"""Synthetic regression cases, not an independent moderation benchmark."""
import json
import unicodedata

import pytest

from apps.validation import validate_result
from kidsafe.errors import ModelNotReadyError
from ml.inference import VideoService, combine_result
from ml.toxic_rules import ToxicRules, lexical_decision, load_rules, rule_audio_eligible

GOOD_AUDIO = {"avg_logprob": -.4, "compression_ratio": 1.4, "no_speech_prob": .9}


@pytest.mark.parametrize("text", [
    "Tôi đéo đồng ý.", "ĐỒ NGU!", "đồ\n  mất dạy", "\"thằng chó\"",
    unicodedata.normalize("NFD", "mày là đồ khốn"),
])
def test_explicit_words_and_phrases_match_with_unicode_boundaries(text):
    result = lexical_decision(load_rules().match(text), GOOD_AUDIO)
    assert result["flagged"]


@pytest.mark.parametrize("text", [
    "Đừng đụng vào cái lồng chim.", "Hôm nay là một buổi sáng đẹp.",
    "Mẹ dắt con chó đi dạo.", "Cá chết vì thiếu oxy.",
    "Bài học về sinh sản và sức khỏe tình dục.", "Đừng chửi tục hay lăng mạ người khác.",
    "Kéo cái lon lại đây.", "deo", "dit me", "buoi", "đồ. Nguồn nước rất sạch.",
    "đéox", "xđéo", "đụng", "lồng", "", "  ",
])
def test_ordinary_words_unaccented_and_substrings_do_not_trigger(text):
    assert load_rules().match(text) == []


def test_quoted_profanity_keeps_project_label_definition():
    assert lexical_decision(load_rules().match('Anh ta nói "đồ ngu".'), GOOD_AUDIO)["flagged"]


@pytest.mark.parametrize("text", ["đm", "ĐCM", "vcl", "VKL"])
def test_ambiguous_abbreviations_require_review(text):
    result = lexical_decision(load_rules().match(text), GOOD_AUDIO)
    assert result["needs_review"] and not result["flagged"]


@pytest.mark.parametrize("quality", [None, {}, {"avg_logprob": -.3},
    {"avg_logprob": -1.1, "compression_ratio": 1.3},
    {"avg_logprob": -.3, "compression_ratio": 2.41},
    {"avg_logprob": float("nan"), "compression_ratio": 1.3},
    {"avg_logprob": -.3, "compression_ratio": float("inf")},
    {"avg_logprob": True, "compression_ratio": 1.3}])
def test_missing_or_weak_audio_cannot_flag_by_rules_alone(quality):
    result = lexical_decision(load_rules().match("đồ ngu"), quality)
    assert result["needs_review"] and not result["flagged"]


def test_good_logprob_is_not_overridden_by_high_no_speech():
    assert rule_audio_eligible(GOOD_AUDIO)


class TextModel:
    def __init__(self, score):
        self.score = score

    def predict(self, text):
        return {"score": self.score, "flagged": self.score >= .78}


def combine(text="Tôi đéo biết.", score=.02, quality=GOOD_AUDIO, hybrid=True):
    return combine_result(.1, .41, [(1, 2, text, quality)], TextModel(score), 3,
                          "fixture+" + load_rules().version, [], toxic_rules=load_rules() if hybrid else None)


def test_hybrid_rule_flags_without_fabricating_probability():
    result = validate_result(combine(), backend="model", duration=3)
    assert result.decision == "flagged"
    assert result.scores.toxic_language == .02
    assert result.evidence[0].score == .02
    assert "Bộ luật từ/cụm từ" in result.evidence[0].description
    assert "Model Toxic vượt ngưỡng" not in result.evidence[0].description
    assert result.scores.sexual is None and result.scores.dangerous_acts is None


def test_model_only_is_preserved_for_comparison():
    result = combine(hybrid=False)
    assert result.decision == "review" and not result.evidence
    assert result.scores.toxic_language == .02


def test_model_positive_without_keywords_still_flags():
    result = combine("Một câu thử nghiệm không chứa từ khóa", score=.9)
    assert result.decision == "flagged"
    assert "Model Toxic vượt ngưỡng" in result.evidence[0].description
    assert "Bộ luật từ/cụm từ" not in result.evidence[0].description


def test_both_sources_share_evidence_without_duplicate_segment():
    result = combine(score=.9)
    assert len(result.evidence) == 1
    assert "Model Toxic" in result.evidence[0].description and "Bộ luật" in result.evidence[0].description


def test_low_asr_quality_or_abbreviation_only_returns_review_evidence():
    for result in (combine(quality=None), combine(text="vcl")):
        assert result.decision == "review" and result.evidence
        assert result.scores.toxic_language == .02


def test_rules_cannot_score_absent_audio_and_model_score_zero_is_preserved():
    result = combine_result(.1, .41, [], TextModel(.02), 3, "fixture", [], toxic_rules=load_rules())
    assert result.scores.toxic_language is None and result.decision == "review"
    assert not result.evidence
    assert combine(score=0).scores.toxic_language == 0


@pytest.mark.parametrize("mode", ["invalid", ""])
def test_invalid_policy_fails_before_loading_models(tmp_path, mode):
    with pytest.raises(ModelNotReadyError, match="KIDSAFE_TOXIC_MODE"):
        VideoService(tmp_path, toxic_mode=mode)


def test_lexicon_version_tracks_file_and_rejects_duplicates(tmp_path):
    from ml.toxic_rules import LEXICON
    data=json.loads(LEXICON.read_text(encoding="utf-8"))
    path=tmp_path/"lexicon.json"
    path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    assert ToxicRules(path).version != load_rules().version  # byte identity is explicit
    data["rules"].append(dict(data["rules"][0]))
    path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        ToxicRules(path)
