"""Versioned Vietnamese lexical evidence; never invent a model probability."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

LEXICON = Path(__file__).resolve().parent / "resources/toxic_lexicon_v1.json"
POLICY_VERSION = "toxic-hybrid-v1"
MODES = ("hybrid", "model_only")


def normalize_rule_text(text):
    # Preserve Vietnamese tone marks. Do not guess ASR substitutions or strip accents.
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


class ToxicRules:
    def __init__(self, path=LEXICON):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if payload.get("version") != "toxic-lexicon-v1" or payload.get("language") != "vi":
            raise ValueError("Unsupported Toxic lexicon")
        self.version = f"{POLICY_VERSION}-{self.sha256[:12]}"
        self.rules, ids, terms = [], set(), set()
        for row in payload["rules"]:
            if (not isinstance(row, dict) or set(row) != {"id", "term", "action", "category"}
                    or not all(isinstance(v, str) and v for v in row.values())
                    or not re.fullmatch(r"[a-z0-9-]+", row["id"])
                    or row["action"] not in {"flag", "review"}):
                raise ValueError("Invalid Toxic rule")
            term = normalize_rule_text(row["term"])
            if row["id"] in ids or term in terms or term != row["term"] or not re.fullmatch(r"\w+(?: \w+)*", term):
                raise ValueError("Duplicate or invalid lexical term")
            ids.add(row["id"]); terms.add(term)
            # Unicode word boundaries prevent 'đụ' matching 'đụng' and 'lồn' matching 'lồng'.
            self.rules.append((row, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")))
        if not self.rules:
            raise ValueError("Empty Toxic lexicon")

    def match(self, text):
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        normalized = normalize_rule_text(text)
        return [dict(row) for row, pattern in self.rules if pattern.search(normalized)]


@lru_cache(maxsize=1)
def load_rules():
    return ToxicRules()


def rule_audio_eligible(quality):
    """Heuristic for rule-only flags; not calibrated ASR correctness/confidence."""
    if not isinstance(quality, dict):
        return False
    values = [quality.get(k) for k in ("avg_logprob", "compression_ratio")]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        return False
    logprob, compression = values
    # Whisper's native silence filter still applies upstream. A high no_speech_prob
    # alone must NOT undo its joint log-probability decision (the 28 Oct bug).
    return -1.0 <= logprob <= 0 and 0 < compression <= 2.4


def lexical_decision(matches, quality):
    eligible = rule_audio_eligible(quality)
    flagged = eligible and any(m["action"] == "flag" for m in matches)
    return dict(flagged=flagged, needs_review=bool(matches) and not flagged,
                audio_eligible=eligible, matches=matches)
