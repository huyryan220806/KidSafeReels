"""Measure how ASR/segmentation changes classifier outputs, without inventing Toxic labels."""
import argparse
import json

from ml.experiment_metrics import write_csv, write_json
from ml.model_experiments import locations
from ml.toxic_baseline import ToxicClassifier


def audit(run_id):
    private, public = locations(run_id, "asr")
    evaluation = json.loads((public / "evaluation.json").read_text(encoding="utf-8"))
    model = ToxicClassifier()
    rows, summary = [], []
    for config in dict.fromkeys(["base-greedy", evaluation["selected"]]):
        for split in ("dev", "test"):
            cases = []
            for path in sorted((private / config).glob(f"{split}-*.json")):
                result = json.loads(path.read_text(encoding="utf-8"))
                reference_score = model.predict(result["reference"])["score"]
                accepted = result["language"] == "vi" and result["language_probability"] >= .5
                segments = [s["text"].strip() for s in result["segments"]
                            if accepted and s["no_speech_prob"] < .6 and s["text"].strip()]
                scores = [model.predict(text)["score"] for text in segments]
                joined_score = model.predict(" ".join(segments))["score"] if segments else None
                for mode, score in (("segment_max", max(scores) if scores else None), ("joined_transcript", joined_score)):
                    case = dict(config=config, split=split, aggregation=mode, sample_id=result["sample_id"],
                                reference_score=reference_score, reference_flag=reference_score >= model.threshold,
                                asr_score=score, asr_flag=score >= model.threshold if score is not None else None,
                                segments=len(segments))
                    cases.append(case); rows.append(case)
            for mode in ("segment_max", "joined_transcript"):
                part = [r for r in cases if r["aggregation"] == mode]
                eligible = [r for r in part if r["asr_flag"] is not None]
                summary.append(dict(config=config, split=split, aggregation=mode, n=len(part),
                    unavailable=len(part)-len(eligible), reference_flags=sum(r["reference_flag"] for r in part),
                    asr_flags=sum(r["asr_flag"] for r in eligible),
                    added_alerts_vs_reference=sum(r["asr_flag"] and not r["reference_flag"] for r in eligible),
                    missing_alerts_vs_reference=sum(not r["asr_flag"] and r["reference_flag"] for r in eligible),
                    matching_outputs=sum(r["asr_flag"] == r["reference_flag"] for r in eligible),
                    available=len(eligible)))
    write_csv(private / "toxic_asr_audit.csv", rows)
    write_csv(public / "toxic_asr_output_changes.csv", summary)
    write_json(public / "toxic_asr_output_changes.json", dict(metrics=summary,
        interpretation="Agreement with classifier prediction on reference transcript, NOT Toxic accuracy or human-labelled FP/FN.",
        production_aggregation="segment_max retained; joined transcript is diagnostic only without target-video labels"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    audit(parser.parse_args().run_id)
