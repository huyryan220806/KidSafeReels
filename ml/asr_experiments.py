"""Small reproducible FLEURS ASR benchmark, separate from Toxic classification accuracy."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import tarfile
import time
import unicodedata

import numpy as np
import requests

from ml.experiment_metrics import write_csv, write_json
from ml.smoke_video import FLEURS_REVISION
from ml.toxic_baseline import ROOT, sha256

DATA = ROOT / "data/raw/fleurs-evaluation"
CONFIGS = {
    "base-greedy": {"directory": "faster-whisper-base", "beam_size": 1},
    "base-beam5": {"directory": "faster-whisper-base", "beam_size": 5, "temperature": 0},
    "small-beam5": {"directory": "faster-whisper-small", "beam_size": 5, "temperature": 0},
}


def normalized(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def errors(reference, hypothesis):
    ref, hyp = normalized(reference), normalized(hypothesis)
    words, predicted = ref.split(), hyp.split()
    chars, pred_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    return dict(word_errors=edit_distance(words, predicted), words=len(words),
                char_errors=edit_distance(chars, pred_chars), chars=len(chars))


def prepare(limit=16):
    manifest = DATA / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing["per_split"] != limit:
            raise ValueError("Sample count differs from frozen FLEURS manifest")
        return existing
    DATA.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}/data/vi_vn"
    rows, earlier_prompts = [], set()
    for split in ("dev", "test"):
        response = requests.get(f"{base}/{split}.tsv", timeout=(15, 90))
        response.raise_for_status()
        metadata = {p[1]: p for line in response.text.splitlines() if len(p := line.split("\t")) >= 7}
        selected_prompts = set()
        with requests.get(f"{base}/audio/{split}.tar.gz", stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
                for member in archive:
                    name = Path(member.name).name
                    meta = metadata.get(name)
                    if not member.isfile() or not meta or member.size > 5 * 1024**2:
                        continue
                    if meta[0] in selected_prompts or meta[0] in earlier_prompts:
                        continue
                    path = DATA / split / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(archive.extractfile(member).read())
                    rows.append(dict(split=split, prompt_id=meta[0], filename=name, reference=meta[2],
                                     sha256=sha256(path), path=str(path.relative_to(DATA))))
                    selected_prompts.add(meta[0])
                    if len(selected_prompts) == limit:
                        break
        if len(selected_prompts) != limit:
            raise ValueError("Insufficient unique FLEURS prompts")
        earlier_prompts.update(selected_prompts)
        print(f"FLEURS {split}: {len(selected_prompts)} unique prompts downloaded", flush=True)
    result = dict(dataset="google/fleurs", subset="vi_vn", revision=FLEURS_REVISION,
                  license="CC-BY-4.0", source="https://huggingface.co/datasets/google/fleurs",
                  sampling="first eligible unique prompts in pinned tar order; no shared prompt IDs across dev/test",
                  per_split=limit, rows=rows)
    write_json(manifest, result)
    return result


def prepare_small():
    from huggingface_hub import HfApi, snapshot_download
    target = ROOT / "models/checkpoints/faster-whisper-small"
    provenance = target / "provenance.json"
    if provenance.exists():
        return
    repo = "Systran/faster-whisper-small"
    revision = HfApi().model_info(repo).sha
    names = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=names)
    write_json(provenance, dict(repo=repo, revision=revision,
                               files=[dict(path=n, sha256=sha256(target / n)) for n in names]))


def measure(name, split, rows, private):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    config = CONFIGS[name]
    directory = ROOT / "models/checkpoints" / config["directory"]
    model = WhisperModel(str(directory), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    output = []
    for i, row in enumerate(r for r in rows if r["split"] == split):
        path = DATA / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError("FLEURS audio checksum mismatch")
        cache = private / name / f"{split}-{row['filename']}.json"
        if cache.exists():
            result = json.loads(cache.read_text(encoding="utf-8"))
        else:
            audio = decode_audio(str(path), sampling_rate=16000)
            start = time.perf_counter()
            chunks, info = model.transcribe(audio, task="transcribe", vad_filter=True, condition_on_previous_text=False,
                    **{k: v for k, v in config.items() if k != "directory"})
            segments = [dict(start=c.start, end=c.end, text=c.text, no_speech_prob=c.no_speech_prob,
                             avg_logprob=c.avg_logprob) for c in chunks]
            seconds = time.perf_counter() - start
            accepted = info.language == "vi" and info.language_probability >= .5
            usable = [s for s in segments if accepted and s["no_speech_prob"] < .6]
            transcript = " ".join(s["text"].strip() for s in usable)
            result = dict(config=name, split=split, sample_id=row["filename"], reference=row["reference"],
                          transcript=transcript, segments=segments, seconds=seconds, audio_seconds=len(audio)/16000,
                          language=info.language, language_probability=info.language_probability,
                          rejected=not bool(usable), **errors(row["reference"], transcript))
            write_json(cache, result)
        output.append(result)
        print(f"ASR {name} {split} {i+1}: WER={result['word_errors']/result['words']:.3f}, {result['seconds']:.1f}s", flush=True)
    del model
    total = {k: sum(r[k] for r in output) for k in ("word_errors", "words", "char_errors", "chars", "seconds", "audio_seconds")}
    return output, dict(config=name, split=split, n=len(output), **total,
                        wer=total["word_errors"]/total["words"], cer=total["char_errors"]/total["chars"],
                        real_time_factor=total["seconds"]/total["audio_seconds"], rejected=sum(r["rejected"] for r in output))


def run(run_id):
    from ml.model_experiments import locations
    private, public = locations(run_id, "asr")
    if (public / "evaluation.json").exists():
        raise ValueError("ASR evaluation already completed")
    data = prepare()
    prepare_small()
    write_json(public / "protocol.json", dict(run_id=run_id, configurations=CONFIGS,
               selection="minimum dev corpus WER among candidates with real-time factor <= 2.5; fallback baseline",
               normalization="NFC + casefold + punctuation to spaces; WER whitespace tokens, CER excludes whitespace",
               samples=32, manifest_sha256=sha256(DATA / "manifest.json"),
               limitations="16 utterances per split, read speech, not conversational TikTok or Toxic ground truth"))
    summary, individual = [], []
    selection_path = public / "selection.json"
    if selection_path.exists():
        selected = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    else:
        for name in CONFIGS:
            results, metrics = measure(name, "dev", data["rows"], private)
            individual.extend(results); summary.append(metrics)
            write_csv(public / "asr_metrics.csv", summary)
        eligible = [r for r in summary if r["real_time_factor"] <= 2.5]
        selected = min(eligible, key=lambda r: (r["wer"], r["cer"], r["real_time_factor"]))["config"] if eligible else "base-greedy"
        write_json(selection_path, dict(selected=selected, dev=summary,
                                       selected_at=datetime.now().astimezone().isoformat(), test_evaluated=False))
    if not summary:
        summary = json.loads(selection_path.read_text(encoding="utf-8"))["dev"]
    for name in dict.fromkeys(["base-greedy", selected]):
        results, metrics = measure(name, "test", data["rows"], private)
        individual.extend(results); summary.append(metrics)
    write_csv(public / "asr_metrics.csv", summary)
    write_csv(private / "asr_predictions.csv", [{k: v for k, v in r.items() if k != "segments"} for r in individual])
    write_json(public / "evaluation.json", dict(selected=selected, metrics=summary,
               evaluated_at=datetime.now().astimezone().isoformat(), sample_manifest_sha256=sha256(DATA / "manifest.json")))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, metric in zip(axes, ("wer", "cer", "real_time_factor")):
        bars = ax.bar(range(len(summary)), [r[metric] for r in summary])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(range(len(summary)), [r["config"]+"\n"+r["split"] for r in summary], rotation=30, ha="right")
        ax.set_title(metric + " (lower is better)")
    fig.suptitle("FLEURS Vietnamese — 16 unique prompts per split; CPU; not Toxic accuracy")
    for suffix in ("png", "svg"):
        fig.savefig(public / f"asr_comparison.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    run(parser.parse_args().run_id)
