"""Discover licensed Vietnamese Commons sources; metadata only, no model predictions."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import hashlib

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "KidSafeReelsCoursePilot/0.1 (https://github.com/huyryan220806/KidSafeReels)"}
CATEGORIES = ["Videos in Vietnamese", "Videos by Galaxy Studio", "Voice of America videos in Vietnamese",
              "Videos by Terra X in Vietnamese", "Videos about COVID-19 in Vietnamese"]
# Fixed without looking at moderation predictions. Related publishers/sites stay together.
SELECTION = [
    (72153596, "saigon-post-office", "validation", [0, 30]),
    (72153708, "saigon-post-office", "validation", [0, 30]),
    (72153611, "thanhhoa-postal-news", "train", [0, 45, 90]),
    (72153742, "thanhhoa-postal-news", "train", [0, 45, 90]),
    (72153668, "nghean-postal-news", "train", [0, 40, 80, 120, 160]),
    (88950159, "vnctv-shopping-covid", "train", [0, 30, 60, 90, 120, 150]),
    (88778336, "sfgov-health-vietnamese", "test", [0]),
    (88778345, "sfgov-health-vietnamese", "test", [0, 60, 120]),
]


def plain(value):
    return html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()


def query(**params):
    response = requests.get(API, params={"action": "query", "format": "json", **params}, headers=HEADERS, timeout=(15, 60))
    response.raise_for_status()
    result = response.json()
    if "error" in result:
        raise ValueError(result["error"])
    return result


def discover():
    directory = ROOT / "data/raw/pilot"
    directory.mkdir(parents=True, exist_ok=True)
    pages = {}
    for category in CATEGORIES:
        result = query(list="categorymembers", cmtitle=f"Category:{category}", cmtype="file", cmlimit=100)
        if "continue" in result:
            raise ValueError("Category grew beyond discovery limit; review pagination")
        for member in result["query"]["categorymembers"]:
            pages.setdefault(member["pageid"], {**member, "categories": []})["categories"].append(category)
    ids = sorted(pages)
    details = []
    for start in range(0, len(ids), 10):
        result = query(pageids="|".join(map(str, ids[start:start + 10])), prop="imageinfo|info",
                       iiprop="url|size|sha1|timestamp|extmetadata|metadata", inprop="url")
        for key, page in result["query"]["pages"].items():
            page["discovery_categories"] = pages[int(key)]["categories"]
            details.append(page)
    payload = {"queried_at": datetime.now(timezone.utc).isoformat(), "categories": CATEGORIES, "pages": details}
    (directory / "commons-discovery.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for page in sorted(details, key=lambda p: p["pageid"]):
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        print(json.dumps({"pageid": page["pageid"], "title": page["title"], "size": info.get("size"),
                          "license": meta.get("LicenseShortName", {}).get("value"),
                          "license_url": meta.get("LicenseUrl", {}).get("value"),
                          "duration": duration(info)}, ensure_ascii=True), flush=True)


def duration(info):
    values = {x["name"]: x.get("value") for x in info.get("metadata", [])}
    return float(values.get("playtime_seconds", values.get("length", 0)))


def freeze():
    """Create the reviewed manifest once, BEFORE downloading or extracting media."""
    target = ROOT / "data/pilot/manifest.json"
    if target.exists():
        raise FileExistsError("Frozen manifest exists; use a new dataset version for changes")
    discovery = json.loads((ROOT / "data/raw/pilot/commons-discovery.json").read_text(encoding="utf-8"))
    pages = {p["pageid"]: p for p in discovery["pages"]}
    rights = query(pageids="|".join(str(x[0]) for x in SELECTION), prop="revisions",
                   rvprop="ids|timestamp|content", rvslots="main")["query"]["pages"]
    sources, clips = [], []
    for pageid, group, split, starts in SELECTION:
        page = pages[pageid]
        info = page["imageinfo"][0]
        meta = info["extmetadata"]
        revision = rights[str(pageid)]["revisions"][0]
        wiki = revision["slots"]["main"]["*"]
        review = re.search(r"\{\{YouTubeReview\|([^}]+)\}\}", wiki)
        if not review or re.search(r"\{\{[Ll]icense[Rr]eview", wiki):
            raise ValueError(f"License evidence needs review: {pageid}")
        fields = dict(part.split("=", 1) for part in review.group(1).split("|") if "=" in part)
        if meta["LicenseShortName"]["value"] != "CC BY 3.0":
            raise ValueError(f"Unexpected license: {pageid}")
        source_id = f"commons-{pageid}"
        source = {"source_id": source_id, "pageid": pageid, "title": page["title"][5:],
                  "source_url": page["fullurl"], "original_url": f"https://www.youtube.com/watch?v={fields['id']}",
                  "download_url": info["url"], "sha1": info["sha1"], "size_bytes": info["size"],
                  "duration_seconds": duration(info), "file_timestamp": info["timestamp"],
                  "author": plain(meta["Artist"]["value"]), "license": "CC-BY-3.0",
                  "license_url": "https://creativecommons.org/licenses/by/3.0/",
                  "rights_evidence_url": f"https://commons.wikimedia.org/w/index.php?oldid={revision['revid']}",
                  "license_review": "Commons YouTubeReviewBot confirmed license tag; not a rights ownership guarantee",
                  "license_review_date": fields["date"], "license_archive_url": fields["archive"],
                  "group_id": group, "split": split,
                  "group_reason": "Same source, related publisher/production or same filming site kept in one split"}
        sources.append(source)
        for start in starts:
            if start + 25 > source["duration_seconds"]:
                raise ValueError("Clip exceeds source duration")
            video_id = f"vi-pilot-{len(clips) + 1:03d}"
            clips.append({"video_id": video_id, "source_id": source_id, "group_id": group, "split": split,
                          "start_seconds": start, "end_seconds": start + 25,
                          "video_path": f"data/processed/pilot-v1/media/{video_id}.mp4",
                          "expected_language": "vi", "language_verified_by_human": None,
                          "labels": {"violence": None, "toxic_language": None,
                                     "sexual": None, "dangerous_acts": None},
                          "annotation_status": "pending",
                          "changes": "25-second excerpt; resize <=640x480 preserving aspect ratio; H264/AAC; no dubbing"})
    payload = {"dataset_version": "vi-public-pilot-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
               "selection_policy": "Rights and source diversity only; no predictions or labels used. Fixed non-overlapping excerpts.",
               "split_policy": "17/4/4 clips (68/16/16), grouped before any frame extraction; 3/1/1 groups.",
               "limitations": ["25 excerpts from 8 parent videos and 5 groups, NOT 25 independent sources",
                               "Convenience sample dominated by informational content; positive class coverage unknown",
                               "Vietnamese language requires human confirmation; titles are not ground truth",
                               "Not representative of TikTok/Reels/Shorts; not a trained moderation dataset"],
               "sources": sources, "clips": clips}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".sha256").write_text(digest + "  manifest.json\n", encoding="ascii")
    print(json.dumps({"manifest": str(target), "sha256": digest, "sources": len(sources), "clips": len(clips)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["discover", "freeze"])
    args = parser.parse_args()
    {"discover": discover, "freeze": freeze}[args.command]()
