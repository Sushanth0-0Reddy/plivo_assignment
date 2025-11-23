import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path: Path):
    stats = Counter()
    lengths = Counter()
    issues = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"]
            for ent in obj.get("entities", []):
                start = ent["start"]
                end = ent["end"]
                label = ent["label"]
                if start < 0 or end > len(text) or start >= end:
                    issues.append((obj["id"], "bounds", start, end))
                    continue
                span = text[start:end]
                if len(span) != end - start:
                    issues.append((obj["id"], "length", start, end))
                if not span.strip():
                    issues.append((obj["id"], "blank", start, end))
                stats[label] += 1
                lengths[label] += end - start
    return stats, lengths, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    label_totals = Counter()
    length_totals = Counter()
    issue_map = defaultdict(list)

    for fname in ["train.jsonl", "dev.jsonl"]:
        stats, lengths, issues = analyze(args.data_dir / fname)
        label_totals.update(stats)
        for label, total in lengths.items():
            length_totals[label] += total
        if issues:
            issue_map[fname].extend(issues)

    print("Label counts:")
    for label, count in label_totals.items():
        avg = length_totals[label] / count
        print(f"  {label:15s}: {count:4d} spans, avg length {avg:.1f}")

    total_spans = sum(label_totals.values())
    print(f"Total spans: {total_spans}")

    if issue_map:
        print("Issues detected:")
        for fname, entries in issue_map.items():
            print(f"  {fname}: {len(entries)} issues (first 5 shown)")
            for issue in entries[:5]:
                print("    ", issue)
    else:
        print("No span alignment issues detected.")


if __name__ == "__main__":
    main()

