"""Join already-downloaded official labels to a manifest;
never imported by inference.
"""

import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True, type=Path)
p.add_argument("--dev-labels", type=Path)
p.add_argument("--eval-labels", type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
records = [
    json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()
]
cache = {}
rows = []
for r in records:
    if r["part"] != "query":
        continue
    key = (r["split"], r["machine"])
    if key not in cache:
        root = a.dev_labels if r["split"] == "Dev" else a.eval_labels
        if root is None:
            raise ValueError(f"Missing official-label root for {key[0]}")
        name = f'ground_truth_{r["machine"]}_section_00_test.csv'

        def read(folder):
            return list(csv.reader((root / folder / name).open()))

        aliases = {
            v[1] + ".wav": v[0] for v in read("ground_truth_attributes")
        }
        y = {v[0]: int(float(v[1])) for v in read("ground_truth_data")}
        d = {v[0]: int(float(v[1])) for v in read("ground_truth_domain")}
        if set(y) != set(d):
            raise ValueError("Official domain/label coverage differs")
        cache[key] = (aliases, y, d, set())
    aliases, y, d, seen = cache[key]
    filename = Path(r["path"]).name
    canonical = aliases.get(filename, filename)
    if canonical in seen:
        raise ValueError("Duplicate official filename join")
    seen.add(canonical)
    rows.append(
        dict(
            id=r["id"],
            split=r["split"],
            machine=r["machine"],
            domain="source" if d[canonical] == 0 else "target",
            label=y[canonical],
        )
    )
for _, y, _, seen in cache.values():
    if seen != set(y):
        raise ValueError("Official label coverage is incomplete")
with a.output.open("x", newline="") as f:
    w = csv.DictWriter(
        f, fieldnames=["id", "split", "machine", "domain", "label"]
    )
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} joined labels")
