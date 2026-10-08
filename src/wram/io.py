"""Manifest and result file helpers."""

from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path):
    path = Path(path)
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    seen = set()
    for row in rows:
        if set(row) - {"id", "path", "split", "machine", "part", "domain"}:
            raise ValueError(
                "Inference manifests accept identity, path, split, machine,"
                " part and normal domain only; labels belong in a separate CSV"
            )
        for key in ["id", "path", "split", "machine", "part"]:
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Invalid manifest field {key}")
        if row["id"] in seen:
            raise ValueError("Duplicate recording id")
        seen.add(row["id"])
        if row["part"] not in ["normal", "query"]:
            raise ValueError("part must be normal or query")
        if row["part"] == "normal" and row.get("domain") not in [
            "source",
            "target",
        ]:
            raise ValueError("Training normal domain is required")
        for key in ["split", "machine"]:
            if any(s in row[key] for s in ["/", "\\", ".."]):
                raise ValueError("Unsafe group name")
        audio = Path(row["path"])
        row["path"] = str(
            (path.parent / audio).resolve()
            if not audio.is_absolute()
            else audio
        )
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def groups(rows, part):
    keys = sorted(
        {(r["split"], r["machine"]) for r in rows if r["part"] == part}
    )
    return [
        (
            s,
            m,
            [
                r
                for r in rows
                if (r["split"], r["machine"], r["part"]) == (s, m, part)
            ],
        )
        for s, m in keys
    ]


def check_run(run):
    run = Path(run)
    info = run / "RUN.json"
    if not info.exists():
        info = run / "RUN_CONTRACT.json"
    state = read(info)
    if state.get("status") != "completed":
        raise ValueError("Inference did not complete")
    if not any((run / "predictions").glob("*.json")):
        raise ValueError("No predictions found")
    return state
