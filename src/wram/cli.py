"""Command line interface. Inference never accepts a labels argument."""

import argparse
import json
from pathlib import Path

from .io import dump


def index(root, split, output):
    root = Path(root).resolve()
    rows = []
    for machine in sorted(p for p in root.iterdir() if p.is_dir()):
        for dirname, part in [("train", "normal"), ("test", "query")]:
            folder = machine / dirname
            for path in sorted(folder.glob("*.wav")):
                name = path.name
                if "section_00_" not in name:
                    raise ValueError("This recipe supports section00 only")
                row = dict(
                    id=split + "/" + str(path.relative_to(root)),
                    path=str(path),
                    split=split,
                    machine=machine.name,
                    part=part,
                )
                if part == "normal":
                    if "_normal_" not in name:
                        raise ValueError("Non-normal file in training folder")
                    row["domain"] = (
                        "source"
                        if "_source_" in name
                        else "target" if "_target_" in name else None
                    )
                rows.append(row)
    target = Path(output)
    with target.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} records to {target}")


def main():
    parser = argparse.ArgumentParser(prog="wram")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "index",
        help="Index already-downloaded DCASE machine/{train,test} folders",
    )
    p.add_argument("--root", required=True)
    p.add_argument("--split", required=True, choices=["Dev", "Eval"])
    p.add_argument("--output", required=True)
    for command in ["run", "smoke"]:
        p = commands.add_parser(
            command, help="Frozen audio inference; no labels read"
        )
        p.add_argument("--manifest", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p = commands.add_parser(
        "evaluate",
        help="Evaluate frozen predictions using separately supplied labels",
    )
    p.add_argument("--run", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser(
        "analyze",
        help=(
            "Post-hoc embedding diagnostics; labels may be used for probes"
            " only"
        ),
    )
    p.add_argument("--run", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "index":
        index(args.root, args.split, args.output)
    elif args.command in ["run", "smoke"]:
        from .pipeline import run

        run(
            args.manifest,
            args.checkpoint,
            args.config,
            args.output,
            args.device,
            args.command == "smoke",
        )
    elif args.command == "evaluate":
        from .pipeline import evaluate

        evaluate(args.run, args.labels, args.output)
    else:
        from .analysis import analyze

        analyze(args.run, args.labels, args.output)
