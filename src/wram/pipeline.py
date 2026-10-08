"""Extract descriptors, score queries, and evaluate DCASE results."""

from pathlib import Path
import csv
import json
import random
import time
import traceback

import numpy as np
import torch

from .io import (
    read,
    dump,
    load_manifest,
    groups,
    now,
    write_csv,
    check_run,
)
from .frontend import Encoder
from .reference import fit_banks, score_banks, fixed_id_score
from .metrics import components, harmonic

PRIMARY = "wiener_rdp4|fps|ratio|1|updated"


def setup(config):
    if config.get("wiener_mode") != "floor_only":
        raise ValueError(
            "This implementation uses floor-only Wiener prediction"
        )
    if config["suite"] not in ["main", "ablations"]:
        raise ValueError("Unknown suite")
    if config["batch_size"] != 16:
        raise ValueError(
            "Verified recipe requires batch size16 with padded tails"
        )
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def json_banks(banks):
    return {
        name: {
            "indices": v["indices"].tolist(),
            "statistics": {k: a.tolist() for k, a in v["statistics"].items()},
        }
        for name, v in banks.items()
    }


def numpy_banks(banks):
    return {
        name: {
            "indices": np.asarray(v["indices"], dtype=np.int64),
            "statistics": {
                k: np.asarray(a) for k, a in v["statistics"].items()
            },
        }
        for name, v in banks.items()
    }


def extract(encoder, rows, config, folder):
    folder.mkdir(parents=True, exist_ok=False)
    parts = {}
    offset = 0
    for values, ids in encoder.batches(rows, config):
        for key, x in values.items():
            parts.setdefault(key, []).append(x)
        if [r["id"] for r in rows[offset : offset + len(ids)]] != ids:
            raise AssertionError("Record ordering changed")
        offset += len(ids)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    for key, value in arrays.items():
        np.save(folder / (key + ".npy"), value, allow_pickle=False)
    dump(
        folder / "RECORDS.json",
        [{k: v for k, v in row.items() if k != "path"} for row in rows],
    )
    return arrays


def run(manifest, checkpoint, config_path, output, device="cuda", smoke=False):
    output = Path(output)
    config = read(config_path)
    setup(config)
    rows = load_manifest(manifest)
    normal = groups(rows, "normal")
    query = groups(rows, "query")
    if {(s, m) for s, m, _ in normal} != {(s, m) for s, m, _ in query}:
        raise ValueError("Normal/query machine sets differ")
    for s, m, g in normal:
        if len(g) != 1000 or sum(r["domain"] == "source" for r in g) != 990:
            raise ValueError(
                f"Expected 990 source +10 target normals: {s}/{m}"
            )
    if any(len(g) != 200 for _, _, g in query):
        raise ValueError("Expected200 query recordings per machine")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(
        status="running",
        started_at=now(),
        config=config,
        device=device,
        smoke=smoke,
    )
    dump(output / "RUN.json", state)
    try:
        encoder = Encoder(
            checkpoint,
            config["checkpoint_sha256"],
            device,
            config["precision"],
        )
        selected = normal[:1] if smoke else normal
        for split, machine, g in selected:
            folder = output / "features" / split / machine / "normal"
            values = extract(encoder, g[:16] if smoke else g, config, folder)
            if smoke:
                continue
            banks = {
                view: fit_banks(
                    v,
                    [r["domain"] for r in g],
                    config["memory_size"],
                    (
                        config["random_seeds"]
                        if view == "wiener_rdp4"
                        and config["suite"] == "ablations"
                        else ()
                    ),
                    device,
                    extra_sizes=(
                        config.get("extra_memory_sizes", [])
                        if view == "wiener_rdp4"
                        and config["suite"] == "ablations"
                        else ()
                    ),
                )
                for view, v in values.items()
            }
            dump(
                output / "models" / f"{split}_{machine}.json",
                {view: json_banks(b) for view, b in banks.items()},
            )
            print(
                json.dumps(
                    dict(
                        event="normal_frozen",
                        split=split,
                        machine=machine,
                        views=list(values),
                        elapsed_seconds=round(time.monotonic() - started, 1),
                    )
                ),
                flush=True,
            )
        if not smoke:
            for split, machine, g in query:
                folder = output / "features" / split / machine / "query"
                values = extract(encoder, g, config, folder)
                bank_config = read(
                    output / "models" / f"{split}_{machine}.json"
                )
                normal_root = output / "features" / split / machine / "normal"
                normals = {
                    view: np.load(normal_root / (view + ".npy"), mmap_mode="r")
                    for view in values
                }
                predictions = {}
                traces = {}
                for view, q in values.items():
                    banks = numpy_banks(bank_config[view])
                    result = score_banks(
                        normals[view],
                        q,
                        banks,
                        device,
                        controls=config["suite"] == "ablations",
                    )
                    for key, v in result.items():
                        name = view + "|" + key
                        predictions[name] = v["scores"].tolist()
                        if view == "wiener_rdp4":
                            traces[name + "|bands"] = v["bands"]
                            traces[name + "|winners"] = v["winners"]
                if config["suite"] == "ablations":
                    for representation in ["wiener_mean", "wiener_rdp4"]:
                        for id_source in ["wiener_mean", "wiener_rdp4"]:
                            ix = np.asarray(
                                bank_config[id_source]["fps"]["indices"]
                            )
                            key = f"crossover|{representation}|{id_source}"
                            predictions[key] = fixed_id_score(
                                normals[representation],
                                values[representation],
                                ix,
                                device,
                            ).tolist()
                target = output / "predictions"
                target.mkdir(exist_ok=True)
                dump(
                    target / f"{split}_{machine}.json",
                    dict(
                        split=split,
                        machine=machine,
                        ids=[r["id"] for r in g],
                        scores=predictions,
                    ),
                )
                np.savez_compressed(
                    target / f"{split}_{machine}_traces.npz", **traces
                )
                print(
                    json.dumps(
                        dict(
                            event="query_scored",
                            split=split,
                            machine=machine,
                            conditions=len(predictions),
                            elapsed_seconds=round(
                                time.monotonic() - started, 1
                            ),
                        )
                    ),
                    flush=True,
                )
        state.update(
            status="completed",
            finished_at=now(),
            elapsed_seconds=time.monotonic() - started,
        )
        dump(output / "RUN.json", state)
        print(
            json.dumps(
                dict(
                    event="completed",
                    seconds=state["elapsed_seconds"],
                    output=str(output),
                )
            ),
            flush=True,
        )
    except BaseException:
        state.update(
            status="failed", error=traceback.format_exc(), finished_at=now()
        )
        dump(output / "RUN.json", state)
        raise


def load_labels(path):
    rows = list(csv.DictReader(Path(path).open()))
    lookup = {}
    for row in rows:
        if set(row) != {"id", "split", "machine", "domain", "label"}:
            raise ValueError(
                "Label columns must be id,split,machine,domain,label"
            )
        if (
            row["id"] in lookup
            or row["domain"] not in ["source", "target"]
            or row["label"] not in ["0", "1"]
        ):
            raise ValueError("Invalid/duplicate label row")
        lookup[row["id"]] = row
    return lookup


def evaluate(run, labels_path, output):
    run = Path(run)
    output = Path(output)
    check_run(run)
    output.mkdir(parents=True, exist_ok=False)
    lookup = load_labels(labels_path)
    records = []
    groups_metric = {}
    error = 0.0
    used = set()
    for file in sorted((run / "predictions").glob("*.json")):
        data = read(file)
        ids = data["ids"]
        used.update(ids)
        truth = [lookup[i] for i in ids]
        if any(
            r["split"] != data["split"] or r["machine"] != data["machine"]
            for r in truth
        ):
            raise ValueError("Label join mismatch")
        y = np.asarray([int(r["label"]) for r in truth])
        domains = np.asarray([r["domain"] for r in truth])
        if len(ids) != len(set(ids)) or len(ids) != 200:
            raise ValueError("Incorrect query coverage")
        if any(
            np.sum((y == label) & (domains == domain)) != 50
            for label in [0, 1]
            for domain in ["source", "target"]
        ):
            raise ValueError("Incorrect DCASE domain/class counts")
        for condition, values in data["scores"].items():
            comp, err = components(y, domains, values)
            error = max(error, err)
            groups_metric.setdefault((data["split"], condition), []).extend(
                comp
            )
            records.append(
                dict(
                    split=data["split"],
                    machine=data["machine"],
                    condition=condition,
                    source_auc=100 * comp[0],
                    target_auc=100 * comp[1],
                    pauc=100 * comp[2],
                    hmean=100 * harmonic(comp),
                )
            )
    if used != set(lookup):
        raise ValueError("Missing or surplus labels")
    aggregates = [
        dict(split=s, condition=k, hmean=100 * harmonic(v), components=len(v))
        for (s, k), v in sorted(groups_metric.items())
    ]
    write_csv(output / "MACHINE_METRICS.csv", records)
    write_csv(output / "AGGREGATES.csv", aggregates)
    random_rows = []
    for split in sorted({r["split"] for r in aggregates}):
        for k in [1, 4]:
            values = [
                r["hmean"]
                for r in aggregates
                if r["split"] == split
                and r["condition"].startswith("wiener_rdp4|random_")
                and r["condition"].endswith(f"|ratio|{k}|updated")
            ]
            if values:
                random_rows.append(
                    dict(
                        split=split,
                        k=k,
                        mean_hmean=float(np.mean(values)),
                        sample_sd=float(np.std(values, ddof=1)),
                        seeds=len(values),
                    )
                )
    if random_rows:
        write_csv(output / "RANDOM_SUMMARY.csv", random_rows)
    main = [r for r in aggregates if r["condition"] == PRIMARY]
    lines = [
        "# Floor-only WRAM reproduction",
        "",
        (
            "Point estimates on historically observed Dev/Eval. No new"
            " coefficient search."
        ),
        "",
        "| Split | hmean (%) |",
        "|---|---:|",
    ]
    lines.extend(f'| {r["split"]} | {r["hmean"]:.6f} |' for r in main)
    lines += [
        "",
        (
            "Full ablations: AGGREGATES.csv and MACHINE_METRICS.csv. Random"
            " controls: RANDOM_SUMMARY.csv."
        ),
        (
            "The primary pipeline has no power-dependent loading; denominator"
            " epsilon remains 1e-12."
        ),
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    dump(
        output / "VALIDATION.json",
        dict(
            numeric_status="PASS",
            max_independent_roc_error=error,
            query_recordings=len(used),
            primary=main,
        ),
    )
    print(
        json.dumps(
            dict(
                event="evaluation_complete", primary=main, max_roc_error=error
            )
        ),
        flush=True,
    )
