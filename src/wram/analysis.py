"""Retrospective diagnostics.

Probe training here never changes the ASD system.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from .io import (
    read,
    dump,
    write_csv,
    validate_freeze,
    sha256,
    source_files,
    now,
)
from .pipeline import load_labels

REPS = ("near_mean", "wiener_mean", "wiener_rdp4")
PAIRS = (("near_mean", "wiener_mean"), ("wiener_mean", "wiener_rdp4"))


def vectors(x):
    return np.asarray(x, dtype=np.float64).reshape(len(x), -1) / np.sqrt(
        x.shape[1]
    )


def cka(a, b):
    a = vectors(a)
    b = vectors(b)
    a -= a.mean(0)
    b -= b.mean(0)
    x = a @ a.T
    y = b @ b.T
    return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))


def neighbors(x, k=10):
    d = 0.5 * (
        1
        - np.clip(
            np.einsum("ibd,jbd->ij", x, x, optimize=True) / x.shape[1], -1, 1
        )
    )
    np.fill_diagonal(d, np.inf)
    return np.argpartition(d, k - 1, axis=1)[:, :k]


def probe(x, y, strata):
    x = vectors(x)
    scores = np.empty(len(y))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260922)
    for train, test in folds.split(x, strata):
        classifier = LogisticRegression(
            C=1.0, solver="liblinear", max_iter=5000, random_state=20260922
        )
        classifier.fit(x[train], y[train])
        scores[test] = classifier.decision_function(x[test])
    return float(100 * roc_auc_score(y, scores))


def margin(y, score):
    normal, anomaly = score[y == 0], score[y == 1]
    variance = (
        (len(normal) - 1) * np.var(normal, ddof=1)
        + (len(anomaly) - 1) * np.var(anomaly, ddof=1)
    ) / (len(score) - 2)
    return float((anomaly.mean() - normal.mean()) / np.sqrt(variance))


def manifold(normal, query, domains):
    centroids = []
    for domain in ["source", "target"]:
        mean = normal[domains == domain].mean(0)
        centroids.append(mean / np.linalg.norm(mean, axis=-1, keepdims=True))
    distances = [
        0.5 * (1 - np.einsum("nbd,bd->nb", query, c)) for c in centroids
    ]
    return np.minimum(*distances)


def analyze(run, labels, output):
    run = Path(run)
    output = Path(output)
    validate_freeze(run)
    output.mkdir(parents=True, exist_ok=False)
    geometries = []
    diagnostics = []
    movement = []
    cache = {}
    for p in sorted((run / "predictions").glob("*.json")):
        entry = read(p)
        s, m = entry["split"], entry["machine"]
        root = run / "features" / s / m
        q = {rep: np.load(root / "query" / (rep + ".npy")) for rep in REPS}
        near = {rep: neighbors(q[rep]) for rep in REPS}
        cache[s, m] = (root, entry, q, near)
        for left, right in PAIRS:
            overlap = np.mean(
                [
                    len(set(a) & set(b)) / 10
                    for a, b in zip(near[left], near[right])
                ]
            )
            geometries.append(
                dict(
                    split=s,
                    machine=m,
                    left=left,
                    right=right,
                    linear_cka=cka(q[left], q[right]),
                    knn10_overlap=float(overlap),
                )
            )
    write_csv(output / "GEOMETRY.csv", geometries)
    dump(
        output / "LABEL_FREE_FREEZE.json",
        dict(
            time=now(),
            geometry_sha256=sha256(output / "GEOMETRY.csv"),
            labels_opened=False,
        ),
    )
    lookup = load_labels(labels)
    dump(
        output / "LABEL_ACCESS.json",
        dict(
            time=now(),
            labels_sha256=sha256(labels),
            use=(
                "Post-hoc diagnostic probes only; never ASD fitting or system"
                " selection"
            ),
        ),
    )
    for (s, m), (root, entry, query, near) in cache.items():
        y = np.array([int(lookup[i]["label"]) for i in entry["ids"]])
        domain = np.array(
            [lookup[i]["domain"] == "target" for i in entry["ids"]], dtype=int
        )
        strata = 2 * y + domain
        normal_domain = np.array(
            [r["domain"] for r in read(root / "normal" / "RECORDS.json")]
        )
        scores = {}
        for rep in REPS:
            normal = np.load(root / "normal" / (rep + ".npy"), mmap_mode="r")
            q = query[rep]
            values = manifold(normal, q, normal_domain)
            scores[rep] = values.mean(1)
            mask = y == 0
            diagnostics.append(
                dict(
                    split=s,
                    machine=m,
                    representation=rep,
                    anomaly_probe_auc=probe(q, y, strata),
                    normal_domain_probe_auc=probe(
                        q[mask], domain[mask], domain[mask]
                    ),
                    label_purity=float(np.mean(y[near[rep]] == y[:, None])),
                    domain_purity=float(
                        np.mean(domain[near[rep]] == domain[:, None])
                    ),
                    normal_manifold_margin=margin(y, scores[rep]),
                )
            )
        for left, right in PAIRS:
            delta = scores[right] - scores[left]
            movement.append(
                dict(
                    split=s,
                    machine=m,
                    left=left,
                    right=right,
                    normal_mean_distance_change=float(delta[y == 0].mean()),
                    anomaly_mean_distance_change=float(delta[y == 1].mean()),
                    selective_change=float(
                        delta[y == 1].mean() - delta[y == 0].mean()
                    ),
                )
            )
        print(f"Diagnostics completed: {s}/{m}", flush=True)
    write_csv(output / "DIAGNOSTICS.csv", diagnostics)
    write_csv(output / "MANIFOLD_MOVEMENT.csv", movement)
    summary = {}
    for left, right in PAIRS:
        pairs = [
            r for r in geometries if r["left"] == left and r["right"] == right
        ]
        old = [r for r in diagnostics if r["representation"] == left]
        new = [r for r in diagnostics if r["representation"] == right]
        summary[left + "->" + right] = dict(
            median_cka=float(np.median([r["linear_cka"] for r in pairs])),
            median_neighbor_overlap=float(
                np.median([r["knn10_overlap"] for r in pairs])
            ),
            median_probe_left=float(
                np.median([r["anomaly_probe_auc"] for r in old])
            ),
            median_probe_right=float(
                np.median([r["anomaly_probe_auc"] for r in new])
            ),
            probe_increases=sum(
                b["anomaly_probe_auc"] > a["anomaly_probe_auc"]
                for a, b in zip(old, new)
            ),
            purity_increases=sum(
                b["label_purity"] > a["label_purity"] for a, b in zip(old, new)
            ),
            margin_increases=sum(
                b["normal_manifold_margin"] > a["normal_manifold_margin"]
                for a, b in zip(old, new)
            ),
        )
    dump(
        output / "SUMMARY.json",
        dict(
            retrospective=True,
            diagnostic_probe_training=True,
            asd_system_retrained=False,
            summary=summary,
            source_hashes=source_files(),
            run_contract_sha256=sha256(run / "RUN_CONTRACT.json"),
        ),
    )
    validate_freeze(run)
