"""Render prepared real-data selection plots; matplotlib/sklearn,
no GPU needed.

Individual PNGs have transparent backgrounds and no text, axes, or titles.
Contact sheets and the index retain the interpretation and scales.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

BLUE = "#77add1"
RED = "#c4372d"
GREEN = "#16877e"
GOLD = "#df9b25"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", required=True, type=Path)
    a = p.parse_args()
    folder = a.folder
    data = np.load(folder / "selection_arrays.npz")
    meta = json.loads((folder / "PROVENANCE.json").read_text())
    tiles = folder / "tiles"
    tiles.mkdir(exist_ok=False)
    x = data["normal"].reshape(1000, -1)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=20260923)
    xy = pca.fit_transform(x)
    query_xy = pca.transform(data["query"].reshape(1, -1))[0]
    ppca = PCA(n_components=2, svd_solver="full")
    projected_xy = ppca.fit_transform(data["projected"])
    np.savez_compressed(
        folder / "display_coordinates.npz",
        original_pca=xy,
        projected_pca=projected_xy,
        query_pca=query_xy,
        original_components=pca.components_,
        original_mean=pca.mean_,
        projected_components=ppca.components_,
        projected_mean=ppca.mean_,
    )
    index = []

    def canvas(width=900, height=620):
        fig = plt.figure(figsize=(width / 150, height / 150), dpi=150)
        ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
        ax.set_axis_off()
        return fig, ax

    def save(fig, name, title, meaning, scale):
        fig.savefig(tiles / name, transparent=True, dpi=150)
        plt.close(fig)
        index.append(
            dict(file=name, title=title, meaning=meaning, scale=scale)
        )

    def limits(ax, points):
        low, high = points.min(0), points.max(0)
        pad = np.maximum((high - low) * 0.08, 1e-5)
        ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
        ax.set_aspect("equal", adjustable="box")

    def cloud(name, title, ids=None, points=xy, only=False, query=False):
        fig, ax = canvas()
        if not only:
            ax.scatter(
                *points.T, s=10, color=BLUE, alpha=0.55, edgecolors="none"
            )
        if ids is not None:
            ax.scatter(
                *points[ids].T, s=19, color=RED, alpha=0.86, edgecolors="none"
            )
        if query:
            ax.scatter(
                *query_xy,
                marker="*",
                s=180,
                c=GOLD,
                edgecolors="#4b3812",
                lw=0.8,
                zorder=5,
            )
        limits(ax, np.vstack([points, query_xy]) if query else points)
        basis = (
            "original normal-only PCA; identical coordinates/limits across"
            " selection policies"
            if points is xy
            else (
                "separate PCA fitted to actual 256-D projected normal"
                " descriptors; not the original PCA axes"
            )
        )
        save(
            fig,
            name,
            title,
            f"{len(points)} actual normal recordings; red = selected recording"
            " IDs; blue = all normal IDs; "
            + (
                "gold star = query index 0; query not added to bank"
                if query
                else "no query used"
            ),
            basis,
        )

    cloud("01_all1000_original_space.png", "All 1,000 normal recordings")
    cloud(
        "02_fps128_original_space_overlay.png",
        "FPS128 in original 6,144-D space",
        data["ids_fps128"],
    )
    cloud(
        "03_projected_coreset128_common_view_overlay.png",
        "Projected coreset128: same original view",
        data["ids_projected_coreset128"],
    )
    cloud(
        "04_all1000_projected_space.png",
        "Actual 256-D projection: all 1,000",
        points=projected_xy,
    )
    cloud(
        "05_projected_coreset128_projected_space_overlay.png",
        "FPS128 selected in projected 256-D space",
        data["ids_projected_coreset128"],
        points=projected_xy,
    )
    cloud(
        "06_projected_coreset128_common_view_only.png",
        "Projected coreset: selected original descriptors",
        data["ids_projected_coreset128"],
        only=True,
    )
    cloud(
        "07_fps128_original_space_only.png",
        "Original FPS128: selected descriptors",
        data["ids_fps128"],
        only=True,
    )
    for label in [
        "fps64",
        "fps256",
        "random_20260613",
        "random_20260614",
        "random_20260615",
    ]:
        cloud(
            f"08_{label}_common_view_overlay.png",
            label.replace("_", " "),
            data["ids_" + label],
        )
    first, second = set(data["ids_fps128"]), set(
        data["ids_projected_coreset128"]
    )
    fig, ax = canvas()
    ax.scatter(*xy.T, s=9, c=BLUE, alpha=0.24, edgecolors="none")
    for ids, color, marker in [
        (first & second, GREEN, "o"),
        (first - second, RED, "o"),
        (second - first, GOLD, "s"),
    ]:
        ax.scatter(
            *xy[sorted(ids)].T,
            s=22,
            c=color,
            marker=marker,
            alpha=0.9,
            edgecolors="none",
        )
    limits(ax, xy)
    save(
        fig,
        "09_selection_overlap.png",
        f"Selection overlap: {len(first & second)}/128",
        "Teal = selected by both; red = original-space FPS only; gold squares"
        " = projected-space FPS only",
        "Same original normal-only PCA and limits as 01–03",
    )

    def heat(values, name, title, meaning, cmap="RdBu_r", value_range=None):
        if value_range is None:
            vmax = float(np.max(np.abs(values)))
            value_range = (-vmax, vmax)
        fig, ax = canvas(1200, 450)
        ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=value_range[0],
            vmax=value_range[1],
        )
        save(
            fig,
            name,
            title,
            meaning,
            f"Color range {value_range}; rows top to bottom, columns left to"
            " right",
        )

    heat(
        data["projected"],
        "10_projected_embedding_matrix_1000x256.png",
        "Projected embeddings: 1,000 x 256",
        "All actual Gaussian-projected descriptor values, row = normal"
        " recording, column = projected dimension; no synthetic cells",
    )
    heat(
        data["projected"][data["ids_projected_coreset128"]],
        "11_projected_selected_matrix_128x256.png",
        "Selected projected embeddings: 128 x 256",
        "Actual projected selected rows in selection order",
        value_range=(
            -float(abs(data["projected"]).max()),
            float(abs(data["projected"]).max()),
        ),
    )
    heat(
        data["query"],
        "12_query_8_band_descriptors.png",
        "Query descriptors: 8 x 768",
        "Query index 0, actual unit descriptors. Rows = ordered bands; columns"
        " = learned coordinates, not physical frequency",
    )
    cloud(
        "13_query_and_fps128.png",
        "Query and normal reference bank",
        data["ids_fps128"],
        query=True,
    )

    pairs = data["distance_pairs"]
    fig, ax = canvas(800, 650)
    maximum = float(pairs.max()) * 1.05
    ax.plot([0, maximum], [0, maximum], color="#999999", lw=1)
    ax.scatter(*pairs.T, s=5, c=GREEN, alpha=0.2, edgecolors="none")
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_aspect("equal")
    save(
        fig,
        "14_projection_pairwise_distances.png",
        "Original vs rescaled projected distances",
        "5,000 seeded normal-normal pairs; x = actual 6,144-D Euclidean"
        " distance; y = 256-D distance times sqrt(6144/256); gray identity"
        " line",
        f"Both axes 0–{maximum:.6g}; scaling is for diagnostic display only",
    )
    fig, ax = canvas()
    policies = [
        ("fps128", RED),
        ("projected_coreset128", GREEN),
        ("random_20260613", GOLD),
    ]
    max_distance = (
        max(data["coverage_" + key].max() for key, _ in policies) * 1.04
    )
    for key, color in policies:
        ax.plot(
            np.sort(data["coverage_" + key]),
            np.arange(1, 1001) / 1000,
            c=color,
            lw=2.4,
        )
    ax.set_xlim(0, max_distance)
    ax.set_ylim(0, 1.02)
    save(
        fig,
        "15_coverage_in_original_6144d.png",
        "Coverage in original 6,144-D space",
        "CDF of nearest-selected-reference Euclidean distance over all 1,000"
        " normals; red original FPS128, teal projected coreset128, gold random"
        " seed20260613; selected normals have zero distance",
        f"x=0–{max_distance:.6g}, y=0–1.02; distances never calculated on the"
        " 2-D plot",
    )

    scale_range = (
        float(np.log10(np.maximum(data["inherited_scales"], 1e-12)).min()),
        float(np.log10(np.maximum(data["refit_scales"], 1e-12)).max()),
    )
    for key in ["inherited", "refit"]:
        heat(
            np.log10(np.maximum(data[key + "_scales"], 1e-12)),
            f"16_{key}_local_scales.png",
            key.title() + " local scales",
            "log10 k=1 self-excluded normal-normal scale; same FPS128 IDs;"
            " rows = 8 bands, columns = FPS selection order",
            "viridis",
            scale_range,
        )
    heat(
        data["raw_query_distances"],
        "17_query_raw_distances.png",
        "Query distances to FPS128",
        "Actual half-cosine distances; rows = bands, columns = same selected"
        " normal IDs",
        "magma",
        (0, float(data["raw_query_distances"].max())),
    )
    cost_limit = max(
        float(data["inherited_cost"].max()), float(data["refit_cost"].max())
    )
    for name, values, winners, title in [
        (
            "inherited",
            data["inherited_cost"],
            data["inherited_winner"],
            "Inherited scales + original winners",
        ),
        (
            "fixed_refit",
            data["refit_cost"],
            data["inherited_winner"],
            "Refitted scales + fixed winners",
        ),
        (
            "rematch",
            data["refit_cost"],
            data["rematch_winner"],
            "Refitted scales + rematched winners",
        ),
    ]:
        fig, ax = canvas(1200, 450)
        ax.imshow(
            values,
            cmap="magma",
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=cost_limit,
        )
        ax.scatter(
            winners,
            np.arange(8),
            s=45,
            facecolors="none",
            edgecolors="#00f1ef",
            lw=1.2,
        )
        save(
            fig,
            f"18_matching_{name}.png",
            title,
            "8 x 128 actual normalized query distances; cyan rings mark used"
            " winners",
            f"Common color range 0–{cost_limit:.6g}; same column order;"
            " fixed/refit and rematch cost matrices are intentionally"
            " identical",
        )
    max_score = (
        max(
            float(data[k + "_band_scores"].max())
            for k in ["inherited", "fixed", "rematch"]
        )
        * 1.08
    )
    for key, color in [
        ("inherited", BLUE),
        ("fixed", GREEN),
        ("rematch", RED),
    ]:
        fig, ax = canvas(900, 450)
        values = data[key + "_band_scores"]
        ax.bar(np.arange(8), values, color=color, width=0.65)
        ax.axhline(values.mean(), c=color, ls="--", lw=1.6)
        ax.set_xlim(-0.6, 7.6)
        ax.set_ylim(0, max_score)
        save(
            fig,
            f"19_{key}_band_scores.png",
            key.title() + " band scores",
            "Eight actual band contributions; dashed line = mean anomaly"
            " score",
            f"Shared y-range 0–{max_score:.6g}",
        )
    with (folder / "TILE_INDEX.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "title", "meaning", "scale"]
        )
        writer.writeheader()
        writer.writerows(index)
    with (folder / "SELECTED_RECORDINGS.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["policy", "selection_order", "normal_index", "recording_id"]
        )
        for key in data.files:
            if key.startswith("ids_"):
                writer.writerows(
                    (key[4:], order, int(i), data["normal_ids"][i])
                    for order, i in enumerate(data[key])
                )

    def sheet(names, target, subtitle):
        cols = 3
        rows = (len(names) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(15, rows * 4.0), squeeze=False
        )
        lookup = {r["file"]: r for r in index}
        for ax, name in zip(axes.flat, names):
            ax.imshow(Image.open(tiles / name))
            ax.set_title(lookup[name]["title"], fontsize=12, pad=7)
            ax.set_axis_off()
        for ax in list(axes.flat)[len(names) :]:
            ax.set_axis_off()
        fig.suptitle(subtitle, fontsize=14, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(folder / target, dpi=160, facecolor="white")
        plt.close(fig)

    sheet(
        [
            "01_all1000_original_space.png",
            "02_fps128_original_space_overlay.png",
            "03_projected_coreset128_common_view_overlay.png",
            "04_all1000_projected_space.png",
            "05_projected_coreset128_projected_space_overlay.png",
            "09_selection_overlap.png",
        ],
        "CONTACT_SHEET_CORESET.png",
        "Real normal descriptors | top row: one shared original-space PCA"
        " view\nBottom-left/middle: separate PCA of the actual 256-D"
        " projection; bottom-right: original view",
    )
    sheet(
        [
            "10_projected_embedding_matrix_1000x256.png",
            "11_projected_selected_matrix_128x256.png",
            "12_query_8_band_descriptors.png",
            "18_matching_inherited.png",
            "18_matching_fixed_refit.png",
            "18_matching_rematch.png",
            "19_inherited_band_scores.png",
            "19_fixed_band_scores.png",
            "19_rematch_band_scores.png",
        ],
        "CONTACT_SHEET_MATCHING.png",
        "Actual matrices and matching | individual tiles contain no text,"
        " axes, or titles",
    )
    meta.update(
        original_pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        projected_pca_explained_variance_ratio=ppca.explained_variance_ratio_.tolist(),
        tile_count=len(index),
    )
    (folder / "DISPLAY_METADATA.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    lines = [
        "# Coreset projection and selection figures — v5",
        "",
        (
            "These figures use 1,000 normal Development/fan embeddings from"
            " the public implementation without the normalized loading"
            " coefficient. Earlier v4 figures used a coefficient of 0.005;"
            " do not combine them as results from the same calculation."
        ),
        "",
        "## Main files",
        "",
        "- `CONTACT_SHEET_CORESET.png`: overview of projection and selection.",
        (
            "- `tiles/03_projected_coreset128_common_view_overlay.png`:"
            " projected coreset selection on the same coordinates as 01"
            " and 02."
        ),
        (
            "- `tiles/04_all1000_projected_space.png`: the actual 256-D"
            " projection displayed in 2-D with a separate PCA."
        ),
        (
            "- `tiles/05_projected_coreset128_projected_space_overlay.png`:"
            " 128 recordings selected in that 256-D space."
        ),
        (
            "- `tiles/06_projected_coreset128_common_view_only.png`:"
            " selected recordings on the original PCA coordinates."
        ),
        (
            "- `tiles/10_projected_embedding_matrix_1000x256.png`:"
            " the actual projected embedding matrix."
        ),
        (
            "- Individual PNGs have transparent backgrounds and no titles,"
            " axes, ticks, or borders. Their meanings and scales are"
            " documented below and in `TILE_INDEX.csv`."
        ),
        "",
        "## How to read the figures",
        "",
        (
            "1. **Selection projection:** a 6,144-to-256-dimensional"
            " Gaussian projection using the seed 20260613 and scaling of"
            " the historical local implementation. Exact greedy FPS128"
            " starts at the point nearest the centroid. This is not a"
            " reproduction of PatchCore's approximate sampler."
        ),
        (
            "2. **Display projection:** a separate 2-D PCA makes the"
            " embeddings visible. FPS was not performed on these"
            " coordinates. PCA was fitted using normal data only."
        ),
        (
            "3. Tiles 01/02/03/06/07/08/09 share original-embedding PCA"
            " coordinates and axis limits. Tiles 04/05 share separate"
            " projected-embedding PCA coordinates. Shape differences"
            " between these views alone do not establish information"
            " preservation or performance."
        ),
        (
            "4. The two 128-recording selections share"
            f' **{meta["shared_selected_ids"]} recordings**. In tile 09,'
            " teal marks shared recordings, red marks original-space"
            " FPS only, and gold squares mark projected coreset only."
        ),
        (
            "5. Projection is used **only to select recording IDs**."
            " Matching uses those recordings' original 8-by-768"
            " descriptors. The projected policy here was computed for"
            " visualization; this folder does not report its full"
            " Development/Evaluation hmean."
        ),
        (
            "6. The query has fixed index 0. It was not used to fit"
            " normal-reference selection, projection, or PCA. This is"
            " one example, not evidence of an effect across all queries."
        ),
        (
            "7. Each of the three random controls uses a fixed seed and"
            " matches the source/target counts of FPS128."
        ),
        (
            "8. The local-scale figures use the same FPS128 IDs and"
            " column order. Fixed-refit and rematch have the same cost"
            " matrix; only the selected winner differs."
        ),
        "",
        "## Diagram labels",
        "",
        (
            "`Normal descriptors → Gaussian projection (6144 → 256, for"
            " selection) → Greedy coreset selection (128 IDs) → Gather"
            " original descriptors`"
        ),
        "",
        (
            "The primary system follows `Normal descriptors → FPS128"
            " (original space) → Reference bank`. Show projected"
            " selection as a separate comparison branch."
        ),
        "",
        "## Reproduction files",
        "",
        (
            "`selection_arrays.npz` contains embeddings, the projection"
            " matrix, selected IDs, original-space distances, and matching"
            " values. `display_coordinates.npz` contains PCA coordinates"
            " and bases. `SELECTED_RECORDINGS.csv` links sample IDs to"
            " each image."
        ),
        "",
        "## Individual files",
        "",
    ]
    for row in index:
        lines.extend([f'### {row["file"]}', row["meaning"], row["scale"], ""])
    (folder / "README.md").write_text("\n".join(lines) + "\n")
    for script in [
        "prepare_reference_visuals.py",
        "render_reference_tiles.py",
    ]:
        shutil.copy2(Path(__file__).with_name(script), folder / script)
    with zipfile.ZipFile(
        folder / "DIAGRAM_TILES.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(tiles.glob("*.png")):
            archive.write(path, path.relative_to(folder))
        for name in [
            "README.md",
            "TILE_INDEX.csv",
            "SELECTED_RECORDINGS.csv",
            "CONTACT_SHEET_CORESET.png",
            "CONTACT_SHEET_MATCHING.png",
            "PROVENANCE.json",
            "DISPLAY_METADATA.json",
        ]:
            archive.write(folder / name, name)
    print(
        json.dumps(
            dict(
                tiles=len(index),
                folder=str(folder),
                overlap=meta["shared_selected_ids"],
            )
        )
    )


if __name__ == "__main__":
    main()
