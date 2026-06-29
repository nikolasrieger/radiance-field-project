#!/usr/bin/env python
"""Download a ~10k-sequence training subset of uCO3D.

uCO3D archives are bundled *per super-category*, so there is no way to fetch an
exact number of sequences. This script reads the dataset ``metadata.sqlite`` to
get the sequence count of every super-category, greedily selects whole
super-categories (smallest-first, to maximize category diversity) until the
cumulative count reaches a target (~10k by default), and then drives the
existing :func:`download_dataset` for the selected categories.

Super-categories whose data is already present in the download folder count
toward the target and are not re-downloaded.

By default this only prints the plan (dry-run). Pass ``--run`` to actually
download.

Example:
    python -m radiance_field_project.data.dataset_download.download_10k_subset
    python -m radiance_field_project.data.dataset_download.download_10k_subset --run
"""

import json
import os
import sqlite3
from argparse import ArgumentParser
from typing import Dict, List, Set

from .download_dataset import (
    DEFAULT_CATEGORY_TO_ARCHIVES_FILE,
    DEFAULT_HF_REPO,
    DEFAULT_LINK_LIST_FILE,
)
from .download_dataset_impl import download_dataset

# Project default: the larger download folder the team is growing.
DEFAULT_DOWNLOAD_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../data/uco3d_data_big")
)
# Modalities required for training (images + gaussian splats + metadata).
DEFAULT_MODALITIES = ["metadata", "rgb_videos", "gaussian_splats"]
DEFAULT_TARGET = 10_000


def _counts_per_super_category(metadata_db: str) -> Dict[str, int]:
    if not os.path.isfile(metadata_db):
        raise FileNotFoundError(
            f"metadata.sqlite not found at {metadata_db}. "
            "Download metadata first (it ships in metadata.zip), or pass "
            "--metadata_db pointing at an existing one."
        )
    con = sqlite3.connect(metadata_db)
    try:
        rows = con.execute(
            "SELECT super_category, COUNT(*) FROM sequence_annots "
            "GROUP BY super_category"
        ).fetchall()
    finally:
        con.close()
    return {sc: n for sc, n in rows}


def _downloadable_super_categories(
    category_to_archives: dict, modalities: List[str]
) -> Set[str]:
    """Super-categories present in *every* requested non-metadata modality."""
    per_modality = []
    for m in modalities:
        if m == "metadata":
            continue
        per_modality.append(set(category_to_archives.get(m, {}).keys()))
    if not per_modality:
        return set()
    return set.intersection(*per_modality)


def _already_present(download_folder: str, candidates: Set[str]) -> Set[str]:
    """Super-categories whose extracted folder already exists on disk."""
    return {
        sc
        for sc in candidates
        if os.path.isdir(os.path.join(download_folder, sc))
    }


def _bytes_for_selection(
    category_to_archives: dict, modalities: List[str], selected: List[str]
) -> int:
    total = 0
    for m in modalities:
        if m == "metadata":
            continue
        modality_links = category_to_archives.get(m, {})
        for sc in selected:
            for archive in modality_links.get(sc, {}).values():
                total += int(archive.get("file_size_bytes", 0) or 0)
    return total


def _fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024 ** 3:.1f} GiB"


def select_super_categories(
    counts: Dict[str, int],
    downloadable: Set[str],
    present: Set[str],
    target: int,
) -> List[str]:
    """Greedily pick smallest categories first to maximize diversity.

    Already-present categories count toward the target but are never selected
    for download. Returns the categories that still need downloading.
    """
    present_total = sum(counts.get(sc, 0) for sc in present)
    remaining = target - present_total

    selectable = sorted(
        (sc for sc in downloadable if sc not in present),
        key=lambda sc: counts.get(sc, 0),
    )

    selected: List[str] = []
    running = 0
    for sc in selectable:
        if running >= remaining:
            break
        selected.append(sc)
        running += counts.get(sc, 0)
    return selected


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--download_folder", type=str, default=DEFAULT_DOWNLOAD_FOLDER)
    parser.add_argument(
        "--metadata_db",
        type=str,
        default=None,
        help="Path to metadata.sqlite (default: <download_folder>/metadata.sqlite).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Approximate number of sequences to end up with.",
    )
    parser.add_argument(
        "--modalities",
        type=lambda x: [s.strip() for s in x.split(",")],
        default=",".join(DEFAULT_MODALITIES),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Actually download. Without this flag the script only prints the plan.",
    )
    parser.add_argument("--n_download_workers", type=int, default=4)
    parser.add_argument("--n_extract_workers", type=int, default=4)
    parser.add_argument("--checksum_check", action="store_true", default=False)
    parser.add_argument(
        "--clear_archives_after_unpacking", action="store_true", default=False
    )
    parser.add_argument("--use_huggingface", action="store_true", default=False)
    parser.add_argument(
        "--category_to_archives_file",
        type=str,
        default=DEFAULT_CATEGORY_TO_ARCHIVES_FILE,
    )
    parser.add_argument("--link_list_file", type=str, default=DEFAULT_LINK_LIST_FILE)
    args = parser.parse_args()

    download_folder = os.path.abspath(args.download_folder)
    metadata_db = args.metadata_db or os.path.join(download_folder, "metadata.sqlite")
    modalities = list(args.modalities)

    with open(args.category_to_archives_file) as f:
        category_to_archives = json.load(f)

    counts = _counts_per_super_category(metadata_db)
    downloadable = _downloadable_super_categories(category_to_archives, modalities)
    present = _already_present(download_folder, downloadable)

    selected = select_super_categories(counts, downloadable, present, args.target)

    present_total = sum(counts.get(sc, 0) for sc in present)
    selected_total = sum(counts.get(sc, 0) for sc in selected)
    grand_total = present_total + selected_total
    est_bytes = _bytes_for_selection(category_to_archives, modalities, selected)

    # metadata.zip is global; skip it if metadata.sqlite is already extracted.
    download_modalities = list(modalities)
    if "metadata" in download_modalities and os.path.isfile(metadata_db):
        download_modalities.remove("metadata")
        metadata_note = (
            "  (metadata.sqlite already present -> 'metadata' modality skipped)"
        )
    else:
        metadata_note = ""

    print("=" * 72)
    print(f"uCO3D ~{args.target}-sequence subset plan")
    print("=" * 72)
    print(f"Download folder : {download_folder}")
    print(f"Modalities      : {modalities}{metadata_note}")
    print(f"Strategy        : smallest-first (max category diversity)")
    print()
    if present:
        print(f"Already present ({len(present)} cats, {present_total} seqs):")
        for sc in sorted(present, key=lambda s: counts.get(s, 0)):
            print(f"  {counts.get(sc, 0):6d}  {sc}")
        print()
    print(f"To download ({len(selected)} cats, {selected_total} seqs):")
    for sc in selected:
        print(f"  {counts.get(sc, 0):6d}  {sc}")
    print()
    print(f"Projected total : {grand_total} sequences across "
          f"{len(present) + len(selected)} categories")
    print(f"Est. archive size to download: {_fmt_gb(est_bytes)} "
          f"(extracted data uses more)")
    print("=" * 72)

    if not selected:
        print("Nothing to download: target already met by present categories.")
        return

    if not args.run:
        print("\nDRY-RUN. Re-run with --run to start the download.")
        return

    download_dataset(
        args.category_to_archives_file,
        args.link_list_file,
        download_folder,
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        download_super_categories=selected,
        download_modalities=download_modalities,
        download_small_subset=False,
        checksum_check=bool(args.checksum_check),
        clear_archives_after_unpacking=bool(args.clear_archives_after_unpacking),
        skip_downloaded_archives=True,
        use_huggingface=bool(args.use_huggingface),
        huggingface_repo=DEFAULT_HF_REPO,
    )


if __name__ == "__main__":
    main()
