"""Generate supervised chokepoint labels via vendor-removal simulation.

For each vendor in a candidate pool (top by footprint), simulate removal
and compute `coverage_drop` = fraction of (agency, naics) pairs the vendor
served that go to zero suppliers after removal.

The pool is partitioned into a deterministic train/test split. Only the
train split is used to fit the supervised model; the test split feeds the
eval harness. Everything outside the pool gets `split = "none"` and a
zero label (coverage_drop is effectively zero for tiny vendors anyway).

Output: data/processed/vendor_labels.parquet
"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.critical_naics import critical_naics_id

GRAPH_PATH = Path("data/processed/supply_graph.pkl")
FEATURES_PATH = Path("data/processed/vendor_features.parquet")
LABELS_PATH = Path("data/processed/vendor_labels.parquet")

CANDIDATE_POOL_SIZE = 1000
TEST_FRACTION = 0.25
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("labels")


def build_pair_coverage(g) -> dict[tuple[str, str], set[str]]:
    vendor_agencies: dict[str, set[str]] = defaultdict(set)
    vendor_naics: dict[str, set[str]] = defaultdict(set)
    for u, v, d in g.edges(data=True):
        et = d.get("edge_type")
        if et == "VENDOR_AGENCY":
            vendor_agencies[u].add(v)
        elif et == "VENDOR_NAICS":
            vendor_naics[u].add(v)
    cov: dict[tuple[str, str], set[str]] = defaultdict(set)
    for vid, agencies in vendor_agencies.items():
        for a in agencies:
            for n in vendor_naics.get(vid, ()):
                cov[(a, n)].add(vid)
    return cov


def simulate(
    base_cov: dict[tuple[str, str], set[str]], vendor_id: str
) -> tuple[float, int, float, int]:
    """Return (coverage_drop, naics_affected, critical_coverage_drop,
    critical_naics_affected).

    `critical_*` versions are computed over the subset of pairs whose NAICS
    is in the DoD critical list. critical_coverage_drop = fraction of
    *critical* pairs served by this vendor that go to zero suppliers on
    removal.
    """
    served = [pair for pair, holders in base_cov.items() if vendor_id in holders]
    if not served:
        return 0.0, 0, 0.0, 0
    lost = [pair for pair in served if base_cov[pair] == {vendor_id}]
    coverage_drop = len(lost) / len(served)
    naics_affected = len({pair[1] for pair in lost})

    served_crit = [pair for pair in served if critical_naics_id(pair[1])]
    if served_crit:
        lost_crit = [pair for pair in served_crit if base_cov[pair] == {vendor_id}]
        critical_drop = len(lost_crit) / len(served_crit)
        critical_naics_affected = len({pair[1] for pair in lost_crit})
    else:
        critical_drop = 0.0
        critical_naics_affected = 0

    return coverage_drop, naics_affected, critical_drop, critical_naics_affected


def main() -> None:
    if not GRAPH_PATH.exists() or not FEATURES_PATH.exists():
        raise SystemExit("Run `make graph && make features` first.")

    log.info("Loading graph + features")
    with open(GRAPH_PATH, "rb") as f:
        g = pickle.load(f)
    feats = pd.read_parquet(FEATURES_PATH)

    log.info("Building (agency, naics) coverage map")
    cov = build_pair_coverage(g)
    log.info("  %d pairs", len(cov))

    footprint = feats["agency_count"] + feats["naics_count"]
    pool_ids = (
        feats.assign(_f=footprint)
        .sort_values("_f", ascending=False)
        .head(CANDIDATE_POOL_SIZE)["vendor_id"]
        .tolist()
    )
    log.info("Candidate pool: top %d by footprint", len(pool_ids))

    rows = []
    for vid in pool_ids:
        drop, naics_n, crit_drop, crit_naics_n = simulate(cov, vid)
        rows.append(
            {
                "vendor_id": vid,
                "coverage_drop": drop,
                "naics_affected_count": naics_n,
                "critical_coverage_drop": crit_drop,
                "critical_naics_affected_count": crit_naics_n,
            }
        )
    pool_df = pd.DataFrame(rows)

    train_ids, test_ids = train_test_split(
        pool_df["vendor_id"].tolist(),
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
    )
    split_map = {vid: "train" for vid in train_ids}
    split_map.update({vid: "test" for vid in test_ids})
    pool_df["split"] = pool_df["vendor_id"].map(split_map)

    # Merge with all vendors; non-pool vendors get split=none and 0 labels.
    all_df = feats[["vendor_id"]].merge(pool_df, on="vendor_id", how="left")
    all_df["split"] = all_df["split"].fillna("none")
    all_df["coverage_drop"] = all_df["coverage_drop"].fillna(0.0).astype(float)
    all_df["naics_affected_count"] = (
        all_df["naics_affected_count"].fillna(0).astype(int)
    )
    all_df["critical_coverage_drop"] = (
        all_df["critical_coverage_drop"].fillna(0.0).astype(float)
    )
    all_df["critical_naics_affected_count"] = (
        all_df["critical_naics_affected_count"].fillna(0).astype(int)
    )

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(LABELS_PATH, index=False)
    log.info("Wrote %s (%d rows)", LABELS_PATH, len(all_df))

    log.info(
        "Pool label stats: mean_drop=%.4f  >0_count=%d  "
        "critical>0_count=%d  train=%d test=%d",
        float(pool_df["coverage_drop"].mean()),
        int((pool_df["coverage_drop"] > 0).sum()),
        int((pool_df["critical_coverage_drop"] > 0).sum()),
        sum(s == "train" for s in pool_df["split"]),
        sum(s == "test" for s in pool_df["split"]),
    )

    top = pool_df.sort_values("coverage_drop", ascending=False).head(10)
    top = top.merge(feats[["vendor_id", "vendor_name"]], on="vendor_id")
    log.info(
        "Top 10 by coverage_drop:\n%s",
        top[["vendor_name", "coverage_drop", "naics_affected_count", "split"]].to_string(
            index=False
        ),
    )


if __name__ == "__main__":
    main()
