"""Build the vendor-agency-NAICS supply graph from cleaned contracts.

Node types: VENDOR, AGENCY, NAICS. Edges run VENDOR->AGENCY and VENDOR->NAICS
with `weight` (sum of award_amount) and `count` (number of contracts).
Saved as a pickled NetworkX MultiDiGraph.
"""
from __future__ import annotations

import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

CONTRACTS_PATH = Path("data/processed/contracts.parquet")
GRAPH_PATH = Path("data/processed/supply_graph.pkl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("build_graph")


def vendor_id(name: str) -> str:
    return f"V::{name}"


def agency_id(name: str) -> str:
    return f"A::{name}"


def naics_id(code: str) -> str:
    return f"N::{code}"


def build(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Aggregate contracts into a weighted supply graph.

    If a `fiscal_year` column exists, also attach per-year weight + count
    dicts to each edge so downstream temporal-persistence features can be
    computed without re-reading the parquet.
    """
    g = nx.MultiDiGraph()
    has_fy = "fiscal_year" in df.columns

    group_cols_va = ["recipient_name", "awarding_agency_name"]
    group_cols_vn = ["recipient_name", "naics_code"]

    # Aggregate per (vendor, agency)
    va = (
        df.groupby(group_cols_va, dropna=False)
        .agg(weight=("award_amount", "sum"), count=("award_amount", "size"))
        .reset_index()
    )
    vn = (
        df.groupby(group_cols_vn, dropna=False)
        .agg(weight=("award_amount", "sum"), count=("award_amount", "size"))
        .reset_index()
    )

    # Per-year breakdowns
    va_by_year: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    vn_by_year: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    if has_fy:
        fy_va = (
            df.groupby(group_cols_va + ["fiscal_year"], dropna=False)
            .agg(weight=("award_amount", "sum"), count=("award_amount", "size"))
            .reset_index()
        )
        for row in fy_va.itertuples(index=False):
            key = (row.recipient_name, row.awarding_agency_name)
            va_by_year.setdefault(key, {})[int(row.fiscal_year)] = {
                "weight": float(row.weight),
                "count": int(row.count),
            }
        fy_vn = (
            df.groupby(group_cols_vn + ["fiscal_year"], dropna=False)
            .agg(weight=("award_amount", "sum"), count=("award_amount", "size"))
            .reset_index()
        )
        for row in fy_vn.itertuples(index=False):
            key = (row.recipient_name, row.naics_code)
            vn_by_year.setdefault(key, {})[int(row.fiscal_year)] = {
                "weight": float(row.weight),
                "count": int(row.count),
            }

    naics_desc = (
        df.dropna(subset=["naics_code"])
        .groupby("naics_code")["naics_description"]
        .agg(lambda s: s.dropna().iloc[0] if s.dropna().size else "")
        .to_dict()
    )

    for v in df["recipient_name"].dropna().unique():
        g.add_node(vendor_id(v), node_type="VENDOR", name=v)
    for a in df["awarding_agency_name"].dropna().unique():
        g.add_node(agency_id(a), node_type="AGENCY", name=a)
    for n in df["naics_code"].dropna().unique():
        g.add_node(
            naics_id(n),
            node_type="NAICS",
            code=n,
            description=naics_desc.get(n, ""),
        )

    for row in va.itertuples(index=False):
        attrs = {
            "edge_type": "VENDOR_AGENCY",
            "weight": float(row.weight),
            "count": int(row.count),
        }
        if has_fy:
            attrs["by_year"] = va_by_year.get(
                (row.recipient_name, row.awarding_agency_name), {}
            )
        g.add_edge(
            vendor_id(row.recipient_name),
            agency_id(row.awarding_agency_name),
            **attrs,
        )
    for row in vn.itertuples(index=False):
        attrs = {
            "edge_type": "VENDOR_NAICS",
            "weight": float(row.weight),
            "count": int(row.count),
        }
        if has_fy:
            attrs["by_year"] = vn_by_year.get(
                (row.recipient_name, row.naics_code), {}
            )
        g.add_edge(
            vendor_id(row.recipient_name),
            naics_id(row.naics_code),
            **attrs,
        )

    if has_fy:
        years = sorted(int(y) for y in df["fiscal_year"].dropna().unique())
        g.graph["fiscal_years"] = years
        log.info("Graph annotated with fiscal years: %s", years)

    return g


def top_vendors_by_degree(g: nx.MultiDiGraph, k: int = 10) -> list[tuple[str, int]]:
    """Return the top-k VENDOR nodes by out-degree."""
    degs = [
        (g.nodes[n]["name"], g.out_degree(n))
        for n in g.nodes
        if g.nodes[n].get("node_type") == "VENDOR"
    ]
    degs.sort(key=lambda x: x[1], reverse=True)
    return degs[:k]


def main() -> None:
    if not CONTRACTS_PATH.exists():
        log.error("Missing %s. Run `make ingest` first.", CONTRACTS_PATH)
        sys.exit(1)

    df = pd.read_parquet(CONTRACTS_PATH)
    log.info("Loaded %d contract rows", len(df))

    g = build(df)
    type_counts = Counter(d.get("node_type") for _, d in g.nodes(data=True))
    log.info("Graph nodes=%d edges=%d", g.number_of_nodes(), g.number_of_edges())
    log.info("Node type counts: %s", dict(type_counts))

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(g, f)
    log.info("Wrote %s", GRAPH_PATH)

    log.info("Top 10 vendors by out-degree:")
    for name, deg in top_vendors_by_degree(g, 10):
        log.info("  %s -> %d", name, deg)


if __name__ == "__main__":
    main()
