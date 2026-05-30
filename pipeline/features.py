"""Compute per-vendor features over the supply graph.

Raw features (in vendor_features.parquet):
  Graph-structural (undirected projection):
    - degree_centrality
    - betweenness_centrality (approximate, k=100)
    - pagerank
    - eigenvector_centrality
    - articulation_point (binary: 1 if removing this vendor disconnects a
      component)
  Counts:
    - agency_count, naics_count, critical_naics_count
  Contract volume / size:
    - contract_count, total_award_value, avg_award_size
  Concentration (vendor-side):
    - sole_source_ratio (fraction of (agency, naics) pairs sole-sourced)
    - critical_sole_source_count (pairs sole-sourced inside critical NAICS)
    - mean_pair_redundancy (mean |competing vendors| across served pairs)
    - hhi_score (HHI of award-value share across agencies)
    - naics_hhi (HHI of award-value share across NAICS)
  Market footprint:
    - critical_naics_market_share (vendor share of total critical-NAICS spend)
"""
from __future__ import annotations

import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from pipeline.critical_naics import critical_naics_id

GRAPH_PATH = Path("data/processed/supply_graph.pkl")
OUT_PATH = Path("data/processed/vendor_features.parquet")

BETWEENNESS_SAMPLES = 100
EIGENVECTOR_MAX_ITER = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("features")


def load_graph() -> nx.MultiDiGraph:
    if not GRAPH_PATH.exists():
        log.error("Missing %s. Run `make graph` first.", GRAPH_PATH)
        sys.exit(1)
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def vendor_nodes(g: nx.MultiDiGraph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("node_type") == "VENDOR"]


def agency_vendors_per_naics(g: nx.MultiDiGraph) -> dict[tuple[str, str], set[str]]:
    """Map (agency_id, naics_id) -> set of vendor_ids serving that pair."""
    vendor_agencies: dict[str, set[str]] = defaultdict(set)
    vendor_naics: dict[str, set[str]] = defaultdict(set)
    for u, v, d in g.edges(data=True):
        et = d.get("edge_type")
        if et == "VENDOR_AGENCY":
            vendor_agencies[u].add(v)
        elif et == "VENDOR_NAICS":
            vendor_naics[u].add(v)
    coverage: dict[tuple[str, str], set[str]] = defaultdict(set)
    for vend in vendor_agencies:
        for a in vendor_agencies[vend]:
            for n in vendor_naics.get(vend, ()):
                coverage[(a, n)].add(vend)
    return coverage


def naics_total_spend(g: nx.MultiDiGraph) -> dict[str, float]:
    """Total dollars flowing through each NAICS node (sum of VENDOR_NAICS weights)."""
    spend: dict[str, float] = defaultdict(float)
    for _, target, d in g.edges(data=True):
        if d.get("edge_type") == "VENDOR_NAICS":
            spend[target] += float(d.get("weight", 0.0))
    return spend


def build_simple_undirected(g: nx.MultiDiGraph) -> nx.Graph:
    """Collapse the MultiDiGraph into a simple weighted undirected graph.

    Direction is meaningless for structural centrality here — a vendor
    bridges an agency and a NAICS, the bridge is structural regardless of
    which way the edge points. Parallel edges sum to a single weight.
    """
    simple = nx.Graph()
    for u, v, d in g.edges(data=True):
        w = float(d.get("weight", 0.0))
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += w
        else:
            simple.add_edge(u, v, weight=w)
    for n, d in g.nodes(data=True):
        if n not in simple:
            simple.add_node(n, **d)
        else:
            simple.nodes[n].update(d)
    return simple


def compute_features(g: nx.MultiDiGraph) -> pd.DataFrame:
    simple = build_simple_undirected(g)

    log.info("Degree centrality")
    deg_cent = nx.degree_centrality(simple)

    log.info("Betweenness centrality (approx, k=%d)", BETWEENNESS_SAMPLES)
    k = min(BETWEENNESS_SAMPLES, simple.number_of_nodes())
    btw_cent = nx.betweenness_centrality(simple, k=k, seed=42, normalized=True)

    log.info("PageRank")
    pr = nx.pagerank(simple, alpha=0.85, weight="weight", max_iter=200)

    log.info("Eigenvector centrality")
    try:
        eig = nx.eigenvector_centrality_numpy(simple, weight="weight")
    except Exception as e:
        log.warning("eigenvector_centrality_numpy failed: %s — falling back to power iter", e)
        eig = nx.eigenvector_centrality(
            simple, max_iter=EIGENVECTOR_MAX_ITER, weight="weight", tol=1e-6
        )

    log.info("Articulation points")
    artic = set(nx.articulation_points(simple))
    log.info("  %d articulation points across full graph", len(artic))

    coverage = agency_vendors_per_naics(g)
    log.info("(agency, naics) coverage map: %d entries", len(coverage))

    naics_spend = naics_total_spend(g)
    critical_total_spend = sum(
        v for nid, v in naics_spend.items() if critical_naics_id(nid)
    )
    log.info("Total critical-NAICS spend: $%.0f", critical_total_spend)

    rows: list[dict] = []
    vendors = vendor_nodes(g)
    log.info("Per-vendor features for %d vendors", len(vendors))

    for vid in vendors:
        name = g.nodes[vid]["name"]
        agencies_val: dict[str, float] = {}
        agencies_count: dict[str, int] = {}
        naics_val: dict[str, float] = {}
        critical_naics_ids: set[str] = set()
        total_value = 0.0
        contract_count = 0

        for _, target, d in g.out_edges(vid, data=True):
            w = float(d.get("weight", 0.0))
            c = int(d.get("count", 0))
            et = d.get("edge_type")
            if et == "VENDOR_AGENCY":
                agencies_val[target] = agencies_val.get(target, 0.0) + w
                agencies_count[target] = agencies_count.get(target, 0) + c
                total_value += w
                contract_count += c
            elif et == "VENDOR_NAICS":
                naics_val[target] = naics_val.get(target, 0.0) + w
                if critical_naics_id(target):
                    critical_naics_ids.add(target)

        # Pair-level concentration features
        pair_total = 0
        pair_sole = 0
        critical_pair_sole = 0
        redundancy_sum = 0.0
        for a in agencies_val:
            for n in naics_val:
                holders = coverage.get((a, n))
                if not holders:
                    continue
                pair_total += 1
                redundancy_sum += len(holders)
                if len(holders) == 1 and vid in holders:
                    pair_sole += 1
                    if critical_naics_id(n):
                        critical_pair_sole += 1
        sole_source_ratio = (pair_sole / pair_total) if pair_total else 0.0
        mean_pair_redundancy = (redundancy_sum / pair_total) if pair_total else 0.0

        # Agency-side HHI (value distribution)
        if total_value > 0 and agencies_val:
            shares_a = [v / total_value for v in agencies_val.values()]
            hhi_agency = sum(s * s for s in shares_a)
        else:
            hhi_agency = 0.0

        # NAICS-side HHI
        naics_total = sum(naics_val.values())
        if naics_total > 0 and naics_val:
            shares_n = [v / naics_total for v in naics_val.values()]
            naics_hhi = sum(s * s for s in shares_n)
        else:
            naics_hhi = 0.0

        # Critical NAICS market share
        critical_value = sum(
            v for nid, v in naics_val.items() if critical_naics_id(nid)
        )
        critical_market_share = (
            critical_value / critical_total_spend if critical_total_spend > 0 else 0.0
        )

        avg_award = (total_value / contract_count) if contract_count > 0 else 0.0

        rows.append(
            {
                "vendor_id": vid,
                "vendor_name": name,
                "degree_centrality": float(deg_cent.get(vid, 0.0)),
                "betweenness_centrality": float(btw_cent.get(vid, 0.0)),
                "pagerank": float(pr.get(vid, 0.0)),
                "eigenvector_centrality": float(eig.get(vid, 0.0)),
                "articulation_point": int(vid in artic),
                "agency_count": len(agencies_val),
                "naics_count": len(naics_val),
                "critical_naics_count": len(critical_naics_ids),
                "contract_count": contract_count,
                "total_award_value": total_value,
                "avg_award_size": avg_award,
                "sole_source_ratio": sole_source_ratio,
                "critical_sole_source_count": critical_pair_sole,
                "mean_pair_redundancy": mean_pair_redundancy,
                "hhi_score": hhi_agency,
                "naics_hhi": naics_hhi,
                "critical_naics_market_share": critical_market_share,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    g = load_graph()
    df = compute_features(g)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s (%d rows)", OUT_PATH, len(df))

    summary_cols = [
        "degree_centrality",
        "betweenness_centrality",
        "pagerank",
        "eigenvector_centrality",
        "articulation_point",
        "agency_count",
        "naics_count",
        "critical_naics_count",
        "contract_count",
        "total_award_value",
        "avg_award_size",
        "sole_source_ratio",
        "critical_sole_source_count",
        "mean_pair_redundancy",
        "hhi_score",
        "naics_hhi",
        "critical_naics_market_share",
    ]
    log.info("Feature stats:\n%s", df[summary_cols].describe().T)


if __name__ == "__main__":
    main()
