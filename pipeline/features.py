"""Compute per-vendor features over the supply graph.

Features:
  - degree_centrality
  - betweenness_centrality (approximate, k=100 samples)
  - agency_count, naics_count
  - total_award_value
  - sole_source_ratio: fraction of (agency, naics) pairs this vendor touches
    where it is the only vendor connecting that agency to that NAICS
  - hhi_score: Herfindahl-Hirschman Index of the vendor's award value
    distribution across agencies (0..1, 1 = fully concentrated)
"""
from __future__ import annotations

import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

from pipeline.critical_naics import critical_naics_id

GRAPH_PATH = Path("data/processed/supply_graph.pkl")
OUT_PATH = Path("data/processed/vendor_features.parquet")

BETWEENNESS_SAMPLES = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("features")


def load_graph() -> nx.MultiDiGraph:
    """Load the pickled supply graph."""
    if not GRAPH_PATH.exists():
        log.error("Missing %s. Run `make graph` first.", GRAPH_PATH)
        sys.exit(1)
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def vendor_nodes(g: nx.MultiDiGraph) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("node_type") == "VENDOR"]


def agency_vendors_per_naics(g: nx.MultiDiGraph) -> dict[tuple[str, str], set[str]]:
    """Map (agency_id, naics_id) -> set of vendor_ids that serve it.

    A vendor 'serves' (agency, naics) if it has an edge to both that agency
    and that NAICS. This is a join over the bipartite vendor->agency and
    vendor->naics edge sets.
    """
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


def compute_features(g: nx.MultiDiGraph) -> pd.DataFrame:
    """Compute the feature matrix indexed by vendor name."""
    log.info("Computing degree centrality...")
    # Project the MultiDiGraph down to a simple UNDIRECTED graph for centrality
    # measures. Direction is meaningless here: a vendor bridges an agency and
    # a NAICS, and that bridge is structural regardless of edge direction.
    # Parallel edges collapse to one weighted edge.
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

    deg_cent = nx.degree_centrality(simple)

    log.info(
        "Computing betweenness centrality (approx, k=%d)...", BETWEENNESS_SAMPLES
    )
    n_nodes = simple.number_of_nodes()
    k = min(BETWEENNESS_SAMPLES, n_nodes)
    btw_cent = nx.betweenness_centrality(simple, k=k, seed=42, normalized=True)

    coverage = agency_vendors_per_naics(g)
    log.info("Built (agency,naics) coverage map: %d entries", len(coverage))

    rows: list[dict] = []
    vendors = vendor_nodes(g)
    log.info("Computing per-vendor features for %d vendors", len(vendors))

    for vid in vendors:
        name = g.nodes[vid]["name"]
        agencies: dict[str, float] = {}
        naics: set[str] = set()
        critical_naics: set[str] = set()
        total_value = 0.0

        for _, target, d in g.out_edges(vid, data=True):
            w = float(d.get("weight", 0.0))
            et = d.get("edge_type")
            if et == "VENDOR_AGENCY":
                agencies[target] = agencies.get(target, 0.0) + w
                total_value += w
            elif et == "VENDOR_NAICS":
                naics.add(target)
                if critical_naics_id(target):
                    critical_naics.add(target)

        # sole-source: of all (agency, naics) pairs this vendor serves, how
        # many have only this vendor as supplier? Also track the count of
        # *critical*-NAICS pairs that are sole-sourced — this is the
        # national-security signal independent of the headline ratio.
        pair_total = 0
        pair_sole = 0
        critical_pair_sole = 0
        for a in agencies:
            for n in naics:
                holders = coverage.get((a, n))
                if not holders:
                    continue
                pair_total += 1
                if len(holders) == 1 and vid in holders:
                    pair_sole += 1
                    if critical_naics_id(n):
                        critical_pair_sole += 1
        sole_source_ratio = (pair_sole / pair_total) if pair_total else 0.0

        # HHI over agency shares of award value
        if total_value > 0 and agencies:
            shares = [val / total_value for val in agencies.values()]
            hhi = sum(s * s for s in shares)
        else:
            hhi = 0.0

        rows.append(
            {
                "vendor_id": vid,
                "vendor_name": name,
                "degree_centrality": float(deg_cent.get(vid, 0.0)),
                "betweenness_centrality": float(btw_cent.get(vid, 0.0)),
                "agency_count": len(agencies),
                "naics_count": len(naics),
                "critical_naics_count": len(critical_naics),
                "total_award_value": total_value,
                "sole_source_ratio": sole_source_ratio,
                "critical_sole_source_count": critical_pair_sole,
                "hhi_score": hhi,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    g = load_graph()
    df = compute_features(g)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s (%d vendor rows)", OUT_PATH, len(df))
    log.info(
        "Sample feature stats:\n%s",
        df[
            [
                "degree_centrality",
                "betweenness_centrality",
                "agency_count",
                "naics_count",
                "critical_naics_count",
                "total_award_value",
                "sole_source_ratio",
                "critical_sole_source_count",
                "hhi_score",
            ]
        ].describe(),
    )


if __name__ == "__main__":
    main()
