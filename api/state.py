"""Process-local state loaded once at startup.

Loading scores, graph, eval report, and the trained models eagerly avoids
per-request I/O. The graph and scores are read-only after load.
"""
from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import pandas as pd

GRAPH_PATH = Path("data/processed/supply_graph.pkl")
SCORES_PATH = Path("data/processed/vendor_scores.parquet")
LABELS_PATH = Path("data/processed/vendor_labels.parquet")
EVAL_PATH = Path("eval_report.json")
GB_PATH = Path("models/artifacts/supervised.joblib")
SCALER_PATH = Path("models/artifacts/scaler.joblib")

log = logging.getLogger("chokepoint.state")


class AppState:
    """Container for all preloaded artifacts."""

    def __init__(self) -> None:
        self.scores: pd.DataFrame
        self.labels: pd.DataFrame | None = None
        self.graph: nx.MultiDiGraph
        self.eval_report: dict[str, Any]
        self.supervised_model: Any | None = None
        self.scaler: Any | None = None
        self.pair_coverage: dict[tuple[str, str], set[str]] = {}
        self.vendor_id_by_name: dict[str, str] = {}
        # Pre-indexed: vendor_id -> set of (agency_id, naics_id) pairs served
        self.served_pairs_by_vendor: dict[str, set[tuple[str, str]]] = {}

    def load(self) -> None:
        log.info("Loading scores from %s", SCORES_PATH)
        self.scores = pd.read_parquet(SCORES_PATH)
        self.vendor_id_by_name = dict(
            zip(self.scores["vendor_name"], self.scores["vendor_id"])
        )

        if LABELS_PATH.exists():
            self.labels = pd.read_parquet(LABELS_PATH)

        log.info("Loading graph from %s", GRAPH_PATH)
        with open(GRAPH_PATH, "rb") as f:
            self.graph = pickle.load(f)

        log.info("Loading eval report from %s", EVAL_PATH)
        if EVAL_PATH.exists():
            self.eval_report = json.loads(EVAL_PATH.read_text())
        else:
            self.eval_report = {}

        if GB_PATH.exists():
            log.info("Loading trained model from %s", GB_PATH)
            self.supervised_model = joblib.load(GB_PATH)
        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)

        log.info("Indexing pair coverage")
        vendor_agencies: dict[str, set[str]] = defaultdict(set)
        vendor_naics: dict[str, set[str]] = defaultdict(set)
        for u, v, d in self.graph.edges(data=True):
            et = d.get("edge_type")
            if et == "VENDOR_AGENCY":
                vendor_agencies[u].add(v)
            elif et == "VENDOR_NAICS":
                vendor_naics[u].add(v)

        pair_cov: dict[tuple[str, str], set[str]] = defaultdict(set)
        served_pairs: dict[str, set[tuple[str, str]]] = {}
        for vid in vendor_agencies:
            agencies = vendor_agencies[vid]
            naics = vendor_naics.get(vid, set())
            pairs = {(a, n) for a in agencies for n in naics}
            served_pairs[vid] = pairs
            for pair in pairs:
                pair_cov[pair].add(vid)
        self.pair_coverage = dict(pair_cov)
        self.served_pairs_by_vendor = served_pairs

        log.info(
            "State loaded: %d vendors  %d graph nodes  %d edges  %d pairs",
            len(self.scores),
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self.pair_coverage),
        )

    def lookup_vendor_id(self, name: str) -> str | None:
        """Resolve a vendor name to an id with simple normalization."""
        if name in self.vendor_id_by_name:
            return self.vendor_id_by_name[name]
        upper = name.strip().upper()
        if upper in self.vendor_id_by_name:
            return self.vendor_id_by_name[upper]
        # last-resort fuzzy contains match on upper-cased names
        for vname, vid in self.vendor_id_by_name.items():
            if upper in vname:
                return vid
        return None


state = AppState()
