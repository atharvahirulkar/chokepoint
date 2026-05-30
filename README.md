# CHOKEPOINT

**Defense Procurement Supply-Graph Intelligence.** Built solo at the SIC × DS³ × Bow Capital Defense Hackathon (May 29–31, 2026).

> Find the single vendors whose failure collapses defense mission coverage — before an adversary or disruption does.

---

## Problem

Defense procurement data is public but unanalyzed for systemic risk. Single vendors quietly sole-source critical NAICS categories across multiple sub-agencies. When they fail — disruption, adversarial compromise, bankruptcy — mission coverage collapses silently, and no public tool surfaces this concentration risk. The DoD Office of Industrial Base Policy currently does this analysis by hand in Excel.

CHOKEPOINT ingests USAspending bulk contract data, builds the vendor-agency-NAICS supply graph, and ranks vendors by a supervised model trained on simulated counterfactual removal labels. It also restricts the analysis to NAICS in the DoD Critical Technology Areas list (aerospace, missiles, naval propulsion, microelectronics, ordnance, guidance systems).

---

## Architecture

```
                  ┌────────────────────────────┐
                  │  USAspending FY2026 bulk   │
                  │  archive (4.4 GB CSV)      │
                  └──────────────┬─────────────┘
                                 │ stream-filter (DoD sub-agencies)
                                 ▼
                  ┌────────────────────────────┐
                  │  contracts.parquet         │  ~1.1M defense rows
                  └──────────────┬─────────────┘
                                 ▼
                  ┌────────────────────────────┐
                  │  supply_graph.pkl          │  17.9k vendors, 24 agencies,
                  │  (NetworkX MultiDiGraph)   │  747 NAICS, 68k edges
                  └──────────────┬─────────────┘
                                 ▼
              ┌──────────────────┴─────────────────┐
              ▼                                    ▼
   ┌─────────────────────┐               ┌─────────────────────┐
   │ vendor_features     │               │ vendor_labels       │
   │  9 graph features   │               │  coverage_drop +    │
   │  + critical-NAICS   │               │  critical-mode via  │
   │                     │               │  simulated removal  │
   └──────────┬──────────┘               └──────────┬──────────┘
              └──────────────────┬──────────────────┘
                                 ▼
              ┌────────────────────────────────────┐
              │  Train (train split only)          │
              │  - Betweenness baseline            │
              │  - IsolationForest (unsupervised)  │
              │  - GradientBoosting supervised     │
              └──────────────────┬─────────────────┘
                                 ▼
              ┌────────────────────────────────────┐
              │  Eval (held-out test split)        │
              │  Recall@k + 95% bootstrap CIs      │
              │  Spearman vs true coverage_drop    │
              │  Critical-NAICS recall@k           │
              └──────────────────┬─────────────────┘
                                 ▼
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   ┌──────────────────┐                    ┌──────────────────┐
   │ FastAPI backend  │ ◄── HTTP/JSON ──▶  │ Streamlit UI     │
   │ /score /stress   │                    │ leaderboard,     │
   │ /explain /eval   │                    │ stress simulator,│
   │ /health /metrics │                    │ explain card,    │
   └──────────────────┘                    │ eval transparency│
                                           └──────────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/atharvahirulkar/chokepoint
cd chokepoint
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Drop the USAspending bulk archive CSVs under data/raw/
# (e.g. FY2026_All_Contracts_Full_*.csv)

make prefilter        # stream-filter to defense sub-agencies → contracts.parquet
make graph            # build vendor-agency-NAICS MultiDiGraph
make features         # 9 per-vendor graph features incl. critical-NAICS
make labels           # simulate removal → coverage_drop labels (train/test split)
make train            # fit baseline + IsolationForest + supervised GB
make eval             # held-out recall@k + bootstrap CIs + critical-mode

make serve            # FastAPI on :8000
make dashboard        # Streamlit on :8501
```

Or with Docker (after running `make all` once to produce the processed artifacts):

```bash
docker compose up --build
```

Dashboard: <http://localhost:8501> · API docs: <http://localhost:8000/docs>

---

## API Reference

| Endpoint | Description |
|---|---|
| `GET /health` | vendors_loaded, graph_nodes, graph_edges |
| `GET /score?limit=20&sort_by=model_score` | Top-K vendors by `model_score`, `baseline_score`, or `iso_score` |
| `GET /stress/{vendor_name}` | Live removal simulation: coverage_drop, critical_coverage_drop, NAICS lost, top vulnerable NAICS |
| `GET /explain/{vendor_name}` | Feature contributions (z-scored value × importance), risk tier, templated rationale |
| `GET /eval` | Full eval_report.json (recall@k, CIs, top chokepoints, critical-mode tables) |
| `GET /metrics` | In-memory request counter, avg latency, stress-sim count |

Example:

```bash
curl "http://localhost:8000/stress/LOCKHEED%20MARTIN"
```

```json
{
  "vendor_name": "LOCKHEED MARTIN",
  "coverage_drop": 0.0876,
  "critical_coverage_drop": 0.010,
  "naics_affected": 16,
  "critical_naics_affected": 1,
  "agencies_impacted": 9,
  "pairs_served": 468,
  "pairs_lost": 41,
  "top_vulnerable_naics": [
    "SOFTWARE AND OTHER PRERECORDED COMPACT DISC, TAPE, AND RECORD REPRODUCING",
    "RELAY AND INDUSTRIAL CONTROL MANUFACTURING"
  ]
}
```

---

## Eval Methodology

There is no public ground truth for which defense vendors are systemic chokepoints. We generate labels via **simulated counterfactual removal**: for each vendor in a candidate pool (top 1000 by graph footprint), remove the vendor and count the fraction of `(agency, NAICS)` pairs it served that lose **all** suppliers. This is the same framework as **N-1 contingency analysis** in power-systems engineering and **DebtRank / SinkRank** for financial systemic risk.

The pool is split 75/25 (seed 42); the supervised model is trained on the train split only and evaluated on the held-out test split.

### Held-out test results (n=250 test vendors, 13 positives)

| Ranker | R@5 | R@10 (95% CI) | R@20 (95% CI) | R@50 |
|---|---:|:---:|:---:|---:|
| Betweenness baseline | 0.23 | 0.39 [0.17, 0.62] | 0.54 [0.29, 0.75] | 0.69 |
| IsolationForest | 0.31 | 0.46 [0.22, 0.67] | 0.54 [0.29, 0.75] | 0.77 |
| **Supervised GB** | **0.39** | **0.54 [0.38, 0.89]** | **0.92 [0.73, 1.00]** | **1.00** |

CIs are 1000-iter percentile bootstrap. Spearman rank correlation vs true `coverage_drop` on the test split: baseline **0.47** · iso **0.53** · model **0.66**.

### Critical-NAICS mode (n=7 critical positives in test)

Vendors are re-scored against the subset of NAICS in the **DoD Critical Technology Areas** list (aerospace, missile, naval propulsion, microelectronics, ordnance, guidance). Top critical-NAICS chokepoints in the candidate pool include **Ecology MIR**, **B & H International**, **Peck & Hale**, **Booz Allen Hamilton**, and **SAIC**.

### Feature importance (supervised model)

| Feature | Importance |
|---|---:|
| `sole_source_breadth` = sole_source_ratio × log(NAICS) | 35% |
| `critical_breadth` = sole_source_ratio × log(critical_NAICS) | 34% |
| `footprint` = log(agencies) × log(NAICS) | 13% |
| Raw centrality, log-counts, HHI | <5% each |

Three engineered interaction features carry 82% of the signal. The model is explainable, not a black box.

---

## Limitations

- **Vendor identity is fragile.** The pipeline uses `recipient_name` with corporate-suffix stripping. Subsidiaries, name changes, and DBA's are not deduplicated. A production version must key on CAGE codes or UEI.
- **Synthetic eval is a proxy.** We have no ground truth for real disruption events. The simulation captures *structural* chokepoint-ness, not realized failures. 100% recall@50 on simulated labels does not mean 100% recall on real-world failures.
- **Single fiscal year.** Built on FY2026 contracts only (May 2026 bulk archive). Multi-year analysis would smooth over award-cycle effects and reveal persistent chokepoints.
- **Bipartite graph, no agency↔NAICS edges.** Centrality measures are computed on the undirected projection because the directed bipartite structure has no shortest paths between agencies and NAICS.
- **Small positive set.** 13 test positives and 7 critical positives — recall numbers have wide bootstrap CIs. Run on multi-year data to tighten them.
- **Sub-agency rollup.** Award-line `awarding_agency_name` is almost always "DEPARTMENT OF DEFENSE"; the pipeline rolls up to `awarding_sub_agency_name` (Army, Navy, AF, DLA, ...). Sub-sub-agency (program office) structure is lost.

---

## What Production Would Require

- **CAGE/UEI as primary key**, not normalized name strings.
- **Multi-year ingestion** with award lifecycle handling (modifications, cancellations).
- **Real-time SAM.gov + USAspending API** integration instead of bulk archive downloads.
- **Graph database** (Neo4j / Memgraph) replacing pickled NetworkX.
- **Human-in-the-loop analyst feedback** — let procurement officers flag false positives and feed corrections back into training.
- **NAICS criticality** sourced from authoritative DoD/CTA mappings instead of a hand-curated list.
- **Multi-objective scoring** — combine chokepoint score with vendor financial health (D&B / Bloomberg) and adversarial-exposure signals.

---

## Layout

```
chokepoint/
├── pipeline/           # ingest, prefilter, graph, features, labels, critical_naics
├── models/             # train (baseline + iso + supervised GB), eval (recall@k + CIs)
├── api/                # FastAPI: routers/score, stress, explain + state, schemas
├── dashboard/          # Streamlit app
├── data/processed/     # parquets (gitignored)
├── models/artifacts/   # joblib (gitignored)
├── eval_report.json    # latest eval output
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── Makefile
└── requirements.txt
```

---

## License

MIT.
