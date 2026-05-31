# CHOKEPOINT

**Defense procurement supply-graph intelligence.** Built at the SIC x DS3 x Bow Capital Defense Hackathon (May 29 to 31, 2026).

> Find the single vendors whose failure would collapse defense mission coverage, before an adversary or a disruption finds them first.

**Live demo:** https://atharvahirulkar-chokepoint.hf.space/

**Source:** https://github.com/atharvahirulkar/chokepoint

---

## The problem in one paragraph

Defense procurement data is public, but nobody analyzes it for systemic concentration risk. A small number of vendors quietly sole-source critical categories across multiple sub-agencies. When one of them fails, through disruption, adversarial compromise, or bankruptcy, mission coverage collapses silently and no public tool surfaces it in advance. A 10,000 dollar drone can be grounded by a 50 dollar component that has exactly one supplier. The DoD Office of Industrial Base Policy tracks this kind of risk today largely by hand in spreadsheets. I built the prototype of the tool that should replace that spreadsheet.

---

## What it does

I ingest three fiscal years of USAspending defense contract awards, build a vendor-to-agency-to-NAICS supply graph, and rank every vendor by a supervised model that learns from simulated vendor-failure outcomes. The system runs as a FastAPI backend and a Streamlit command-center dashboard, both containerized and deployed.

The headline result: on a held-out test set, the supervised model puts **80 percent of true chokepoints in its top 10**, double the betweenness-centrality baseline, and **6 of 10 publicly documented real-world disruption events land in the top 1 percent** of its 49,842-vendor ranking.

---

## Architecture

![Chokepoint architecture](docs/assets/architecture.svg)

The flow is data to demo in six stages: stream-filter the raw archives, build the graph, engineer features and simulated labels, train three rankers, evaluate on a held-out split plus real events, and serve through an API and dashboard.

---

## Results

I evaluate on a held-out test split (250 vendors, 10 positives) that the supervised model never sees during training. Confidence intervals are 1000-iteration percentile bootstraps.

| Ranker | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---|---|---|---|
| Betweenness baseline | 0.20 | 0.40 | 0.40 | 0.40 |
| IsolationForest (unsupervised) | 0.10 | 0.20 | 0.50 | 0.80 |
| **GradientBoosting (supervised)** | **0.40** | **0.80** | **0.90** | **1.00** |

The supervised model wins at every cutoff and beats the centrality baseline by 2x at recall@10 and recall@20. Spearman rank correlation against true simulated coverage drop on the test split: baseline 0.47, IsolationForest 0.50, supervised model **0.65**.

### Validation against real events

There is no public ground truth for real chokepoints, so I also tested the model against ten publicly documented defense supply-chain disruption events from 2022 to 2024 (Sentinel ICBM cost breach, Patriot interceptor capacity, Columbia-class submarines, Boeing KC-46, F-35 mission computer, BAE Paladin, and others).

- **6 of 10** events rank in the **top 1 percent** of the 49,842-vendor model ranking
- **8 of 10** rank in the top 10 percent
- **Median percentile: 99.16**

The two misses (Aerojet Rocketdyne and TransDigm) fail for the same documented reason: their concentration is split across subsidiaries and acquired-parent rollups, which is exactly the vendor-identity limitation my production roadmap calls out. Aerojet was acquired by L3Harris in 2023, and L3Harris itself ranks in the top 1 percent.

---

## How the evaluation works

The core methodology is what I am most proud of, because it solves the absence of ground truth honestly.

I generate labels by **simulated counterfactual removal**, which is N-1 contingency analysis borrowed from power-systems engineering and closely related to DebtRank and SinkRank in financial systemic-risk literature. For every vendor in a candidate pool (the top 1000 by graph footprint), I remove the vendor from a copy of the graph and count the fraction of the (agency, NAICS) pairs it served that drop to zero remaining suppliers. That fraction is the `coverage_drop` label.

I split the pool 75/25 with a fixed seed, train the GradientBoosting ranker on the train portion only, and report every metric on the held-out test portion. The betweenness baseline and the IsolationForest are scored on the same held-out vendors so the comparison is fair.

### Feature importance

| Feature | Importance |
|---|---|
| `sole_source_breadth` = sole_source_ratio x log(NAICS count) | 33 percent |
| `critical_breadth` = sole_source_ratio x log(critical-NAICS count) | 31 percent |
| `footprint` = log(agencies) x log(NAICS count) | 19 percent |
| Raw centrality, log counts, HHI, temporal | under 5 percent each |

Three engineered interaction features carry roughly 80 percent of the signal, so the model is interpretable rather than a black box. I also engineered temporal persistence features (years active, year-over-year growth, emerging-concentration flags). They added no predictive lift over the structural signals, which I report as an honest null result and keep in the system as descriptive analyst-facing labels.

---

## What I built, by skill area

I designed and implemented every layer of this system solo. Mapping it to the things a data science or ML role cares about:

**End-to-end ML engineering.** I built the full path from 72 GB of raw public CSVs to a deployed product: a streaming chunked ingestion that never loads a full file into memory, a NetworkX graph builder, a feature pipeline, a model trainer with MLflow tracking, a FastAPI service that preindexes the graph for sub-millisecond stress simulations, a Streamlit dashboard, Docker images for both services, and a public deployment on Hugging Face Spaces.

**Rigorous evaluation.** I treated the lack of ground truth as the central research problem, not an afterthought. I designed a counterfactual labeling scheme, enforced a held-out train/test split, quantified uncertainty with bootstrap confidence intervals, and benchmarked against both an unsupervised and a no-ML baseline. When my first evaluation design turned out to be circular (positives defined by the same metric as the baseline), I caught it and rebuilt the ground truth around simulated coverage drop.

**Applied graph and supervised ML.** I computed graph-structural features (degree, betweenness, PageRank, eigenvector centrality, articulation points) on a 50,000-node bipartite graph, engineered domain interaction features, and trained a GradientBoosting ranker that meaningfully outperforms classical centrality.

**Domain framing and product sense.** I aligned the analysis to the DoD Critical Technology Areas list, the DPA Title III investment categories, and NDAA 2022 Section 855, then validated against real reported disruptions. The dashboard is built for an analyst persona, not just a notebook reader.

---

## Quickstart

```bash
git clone https://github.com/atharvahirulkar/chokepoint
cd chokepoint
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Drop USAspending bulk archive CSVs under data/raw/FY{year}_All_Contracts_Full_*/
make prefilter   # stream + filter to defense sub-agencies, FY-tagged
make graph       # build the vendor-agency-NAICS MultiDiGraph
make features    # 22 per-vendor features
make labels      # simulated removal coverage_drop labels, train/test split
make train       # baseline + IsolationForest + supervised GradientBoosting
make eval        # held-out recall@k with bootstrap CIs
make events      # validate against documented real-world disruptions

make serve       # FastAPI on :8000
make dashboard   # Streamlit on :8501
```

Or run the whole thing with Docker after generating the processed artifacts once:

```bash
docker compose up --build
```

---

## API reference

| Endpoint | Description |
|---|---|
| `GET /health` | vendors loaded, graph node and edge counts |
| `GET /score?limit=20&sort_by=model_score` | top-K vendors by model, baseline, or IsolationForest score |
| `GET /stress/{vendor_name}` | live removal simulation: coverage drop, critical-NAICS drop, vulnerable NAICS |
| `GET /explain/{vendor_name}` | feature contributions, risk tier, templated rationale |
| `GET /events` | predictions-vs-reality validation table |
| `GET /heatmap/critical` | sub-agency by critical-NAICS supplier-count matrix |
| `GET /eval` | full evaluation report |
| `GET /metrics` | request counter, average latency, stress-sim count |

Example:

```bash
curl "http://localhost:8000/stress/RAYTHEON"
```

```json
{
  "vendor_name": "RAYTHEON",
  "coverage_drop": 0.043,
  "critical_coverage_drop": 0.071,
  "naics_affected": 32,
  "critical_naics_affected": 6,
  "agencies_impacted": 10,
  "pairs_served": 480,
  "pairs_lost": 73,
  "top_vulnerable_naics": ["SPACE RESEARCH AND TECHNOLOGY", "..."]
}
```

---

## Limitations

I would rather state these plainly than hide them.

- **Vendor identity is name-string based.** I normalize `recipient_name` and strip corporate suffixes, but subsidiaries, name changes, and acquired parents are not unified. This is why Aerojet Rocketdyne and TransDigm rank low despite real concentration. Production needs CAGE or UEI parent rollup.
- **Synthetic eval is a proxy.** Simulated coverage drop captures structural chokepoint risk, not realized failures. Perfect recall on simulated labels does not mean perfect recall on the real world.
- **Small positive set.** With 10 test positives, the recall numbers carry wide bootstrap intervals. More fiscal years would tighten them.
- **Bipartite projection.** Centrality is computed on the undirected projection because the directed bipartite structure has no shortest paths between agencies and NAICS.
- **Sub-agency rollup.** The award line agency is almost always "Department of Defense," so I roll up to the sub-agency (Army, Navy, Air Force, DLA). Program-office structure below that is lost.

---

## What production would require

- CAGE or UEI as the vendor primary key instead of name strings
- Real-time SAM.gov and USAspending API ingestion instead of bulk archives
- A graph database (Neo4j or Memgraph) in place of an in-memory pickle
- A human-in-the-loop feedback channel so procurement analysts can correct false positives
- NAICS criticality sourced from an authoritative DoD mapping rather than my curated list

---

## Repository layout

```
chokepoint/
  pipeline/      ingest, prefilter, build_graph, features, labels, critical_naics, event_validation
  models/        train (baseline + IsolationForest + supervised GB), eval (recall@k + bootstrap CIs)
  api/           FastAPI: routers for score, stress, explain + state, schemas, main
  dashboard/     Streamlit command-center app
  deploy/        Hugging Face Space bundle (combined image, deploy script)
  docs/assets/   architecture diagram
  data/          processed parquet and synthetic eval artifacts
  Dockerfile.api, Dockerfile.dashboard, docker-compose.yml, Makefile, requirements.txt
```



## Author

**Atharva Hirulkar** - MS Data Science, UC San Diego  
[GitHub](https://github.com/atharvahirulkar) · [LinkedIn](https://linkedin.com/in/atharva-hirulkar)
