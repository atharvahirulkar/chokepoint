# Chokepoint

**Graph ML for defense procurement resilience - find the vendors whose failure collapses mission coverage before disruption finds them.**

Public defense contract data contains 180,000+ awards across hundreds of vendors, agencies, and supply categories. Nobody has built a graph over it to surface structural single points of failure. Chokepoint does.

---

## What It Does

Chokepoint ingests public USAspending contract data, constructs a vendor-agency-NAICS supply graph, and scores each vendor by how much mission coverage collapses if they disappear - through disruption, adversarial compromise, or bankruptcy.

It goes beyond standard graph centrality by combining structural position with sole-source concentration and Herfindahl-Hirschman Index scoring. A vendor with moderate centrality but 100% sole-source supply of a critical NAICS category is more dangerous than a highly connected generalist. Centrality alone misses that.

---

## Architecture

```
USAspending CSV
      │
      ▼
pipeline/ingest.py        - normalize, deduplicate, clean
      │
      ▼
pipeline/build_graph.py   - NetworkX vendor-agency-NAICS graph
      │
      ▼
pipeline/features.py      - centrality, HHI, sole-source ratio per vendor
      │
      ▼
models/train.py           - IsolationForest + centrality baseline
      │
      ▼
models/eval.py            - synthetic failure injection, Recall@k
      │
      ├──▶ api/main.py         FastAPI  :8000
      └──▶ dashboard/app.py    Streamlit :8501
```

---

## Quickstart

```bash
git clone https://github.com/atharva/<your-repo>/chokepoint
cd chokepoint
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download USAspending data to data/raw/ first (see Data section below)

make all        # ingest → graph → features → train → eval
make serve      # FastAPI at localhost:8000
make dashboard  # Streamlit at localhost:8501
```

Or with Docker:

```bash
docker compose up --build
```

---

## Data

Download from [USAspending Bulk Download](https://www.usaspending.gov/download_center/award_data_archive). Select FY2023 or FY2024, Contracts. Save CSV files to `data/raw/`.

No API key required. All data is public.

---

## API

**GET /health**
```json
{ "status": "ok", "vendors_loaded": 4821, "graph_nodes": 9204 }
```

**GET /score?limit=20&sort_by=model_score**
```json
[
  {
    "vendor_name": "ACME DEFENSE SYSTEMS",
    "model_score": 0.91,
    "baseline_score": 0.74,
    "agency_count": 6,
    "naics_count": 4,
    "sole_source_ratio": 0.83
  }
]
```

**GET /stress/{vendor_name}**
```json
{
  "vendor_name": "ACME DEFENSE SYSTEMS",
  "coverage_drop": 0.47,
  "naics_affected": 3,
  "agencies_impacted": 4,
  "top_vulnerable_naics": ["336411 - Aircraft Manufacturing", "...]
}
```

**GET /explain/{vendor_name}**
```json
{
  "vendor_name": "ACME DEFENSE SYSTEMS",
  "model_score": 0.91,
  "risk_tier": "HIGH",
  "explanation_text": "Elevated risk driven by high sole-source ratio and concentrated agency dependency.",
  "feature_contributions": { "sole_source_ratio": 0.41, "hhi_score": 0.29, ... }
}
```

---

## Evaluation

No public ground truth exists for real procurement disruptions. We evaluate using synthetic failure injection:

1. Identify top 10 vendors by betweenness centrality as ground truth positives
2. Sample 10 mid-tier vendors as negatives
3. Simulate removal of each positive vendor from a graph copy
4. Measure coverage drop: fraction of NAICS codes losing all vendor coverage for at least one connected agency
5. Compute Recall@k: how many true positives appear in the model's top-k ranked vendors

| Metric | Model (IsolationForest) | Baseline (Centrality) |
|--------|------------------------|----------------------|
| Recall@5 | - | - |
| Recall@10 | - | - |
| Recall@20 | - | - |

*Populated automatically after running `make eval`. Results written to `eval_report.json`.*

---

## Limitations

These are real and worth understanding:

- **Vendor name normalization is imperfect.** USAspending uses raw vendor strings. The same company appears under multiple name variants across fiscal years. CAGE codes would solve this but are not always present in bulk exports.
- **Synthetic eval is a proxy.** Ground truth for real chokepoints does not exist publicly. Recall@k measures model ranking against centrality-derived positives, not against actual disruption events.
- **IsolationForest contamination is a hyperparameter.** The 0.05 default flags ~5% of vendors as anomalous. Domain calibration with procurement analysts would be required in production.
- **Subsidiaries and parent companies are invisible.** A vendor flagged as low-risk may be a subsidiary of a high-risk parent. The model operates on contract-level vendor strings only.
- **Static snapshot.** The graph reflects a single fiscal year. Temporal drift, new entrants, and vendor consolidation are not modeled.

---

## What Production Would Require

- SAM.gov real-time API integration for live contract ingestion
- Neo4j or similar graph database instead of in-memory NetworkX
- CAGE code as the canonical vendor identifier
- Human-in-the-loop validation with procurement analysts to establish real ground truth
- Temporal graph modeling to detect vendor consolidation trends

---

## Stack

Python, pandas, NetworkX, scikit-learn, FastAPI, Streamlit, MLflow, Plotly, Docker

---

## License

MIT

---

*Built at the SIC x DS3 x Bow Capital Defense Hackathon, UCSD, May 2026.*
