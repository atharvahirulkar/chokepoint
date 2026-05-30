---
title: CHOKEPOINT
emoji: 🛰️
colorFrom: gray
colorTo: red
sdk: docker
app_port: 8501
pinned: true
short_description: Defense procurement supply-graph intelligence
license: mit
---

# CHOKEPOINT — Defense Procurement Supply-Graph Intelligence

Live dashboard for the [Chokepoint project](https://github.com/atharvahirulkar/chokepoint).

Built solo at the SIC × DS³ × Bow Capital Defense Hackathon (May 29-31, 2026).

**What it does.** Ingests 9.1M defense contract rows (FY2024-2026 USAspending bulk archives), builds a vendor-agency-NAICS supply graph (49,842 vendors, 228k edges), trains a supervised chokepoint ranker on simulated counterfactual vendor-removal labels, and serves it through a FastAPI backend + Streamlit command-center dashboard.

**Validation.** Six of ten publicly-documented defense supply-chain disruption events (Sentinel ICBM, Patriot capacity, Columbia-class submarines, F-35 mission computer, BAE Paladin, Boeing KC-46) appear in the top 1% of our model's ranking. Median percentile across all ten: 99.16%.

**On the held-out test split (FY24-26 data, 75/25 split):**
- Recall@10: **0.80** [0.43, 1.00] vs betweenness baseline 0.40
- Recall@20: **0.90** [0.71, 1.00]
- Recall@50: **1.00** (deterministic, n=1000 bootstrap)
- Spearman rank correlation vs true coverage_drop: **0.65**

See [the GitHub repo](https://github.com/atharvahirulkar/chokepoint) for source, methodology, and limitations.
