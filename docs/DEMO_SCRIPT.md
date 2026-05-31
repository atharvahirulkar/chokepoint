# CHOKEPOINT: Demo Script and Q&A Prep

Everything I need to present and defend the project in two minutes plus questions.

Live demo: https://atharvahirulkar-chokepoint.hf.space/

---

## Numbers to know cold

| Metric | Value |
|---|---|
| Vendors in graph | 49,842 |
| Graph edges | 228,336 |
| Defense contract rows ingested | 9.1 million (FY2024 to FY2026) |
| Sub-agencies | 25 |
| NAICS codes | 954 |
| Model recall@10 (held-out) | 0.80 |
| Baseline recall@10 | 0.40 |
| Model recall@20 | 0.90 |
| Model recall@50 | 1.00 |
| Spearman vs true coverage drop | 0.65 |
| Real events in top 1 percent | 6 of 10 |
| Real events in top 10 percent | 8 of 10 |
| Median percentile of real events | 99.16 |

If I forget a number, I say "roughly" and give the rounded version. I never invent precision.

---

## The two-minute script

### 0:00 to 0:20  Hook

"Every other team here built something on top of the stack: autonomy, computer vision, cyber. All of it depends on something boring underneath. Someone shipped the chips, the valves, the rocket motors. A ten thousand dollar drone gets grounded by a fifty dollar component that has exactly one supplier. Nobody else is looking at that layer. Chokepoint finds those single points of failure across nine million defense contracts."

### 0:20 to 0:50  Predictions vs reality

"I will start with the result that matters. I took ten publicly documented defense supply-chain disruptions from the last three years: the Sentinel ICBM cost breach, Patriot interceptor capacity, Columbia-class submarines, the F-35 mission computer. For each one I checked where the responsible vendor ranks in my model, out of almost fifty thousand. Six of the ten land in the top one percent. Eight in the top ten percent. Median percentile is ninety-nine point two. The model would have flagged these before the news did."

(Point at the four metric cards, then the top rows of the table.)

### 0:50 to 1:20  Live stress test

"This is not a static report. Pick any vendor and I remove them from the live supply graph and recompute who is left."

(Select Raytheon, click Simulate Failure. Let the animation run.)

"Removing Raytheon drops fifteen percent of its served agency-NAICS pairs to zero suppliers, and six of those are defense-critical categories. This graph is their local neighborhood, and the red nodes are the capabilities that go dark. It runs in under a millisecond because I preindex the coverage map at startup."

(Drag a node so they see it is interactive.)

### 1:20 to 1:40  Heatmap

"Zooming out: this is every sub-agency against every defense-critical NAICS category. Each red cell is a single vendor standing between the Department of Defense and that capability. This is the screen a Title III industrial-base analyst should be using to decide where to fund a second source. Right now they do this in spreadsheets."

### 1:40 to 1:55  Evidence

"Under the hood, the ranking is a supervised model. I had no ground truth, so I generated labels by simulating vendor removal, the same N-1 contingency idea used in power grids. On a held-out test set the model hits eighty percent recall at ten, double the centrality baseline, with bootstrap confidence intervals."

### 1:55 to 2:00  Close

"The production version uses CAGE codes, a real-time SAM.gov feed, and analyst feedback. This is the prototype of the tool the DoD industrial-base office currently runs by hand. Anyone can build a better drone. I built the tool that tells you which vendor failure grounds it."

---

## Section-by-section talking points (if I have more time or get walked through)

**Header.** 49,842 vendors, FY2024 to 2026 USAspending, live and operational.

**Leaderboard.** Three rankers side by side. I can toggle the sort to show the betweenness baseline produces a different and worse top 10. Caption shows recall@10 of 0.80 vs 0.40.

**Predictions vs reality.** The headline credibility section. Lead with it.

**Stress test.** Live N-1 simulation, interactive graph, sub-millisecond.

**Explain card.** Every score is interpretable. Top three engineered features carry about 80 percent of the weight. Not a black box.

**Critical-NAICS table.** Restricted to the DoD Critical Technology Areas list. Surfaces small sole-source vendors the primes overshadow.

**Heatmap.** Sole-source cells in red. The actionable artifact.

**Sankey.** Money flow pooling at chokepoint vendors. Pure visual support.

**Eval expander.** Methodology, recall@k with CIs, Spearman, train/test discipline.

---

## Judge questions and my answers

**Why this instead of autonomy or cyber?**
A ten thousand dollar drone fails because of a fifty dollar component sole-sourced from one vendor. Everyone built the drone. I built the tool that finds the fifty dollar chokepoint. It is also the layer a defense investor like Bow actually funds, because it is government-tech with a captive buyer.

**How do I know it works?**
Two independent ways. Held-out simulated removal gives 0.80 recall at 10, double the baseline. And six of ten real documented disruption events land in the top one percent of the ranking, median percentile 99.16. One is a simulation metric, the other is reality.

**Is this just a graph database project?**
No. A graph database is about storage and querying. This is graph-structured machine learning: structural features, simulated counterfactual labels, a supervised ranker, and live contingency simulation. NetworkX is the in-memory analytical substrate. Production would swap it for Neo4j as storage, but the methodology is independent of where the graph lives.

**Why not just use betweenness centrality?**
Centrality measures bridging, so it favors large generalist primes. It misses a small vendor that is the only supplier in one critical category. My model adds sole-source ratio and critical-NAICS concentration, and recall at 10 doubles from 0.40 to 0.80.

**How did you pick the ten validation events?**
From public DoD acquisition reports, GAO reports, NDAA filings, and FTC merger filings. Each event lists the vendor, year, and citation in the repo. The selection is independent of the model. They are the cases a procurement analyst would have wanted flagged in advance.

**Two of the events ranked low. What happened?**
Aerojet Rocketdyne and TransDigm. Both are vendor-identity issues, not model failures. Aerojet was acquired by L3Harris in 2023, and L3Harris ranks in the top one percent. TransDigm operates as dozens of subsidiaries with separate CAGE codes, so its concentration is split. This is exactly the limitation my production roadmap calls out: switch from name-string matching to CAGE or UEI parent rollup.

**How are the labels generated, exactly?**
For each vendor in the top-1000 candidate pool by footprint, I copy the graph, remove the vendor, and count the fraction of agency-NAICS pairs it served that now have zero suppliers. That fraction is the coverage_drop label. I split 75/25, train only on the train portion, and report on the held-out portion.

**Did the temporal features help?**
No, and I report that honestly. I engineered persistence and growth features over three fiscal years. The structural sole-source signals dominate the coverage_drop label so completely that the temporal features added no lift to the ranker. I kept them as descriptive analyst-facing labels in the UI.

**What is the false positive story?**
The top of the ranking includes large primes whose removal does not actually zero out coverage because alternatives exist. That is why I report coverage_drop alongside the score and why the critical-NAICS view matters. A high score plus a high simulated coverage drop is the real signal.

**How fast is the stress test?**
Sub-millisecond. The (agency, NAICS) coverage map is preindexed at startup, so removal is a set difference, not a graph rebuild.

**What would you do with another week?**
CAGE/UEI parent rollup to fix the vendor-identity misses, a calibration plot of predicted score against true coverage drop, and a NAICS-stratified evaluation so I can report recall inside specific critical categories like missiles or microelectronics.

---

## If something breaks during the demo

- If the live Space is slow, I have the local Docker version on my laptop as backup.
- If a stress test returns a small number, I lean into it: "this vendor is not a chokepoint, and the model correctly says so. Let me show you one that is."
- If asked about a vendor not in the pool, I explain the candidate pool is the top 1000 by footprint and the rest score near zero by construction.

---

## One-sentence version for hallway conversations

I built a system that ingests nine million public defense contracts, simulates what breaks if each vendor disappears, and trains a model to rank those single points of failure, validated against real documented disruptions where it puts six of ten in the top one percent.
