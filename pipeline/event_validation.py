"""Match known real-world defense supply-chain disruption events against
the Chokepoint scores, compute each vendor's rank/percentile across all
three rankers, and emit a validation report.

This is the "predictions vs reality" panel for the dashboard: it tells the
demo audience whether the supervised model would have surfaced each
documented chokepoint *without* the news telling us.

Output: data/processed/event_validation.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

EVENTS_PATH = Path("data/synthetic/known_events.json")
SCORES_PATH = Path("data/processed/vendor_scores.parquet")
OUT_PATH = Path("data/processed/event_validation.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("event_validation")


def best_match(query: str, scores: pd.DataFrame) -> pd.Series | None:
    """Pick the highest-model-score vendor whose name CONTAINS the query.

    Vendor name strings are messy (subsidiaries, divisions, legal suffixes).
    We accept the highest-ranked containing match because chokepoint-ness
    typically sits at the highest-aggregated parent entity in our graph.
    """
    q = query.strip().upper()
    hits = scores[scores["vendor_name"].str.contains(q, na=False)]
    if hits.empty:
        return None
    return hits.nlargest(1, "model_score").iloc[0]


def rank_in(series: pd.Series, target_value: float) -> int:
    """Return rank (1 = highest) of target_value within series (descending)."""
    return int((series > target_value).sum()) + 1


def main() -> None:
    if not EVENTS_PATH.exists():
        raise SystemExit(f"Missing {EVENTS_PATH}")
    if not SCORES_PATH.exists():
        raise SystemExit(f"Missing {SCORES_PATH}. Run `make train` first.")

    payload = json.loads(EVENTS_PATH.read_text())
    events = payload["events"]
    scores = pd.read_parquet(SCORES_PATH)
    n_vendors = len(scores)
    log.info("Validating %d events against %d vendors", len(events), n_vendors)

    results = []
    for ev in events:
        match = best_match(ev["vendor_query"], scores)
        if match is None:
            log.warning("No match for query: %s", ev["vendor_query"])
            results.append(
                {
                    **ev,
                    "matched_vendor": None,
                    "found": False,
                }
            )
            continue

        rank_model = rank_in(scores["model_score"], float(match.model_score))
        rank_baseline = rank_in(scores["baseline_score"], float(match.baseline_score))
        rank_iso = rank_in(scores["iso_score"], float(match.iso_score))
        pctile_model = 100.0 * (1 - rank_model / n_vendors)
        pctile_baseline = 100.0 * (1 - rank_baseline / n_vendors)
        pctile_iso = 100.0 * (1 - rank_iso / n_vendors)

        # Risk tier echo (matches the API definition)
        ms = float(match.model_score)
        if ms > 0.7:
            tier = "HIGH"
        elif ms > 0.4:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        results.append(
            {
                **ev,
                "found": True,
                "matched_vendor": str(match.vendor_name),
                "model_score": float(match.model_score),
                "baseline_score": float(match.baseline_score),
                "iso_score": float(match.iso_score),
                "risk_tier": tier,
                "rank_model": rank_model,
                "rank_baseline": rank_baseline,
                "rank_iso": rank_iso,
                "percentile_model": round(pctile_model, 2),
                "percentile_baseline": round(pctile_baseline, 2),
                "percentile_iso": round(pctile_iso, 2),
                "agency_count": int(match.agency_count),
                "naics_count": int(match.naics_count),
                "critical_naics_count": int(match.critical_naics_count),
            }
        )
        log.info(
            "  %s -> %s · rank=%d/%d (top %.2f%%) · score=%.3f",
            ev["vendor_query"],
            match.vendor_name,
            rank_model,
            n_vendors,
            pctile_model,
            ms,
        )

    # Aggregate stats
    found = [r for r in results if r.get("found")]
    if found:
        top_1pct = sum(1 for r in found if r["percentile_model"] >= 99.0)
        top_5pct = sum(1 for r in found if r["percentile_model"] >= 95.0)
        top_10pct = sum(1 for r in found if r["percentile_model"] >= 90.0)
        report = {
            "n_events": len(events),
            "n_found": len(found),
            "summary": {
                "in_top_1_pct": top_1pct,
                "in_top_5_pct": top_5pct,
                "in_top_10_pct": top_10pct,
                "median_percentile": round(
                    pd.Series([r["percentile_model"] for r in found]).median(), 2
                ),
            },
            "events": results,
            "n_vendors": n_vendors,
        }
    else:
        report = {
            "n_events": len(events),
            "n_found": 0,
            "summary": {},
            "events": results,
            "n_vendors": n_vendors,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", OUT_PATH)

    # Print a clean summary
    print("\n=== Event Validation Summary ===")
    print(f"Events: {len(found)}/{len(events)} matched against {n_vendors} vendors\n")
    print(f"{'Event':<55}{'Rank':>14}{'%ile':>8}{'Tier':>8}")
    for r in results:
        if not r.get("found"):
            print(f"{r['event'][:55]:<55}{'NOT FOUND':>30}")
            continue
        print(
            f"{r['event'][:55]:<55}{r['rank_model']:>6}/{r['rank_baseline']:<7}"
            f"{r['percentile_model']:>7.2f}%{r['risk_tier']:>8}"
        )
    if report["summary"]:
        print(f"\nIn top 1%: {report['summary']['in_top_1_pct']}/{len(found)}")
        print(f"In top 5%: {report['summary']['in_top_5_pct']}/{len(found)}")
        print(f"In top 10%: {report['summary']['in_top_10_pct']}/{len(found)}")
        print(f"Median percentile: {report['summary']['median_percentile']}%")


if __name__ == "__main__":
    main()
