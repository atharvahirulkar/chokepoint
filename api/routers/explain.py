"""GET /explain/{vendor_name} — feature contributions + risk tier + text."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from api.routers.score import risk_tier
from api.schemas import ExplainResult, FeatureContribution
from api.state import state
from models.train import MODEL_FEATURE_COLS, engineer_features

router = APIRouter()


# Templates keyed by feature name; %.2f-formatted value gets injected.
_FEATURE_PHRASES: dict[str, str] = {
    "sole_source_breadth": (
        "sole-sources {value:.2f}-weighted breadth of NAICS categories"
    ),
    "critical_breadth": "concentrates sole-source risk in defense-critical NAICS",
    "footprint": "spans a {value:.1f}-weighted agency × NAICS footprint",
    "betweenness_centrality": "occupies high structural betweenness in the supply graph",
    "pagerank": "has high PageRank — connected to important agencies and NAICS",
    "eigenvector_centrality": "has high eigenvector centrality (neighbors of neighbors)",
    "articulation_point": "is a graph articulation point — removal disconnects the supply graph",
    "sole_source_ratio": "is sole supplier for {value:.0%} of its served pairs",
    "mean_pair_redundancy": "operates in low-redundancy markets (avg {value:.1f} suppliers per pair)",
    "log_naics_count": "supplies a broad NAICS spread",
    "log_agency_count": "serves many distinct sub-agencies",
    "log_total_award_value": "carries a large total contract value",
    "log_contract_count": "has a high contract volume",
    "log_avg_award_size": "carries a large average award size",
    "log_critical_sole_source_count": "is the sole source for multiple critical NAICS pairs",
    "log_critical_naics_count": "is active in many defense-critical NAICS",
    "naics_hhi": "concentrates award value in a narrow NAICS slice",
    "critical_naics_market_share": "holds a meaningful share of all defense-critical spend",
    "hhi_score": "shows concentrated value across few agencies",
    "degree_centrality": "has high degree centrality in the graph",
    "years_active": "has been active across multiple fiscal years",
    "years_active_ratio": "has been active across {value:.0%} of the observation window",
    "log_award_growth_ratio": "shows accelerating award growth over the observation window",
    "is_emerging_concentration": "is an emerging-concentration vendor (newly sole-sourcing)",
    "is_persistent_supplier": "is a persistent supplier (multi-year presence)",
    "persistent_critical_signal": "is the persistent sole source for defense-critical NAICS",
}


def _build_explanation_text(top_features: list[tuple[str, float]]) -> str:
    if not top_features:
        return "No dominant feature contributions detected."
    phrases: list[str] = []
    for name, value in top_features[:2]:
        template = _FEATURE_PHRASES.get(name, name + " is high")
        try:
            phrases.append(template.format(value=value))
        except (KeyError, ValueError):
            phrases.append(template)
    return "Flagged because vendor " + " and ".join(phrases) + "."


@router.get("/explain/{vendor_name}", response_model=ExplainResult)
def explain(vendor_name: str) -> ExplainResult:
    vid = state.lookup_vendor_id(vendor_name)
    if vid is None:
        raise HTTPException(status_code=404, detail=f"Vendor not found: {vendor_name}")

    row = state.scores.loc[state.scores["vendor_id"] == vid].iloc[0]
    model_score = float(row.model_score)

    if state.supervised_model is None or state.scaler is None:
        raise HTTPException(
            status_code=503, detail="Trained model artifacts not loaded"
        )

    # Engineer features on a one-row frame so the model sees the same inputs
    single = engineer_features(state.scores.loc[state.scores["vendor_id"] == vid])
    x = single[MODEL_FEATURE_COLS].to_numpy(dtype=float)
    x_scaled = state.scaler.transform(x)[0]

    importances = state.supervised_model.feature_importances_
    # Contribution = scaled feature * importance. The scaled value is z-score
    # against the population, so positive contribution = above-average pressure
    # on this dimension. Sort by absolute contribution.
    contributions = []
    for name, value, scaled_val, imp in zip(
        MODEL_FEATURE_COLS,
        single[MODEL_FEATURE_COLS].iloc[0].to_list(),
        x_scaled.tolist(),
        importances.tolist(),
    ):
        contributions.append(
            FeatureContribution(
                feature=name,
                value=float(value),
                importance=float(imp),
                contribution=float(scaled_val * imp),
            )
        )
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

    top_pairs = [(c.feature, c.value) for c in contributions[:3] if c.contribution > 0]
    return ExplainResult(
        vendor_name=str(row.vendor_name),
        model_score=model_score,
        risk_tier=risk_tier(model_score),
        feature_contributions=contributions,
        explanation_text=_build_explanation_text(top_pairs),
    )
