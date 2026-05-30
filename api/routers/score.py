"""GET /score — leaderboard of vendors by chosen sort column."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas import VendorScore
from api.state import state

router = APIRouter()

ALLOWED_SORT = {"model_score", "baseline_score", "iso_score"}


def risk_tier(model_score: float) -> str:
    if model_score > 0.7:
        return "HIGH"
    if model_score > 0.4:
        return "MEDIUM"
    return "LOW"


@router.get("/score", response_model=list[VendorScore])
def score(
    limit: int = Query(20, ge=1, le=500),
    sort_by: str = Query("model_score"),
) -> list[VendorScore]:
    if sort_by not in ALLOWED_SORT:
        raise HTTPException(
            status_code=422, detail=f"sort_by must be one of {sorted(ALLOWED_SORT)}"
        )
    df = state.scores.nlargest(limit, sort_by)
    return [
        VendorScore(
            vendor_name=row.vendor_name,
            model_score=float(row.model_score),
            baseline_score=float(row.baseline_score),
            iso_score=float(row.iso_score),
            risk_tier=risk_tier(float(row.model_score)),
            agency_count=int(row.agency_count),
            naics_count=int(row.naics_count),
            critical_naics_count=int(row.critical_naics_count),
            total_award_value=float(row.total_award_value),
            sole_source_ratio=float(row.sole_source_ratio),
            critical_sole_source_count=int(row.critical_sole_source_count),
        )
        for row in df.itertuples()
    ]
