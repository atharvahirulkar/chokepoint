"""GET /stress/{vendor_name} — simulate vendor removal on the live graph."""
from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, HTTPException

from api.schemas import StressResult
from api.state import state
from pipeline.critical_naics import critical_naics_id

router = APIRouter()


def _vulnerable_naics_descriptions(
    lost_pairs: list[tuple[str, str]], graph
) -> list[str]:
    """Pick the NAICS whose coverage loss is most severe (most agencies lost)."""
    loss_per_naics: dict[str, int] = defaultdict(int)
    for _, n in lost_pairs:
        loss_per_naics[n] += 1
    top = sorted(loss_per_naics.items(), key=lambda x: x[1], reverse=True)[:5]
    return [
        graph.nodes[nid].get("description") or graph.nodes[nid].get("code") or nid
        for nid, _ in top
    ]


@router.get("/stress/{vendor_name}", response_model=StressResult)
def stress(vendor_name: str) -> StressResult:
    vid = state.lookup_vendor_id(vendor_name)
    if vid is None:
        raise HTTPException(status_code=404, detail=f"Vendor not found: {vendor_name}")

    served = state.served_pairs_by_vendor.get(vid, set())
    if not served:
        raise HTTPException(
            status_code=404,
            detail=f"Vendor {vendor_name} has no graph edges (no contracts)",
        )

    lost = [pair for pair in served if state.pair_coverage[pair] == {vid}]
    served_critical = [p for p in served if critical_naics_id(p[1])]
    lost_critical = [p for p in lost if critical_naics_id(p[1])]

    coverage_drop = len(lost) / len(served) if served else 0.0
    critical_drop = (
        len(lost_critical) / len(served_critical) if served_critical else 0.0
    )
    agencies_impacted = len({a for a, _ in lost})

    # Track running counter for /metrics
    from api.main import _metrics
    _metrics["stress_simulations_run"] += 1

    return StressResult(
        vendor_name=state.graph.nodes[vid]["name"],
        coverage_drop=float(coverage_drop),
        critical_coverage_drop=float(critical_drop),
        naics_affected=len({n for _, n in lost}),
        critical_naics_affected=len({n for _, n in lost_critical}),
        agencies_impacted=agencies_impacted,
        pairs_served=len(served),
        pairs_lost=len(lost),
        top_vulnerable_naics=_vulnerable_naics_descriptions(lost, state.graph),
    )
