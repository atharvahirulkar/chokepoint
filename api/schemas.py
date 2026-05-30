"""Pydantic response schemas for the Chokepoint API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class VendorScore(BaseModel):
    vendor_name: str
    model_score: float
    baseline_score: float
    iso_score: float
    risk_tier: str
    agency_count: int
    naics_count: int
    critical_naics_count: int
    contract_count: int
    total_award_value: float
    avg_award_size: float
    sole_source_ratio: float
    critical_sole_source_count: int
    mean_pair_redundancy: float
    naics_hhi: float
    critical_naics_market_share: float
    articulation_point: int


class StressResult(BaseModel):
    vendor_name: str
    coverage_drop: float = Field(
        ..., description="Fraction of (agency, NAICS) pairs that go to zero suppliers"
    )
    critical_coverage_drop: float
    naics_affected: int
    critical_naics_affected: int
    agencies_impacted: int
    pairs_served: int
    pairs_lost: int
    top_vulnerable_naics: list[str]


class FeatureContribution(BaseModel):
    feature: str
    value: float
    importance: float
    contribution: float = Field(
        ..., description="value * importance, signed by deviation from median"
    )


class ExplainResult(BaseModel):
    vendor_name: str
    model_score: float
    risk_tier: str
    feature_contributions: list[FeatureContribution]
    explanation_text: str


class HealthResult(BaseModel):
    status: str
    vendors_loaded: int
    graph_nodes: int
    graph_edges: int


class MetricsResult(BaseModel):
    api_requests_total: int
    avg_response_time_ms: float
    stress_simulations_run: int
    last_request_timestamp: str | None
