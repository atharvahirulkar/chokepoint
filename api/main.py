"""FastAPI app for Chokepoint. Loads scores + graph at startup."""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routers import explain, score, stress
from api.schemas import HealthResult, MetricsResult
from api.state import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("chokepoint.api")

# In-memory counters for /metrics. Process-local, resets on restart.
_metrics: dict[str, float | int | str | None] = {
    "api_requests_total": 0,
    "total_response_time_ms": 0.0,
    "stress_simulations_run": 0,
    "last_request_timestamp": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="Chokepoint API",
    version="0.1.0",
    description=(
        "Defense procurement supply-graph intelligence. "
        "Endpoints: /score, /stress/{vendor}, /explain/{vendor}, /health, /metrics, /eval."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_and_meter(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    _metrics["api_requests_total"] = int(_metrics["api_requests_total"]) + 1
    _metrics["total_response_time_ms"] = float(_metrics["total_response_time_ms"]) + duration_ms
    _metrics["last_request_timestamp"] = datetime.now(timezone.utc).isoformat()
    log.info(
        "%s %s -> %d  %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(score.router)
app.include_router(stress.router)
app.include_router(explain.router)


@app.get("/health", response_model=HealthResult)
def health() -> HealthResult:
    return HealthResult(
        status="ok",
        vendors_loaded=len(state.scores),
        graph_nodes=state.graph.number_of_nodes(),
        graph_edges=state.graph.number_of_edges(),
    )


@app.get("/metrics", response_model=MetricsResult)
def metrics() -> MetricsResult:
    n = int(_metrics["api_requests_total"]) or 1
    avg = float(_metrics["total_response_time_ms"]) / n
    return MetricsResult(
        api_requests_total=int(_metrics["api_requests_total"]),
        avg_response_time_ms=avg,
        stress_simulations_run=int(_metrics["stress_simulations_run"]),
        last_request_timestamp=_metrics["last_request_timestamp"],
    )


@app.get("/eval")
def eval_report() -> dict:
    """Return the persisted eval_report.json contents for the dashboard."""
    return state.eval_report
