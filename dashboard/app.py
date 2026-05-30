"""CHOKEPOINT — Streamlit dashboard.

Reads only from the FastAPI backend; does not touch parquet files directly.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("CHOKEPOINT_API", "http://localhost:8000")
CACHE_TTL = 300


# ---------------------------------------------------------------------------
# API helpers (all cached for the dashboard's idle TTL)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def api_get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        r = requests.get(f"{API_URL}{path}", params=params or {}, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error on {path}: {e}")
        return None


def get_health() -> dict:
    return api_get("/health") or {}


def get_metrics() -> dict:
    return api_get("/metrics") or {}


def get_scores(limit: int = 50, sort_by: str = "model_score") -> pd.DataFrame:
    data = api_get("/score", {"limit": limit, "sort_by": sort_by}) or []
    return pd.DataFrame(data)


def get_stress(vendor: str) -> dict | None:
    return api_get(f"/stress/{vendor}")


def get_explain(vendor: str) -> dict | None:
    return api_get(f"/explain/{vendor}")


def get_eval() -> dict:
    return api_get("/eval") or {}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="CHOKEPOINT",
    page_icon="⚠️",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .chokepoint-title { font-size: 3.2rem; font-weight: 800; letter-spacing: 0.08em; }
      .chokepoint-sub { color: #888; margin-top: -0.6rem; }
      .risk-HIGH { color: #ff4d4f; font-weight: 700; }
      .risk-MEDIUM { color: #faad14; font-weight: 700; }
      .risk-LOW { color: #52c41a; font-weight: 700; }
      .footer { color: #555; text-align: center; padding-top: 2rem; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="chokepoint-title">CHOKEPOINT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="chokepoint-sub">Defense Procurement Supply-Graph Intelligence</div>',
    unsafe_allow_html=True,
)

# Header metrics
health = get_health()
eval_report = get_eval()
metrics = get_metrics()

scores_top = get_scores(limit=500, sort_by="model_score")
n_high = int((scores_top["risk_tier"] == "HIGH").sum()) if not scores_top.empty else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Vendors Monitored", f"{health.get('vendors_loaded', 0):,}")
c2.metric("High-Risk Vendors", f"{n_high:,}")
c3.metric("Graph Edges", f"{health.get('graph_edges', 0):,}")
freshness = eval_report.get("timestamp", "—")
if isinstance(freshness, str) and freshness != "—":
    try:
        freshness = datetime.fromisoformat(freshness).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        pass
c4.metric("Data Freshness", freshness)

st.divider()

# -----------------------------------------------------------------------------
# SECTION 1 — Threat Leaderboard
# -----------------------------------------------------------------------------
st.subheader("1. Threat Leaderboard")

sort_label_to_key = {
    "Supervised model": "model_score",
    "Betweenness baseline": "baseline_score",
    "IsolationForest (unsupervised)": "iso_score",
}
left, right = st.columns([3, 1])
with right:
    sort_label = st.selectbox(
        "Rank by", list(sort_label_to_key.keys()), index=0, key="sort_select"
    )
sort_key = sort_label_to_key[sort_label]
leaderboard = get_scores(limit=20, sort_by=sort_key)

if leaderboard.empty:
    st.info("No vendors returned by API.")
else:
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1

    def tier_badge(tier: str) -> str:
        return f"🔴 {tier}" if tier == "HIGH" else (f"🟠 {tier}" if tier == "MEDIUM" else f"🟢 {tier}")

    display = leaderboard.assign(
        Rank=leaderboard.index,
        Vendor=leaderboard["vendor_name"],
        Tier=leaderboard["risk_tier"].map(tier_badge),
        Model=leaderboard["model_score"].round(3),
        Baseline=leaderboard["baseline_score"].round(3),
        Iso=leaderboard["iso_score"].round(3),
        Agencies=leaderboard["agency_count"].astype(int),
        NAICS=leaderboard["naics_count"].astype(int),
        CritNAICS=leaderboard["critical_naics_count"].astype(int),
        SoleSrcRatio=leaderboard["sole_source_ratio"].round(3),
        AwardValue=leaderboard["total_award_value"].map(lambda v: f"${v:,.0f}"),
    )[
        [
            "Rank",
            "Vendor",
            "Tier",
            "Model",
            "Baseline",
            "Iso",
            "Agencies",
            "NAICS",
            "CritNAICS",
            "SoleSrcRatio",
            "AwardValue",
        ]
    ]
    st.dataframe(display, hide_index=True, use_container_width=True)

    recall = eval_report.get("recall_at_k", {})
    if recall:
        st.caption(
            f"Held-out test set Recall@10 — supervised model: "
            f"**{recall.get('model', {}).get('recall@10', 0):.2f}** | "
            f"betweenness baseline: {recall.get('baseline', {}).get('recall@10', 0):.2f} | "
            f"IsolationForest: {recall.get('iso', {}).get('recall@10', 0):.2f}"
        )

st.divider()

# -----------------------------------------------------------------------------
# SECTION 2 — Stress Test Simulator
# -----------------------------------------------------------------------------
st.subheader("2. Stress Test Simulator")
st.caption("Simulate vendor removal on the live supply graph and measure coverage collapse.")

vendor_options = get_scores(limit=500, sort_by="model_score")["vendor_name"].tolist()
default_idx = 0
selected_vendor = st.selectbox(
    "Select a vendor to simulate failure",
    options=vendor_options,
    index=default_idx,
    key="vendor_select",
)

if st.button("Simulate Vendor Failure", type="primary"):
    st.session_state["stress_target"] = selected_vendor

stress_target = st.session_state.get("stress_target", selected_vendor)
stress = get_stress(stress_target) if stress_target else None

if stress:
    drop_pct = stress["coverage_drop"] * 100
    crit_drop_pct = stress["critical_coverage_drop"] * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coverage Drop", f"{drop_pct:.1f}%")
    m2.metric("Critical-NAICS Drop", f"{crit_drop_pct:.1f}%")
    m3.metric("NAICS Affected", stress["naics_affected"])
    m4.metric("Agencies Impacted", stress["agencies_impacted"])

    if drop_pct >= 20:
        st.error(
            f"Removing **{stress['vendor_name']}** collapses {drop_pct:.1f}% of "
            f"its served (agency × NAICS) coverage. {stress['pairs_lost']} of "
            f"{stress['pairs_served']} pairs go to zero suppliers."
        )
    elif drop_pct >= 5:
        st.warning(
            f"Removing **{stress['vendor_name']}** drops {drop_pct:.1f}% of coverage "
            f"({stress['pairs_lost']} of {stress['pairs_served']} pairs)."
        )
    else:
        st.success(
            f"Removing **{stress['vendor_name']}** has limited structural impact "
            f"({drop_pct:.1f}% coverage drop)."
        )

    if stress["top_vulnerable_naics"]:
        st.markdown("**Top vulnerable NAICS (most agencies losing coverage):**")
        for desc in stress["top_vulnerable_naics"]:
            st.markdown(f"- {desc}")

st.divider()

# -----------------------------------------------------------------------------
# SECTION 3 — Vendor Explain Card
# -----------------------------------------------------------------------------
st.subheader("3. Vendor Explain Card")
explain = get_explain(stress_target) if stress_target else None
if explain:
    cA, cB = st.columns([1, 2])
    with cA:
        tier = explain["risk_tier"]
        tier_color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(tier, "⚪")
        st.markdown(f"### {tier_color} {tier} risk")
        st.metric("Model score", f"{explain['model_score']:.3f}")
        st.write(explain["explanation_text"])

    with cB:
        contribs = pd.DataFrame(explain["feature_contributions"])
        contribs = contribs.iloc[:9]  # top 9 by importance ordering from API
        contribs = contribs.sort_values("contribution", ascending=True)
        fig = px.bar(
            contribs,
            x="contribution",
            y="feature",
            orientation="h",
            title="Feature contributions (z-scored value × importance)",
            color="contribution",
            color_continuous_scale="RdBu_r",
        )
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# SECTION 4 — Critical-NAICS Chokepoints (bonus panel — on-theme)
# -----------------------------------------------------------------------------
st.subheader("4. Defense-Critical NAICS Chokepoints")
crit_top = eval_report.get("critical_ground_truth_top") or []
if crit_top:
    crit_df = pd.DataFrame(crit_top)
    crit_df["critical_coverage_drop"] = (crit_df["critical_coverage_drop"] * 100).round(1)
    crit_df["coverage_drop"] = (crit_df["coverage_drop"] * 100).round(1)
    crit_df = crit_df.rename(
        columns={
            "vendor": "Vendor",
            "critical_coverage_drop": "Critical Drop %",
            "critical_naics_affected": "Critical NAICS Lost",
            "coverage_drop": "Overall Drop %",
            "split": "Eval split",
        }
    )
    st.caption(
        "Vendors ranked by simulated coverage collapse restricted to the DoD "
        "Critical Technology Areas NAICS list (aerospace, missiles, naval propulsion, "
        "microelectronics, ordnance, guidance systems)."
    )
    st.dataframe(crit_df, hide_index=True, use_container_width=True)
else:
    st.info("No critical-NAICS chokepoints in current eval report.")

st.divider()

# -----------------------------------------------------------------------------
# SECTION 5 — Eval Transparency
# -----------------------------------------------------------------------------
with st.expander("5. Model Evaluation Methodology"):
    st.markdown(eval_report.get("ground_truth_rule", "_no eval report loaded_"))

    recall = eval_report.get("recall_at_k", {})
    recall_ci = eval_report.get("recall_at_k_95ci", {})
    if recall:
        rows = []
        for name in ("baseline", "iso", "model"):
            vals = recall.get(name, {})
            cis = recall_ci.get(name, {})
            row = {"Ranker": name}
            for k in (5, 10, 20, 50):
                key = f"recall@{k}"
                v = vals.get(key)
                ci = cis.get(key)
                if v is None:
                    continue
                if ci:
                    row[f"R@{k}"] = (
                        f"{v:.2f}  [{ci['lo']:.2f}, {ci['hi']:.2f}]"
                    )
                else:
                    row[f"R@{k}"] = f"{v:.2f}"
            rows.append(row)
        st.markdown("**Recall@k on held-out test split (95% bootstrap CI, n=1000):**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    spearman = eval_report.get("spearman_vs_true_coverage_drop", {})
    if spearman:
        st.markdown("**Spearman rank correlation vs true coverage_drop (test split):**")
        st.dataframe(
            pd.DataFrame(
                [{"Ranker": k, "Spearman": round(v, 3)} for k, v in spearman.items()]
            ),
            hide_index=True,
        )

    crit_recall = eval_report.get("critical_recall_at_k", {})
    if crit_recall:
        rows = []
        for name in ("baseline", "iso", "model"):
            vals = crit_recall.get(name, {})
            rows.append(
                {
                    "Ranker": name,
                    **{f"R@{k}": round(vals.get(f"recall@{k}", 0), 2) for k in (5, 10, 20, 50)},
                }
            )
        st.markdown(
            f"**Critical-NAICS recall@k** "
            f"(n={eval_report.get('n_test_critical_positives', 0)} test positives):"
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.caption(
        "Methodology: labels generated by simulating vendor removal on the bipartite "
        "(agency × NAICS) supply graph; train/test split (75/25, seed 42) on the "
        "candidate pool (top-1000 vendors by footprint). GradientBoosting trained on "
        "the train portion only; metrics reported on the held-out test portion."
    )

st.markdown(
    '<div class="footer">Built solo at SIC × DS³ × Bow Capital Defense Hackathon · '
    f"API: {API_URL}</div>",
    unsafe_allow_html=True,
)
