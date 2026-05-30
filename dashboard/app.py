"""CHOKEPOINT - defense procurement command center.

Reads only from the FastAPI backend; does not touch parquet files directly.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

API_URL = os.environ.get("CHOKEPOINT_API", "http://localhost:8000")
CACHE_TTL = 300


# ---------------------------------------------------------------------------
# API helpers (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def api_get(path: str, params: dict | None = None):
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
# Page setup + global theme
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="CHOKEPOINT",
    page_icon="🛰️",
    initial_sidebar_state="collapsed",
)

# Command-center CSS: dark grid, monospace, terminal accents, radar sweep.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');

      :root {
        --bg-0: #06090d;
        --bg-1: #0b1118;
        --bg-2: #11181f;
        --border: #1c2630;
        --accent: #00ff9c;
        --accent-dim: #44b389;
        --amber: #ffb020;
        --alert: #ff3b3b;
        --text: #d8e2ec;
        --text-dim: #6b7884;
      }

      .stApp {
        background:
          radial-gradient(1200px 600px at 80% -200px, rgba(0, 255, 156, 0.04), transparent 60%),
          radial-gradient(900px 500px at 0% 100%, rgba(255, 59, 59, 0.04), transparent 60%),
          linear-gradient(180deg, #060a0e 0%, #060a0e 100%);
        background-attachment: fixed;
        color: var(--text);
      }
      .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background-image:
          linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 32px 32px;
        pointer-events: none;
        z-index: 0;
      }
      .block-container { padding-top: 1.2rem; position: relative; z-index: 1; }

      html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
      code, .stCode, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

      /* Header bar */
      .cp-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 1rem 1.25rem; margin-bottom: 1rem;
        background: linear-gradient(90deg, rgba(11,17,24,0.95), rgba(11,17,24,0.6));
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        position: relative; overflow: hidden;
      }
      .cp-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem; font-weight: 800; letter-spacing: 0.18em;
        color: var(--accent);
        text-shadow: 0 0 12px rgba(0,255,156,0.35);
      }
      .cp-sub {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-dim); letter-spacing: 0.15em;
        font-size: 0.85rem; margin-top: 0.2rem;
      }
      .cp-status {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem; color: var(--accent);
        display: flex; align-items: center; gap: 0.6rem;
      }
      .cp-status .dot {
        width: 0.6rem; height: 0.6rem; border-radius: 50%;
        background: var(--accent); box-shadow: 0 0 8px var(--accent);
        animation: pulse-dot 1.6s ease-in-out infinite;
      }
      @keyframes pulse-dot {
        0%, 100% { opacity: 0.5; transform: scale(0.85); }
        50%      { opacity: 1; transform: scale(1.1); }
      }

      /* Radar sweep in header */
      .cp-radar {
        position: absolute; top: -60px; right: -60px;
        width: 240px; height: 240px;
        border-radius: 50%;
        background:
          radial-gradient(circle at center, rgba(0,255,156,0.06) 0%, transparent 70%),
          conic-gradient(from 0deg, transparent 0deg, rgba(0,255,156,0.22) 30deg, transparent 60deg);
        animation: cp-sweep 4s linear infinite;
        pointer-events: none;
      }
      @keyframes cp-sweep {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
      }

      /* Section labels */
      .cp-section {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        letter-spacing: 0.18em;
        font-size: 0.8rem;
        margin-top: 1.4rem; margin-bottom: 0.3rem;
        padding-left: 0.6rem; border-left: 2px solid var(--accent);
      }
      .cp-section-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 0.4rem; color: var(--text); }
      .cp-section-cap { color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.8rem; }

      /* Metrics — overrides Streamlit defaults */
      [data-testid="stMetric"] {
        background: linear-gradient(180deg, var(--bg-1), var(--bg-2));
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.9rem 1rem;
      }
      [data-testid="stMetricLabel"] {
        color: var(--text-dim) !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.7rem !important;
      }
      [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 800 !important;
      }

      /* Risk tier badges */
      .risk-HIGH   { color: var(--alert); font-weight: 700; }
      .risk-MEDIUM { color: var(--amber); font-weight: 700; }
      .risk-LOW    { color: var(--accent-dim); font-weight: 700; }

      /* Buttons */
      .stButton > button {
        background: linear-gradient(180deg, #142028, #0e1820);
        color: var(--accent); border: 1px solid var(--accent-dim);
        font-family: 'JetBrains Mono', monospace; letter-spacing: 0.12em;
        text-transform: uppercase; font-weight: 700;
      }
      .stButton > button:hover {
        background: linear-gradient(180deg, #1a2a36, #142028);
        border-color: var(--accent);
        box-shadow: 0 0 10px rgba(0,255,156,0.2);
      }

      /* Alert box for stress test */
      .cp-alert {
        background: linear-gradient(90deg, rgba(255,59,59,0.18), rgba(255,59,59,0.05));
        border: 1px solid var(--alert); border-left: 4px solid var(--alert);
        padding: 0.9rem 1.1rem; margin: 0.8rem 0;
        font-family: 'JetBrains Mono', monospace;
        animation: cp-alertpulse 1.6s ease-in-out infinite;
      }
      @keyframes cp-alertpulse {
        0%, 100% { box-shadow: 0 0 0 rgba(255,59,59,0); }
        50%      { box-shadow: 0 0 18px rgba(255,59,59,0.35); }
      }
      .cp-warn {
        background: linear-gradient(90deg, rgba(255,176,32,0.15), rgba(255,176,32,0.05));
        border-left: 4px solid var(--amber);
        padding: 0.9rem 1.1rem; margin: 0.8rem 0;
        font-family: 'JetBrains Mono', monospace;
      }
      .cp-ok {
        background: linear-gradient(90deg, rgba(0,255,156,0.12), rgba(0,255,156,0.02));
        border-left: 4px solid var(--accent);
        padding: 0.9rem 1.1rem; margin: 0.8rem 0;
        font-family: 'JetBrains Mono', monospace;
      }

      /* Dataframe polish */
      [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 4px;
      }

      .footer {
        color: var(--text-dim); text-align: center;
        padding: 2rem 0 1rem 0; font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; letter-spacing: 0.15em;
      }

      /* Pulse rows in critical chokepoint table */
      .pulse-row {
        background: rgba(255, 59, 59, 0.08);
        animation: cp-rowpulse 2.4s ease-in-out infinite;
      }
      @keyframes cp-rowpulse {
        0%, 100% { background: rgba(255,59,59,0.06); }
        50%      { background: rgba(255,59,59,0.14); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Live header bar
health = get_health()
st.markdown(
    f"""
    <div class="cp-header">
      <div class="cp-radar"></div>
      <div>
        <div class="cp-title">CHOKEPOINT</div>
        <div class="cp-sub">DEFENSE PROCUREMENT &nbsp;&nbsp;·&nbsp;&nbsp; SUPPLY-GRAPH INTELLIGENCE &nbsp;&nbsp;·&nbsp;&nbsp; FY2026 USAspending</div>
      </div>
      <div class="cp-status">
        <span class="dot"></span>
        OPERATIONAL · {health.get('vendors_loaded', 0):,} VENDORS LOADED · {health.get('graph_nodes', 0):,} NODES
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Header metrics ------------------------------------------------------------
eval_report = get_eval()
scores_500 = get_scores(limit=500, sort_by="model_score")
n_high = int((scores_500["risk_tier"] == "HIGH").sum()) if not scores_500.empty else 0
n_articulation = (
    int(scores_500["articulation_point"].sum()) if "articulation_point" in scores_500 else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vendors Monitored", f"{health.get('vendors_loaded', 0):,}")
c2.metric("High-Risk Vendors", f"{n_high:,}")
c3.metric("Articulation Points", f"{n_articulation:,}")
c4.metric("Graph Edges", f"{health.get('graph_edges', 0):,}")


# ---------------------------------------------------------------------------
# SECTION 1 — Threat Leaderboard
# ---------------------------------------------------------------------------
st.markdown('<div class="cp-section">// 01 · THREAT LEADERBOARD</div>', unsafe_allow_html=True)
st.markdown('<div class="cp-section-cap">Top vendors ranked by supervised chokepoint model. Compare to centrality baseline and unsupervised IsolationForest.</div>', unsafe_allow_html=True)

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

if not leaderboard.empty:
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
        SoleSrc=leaderboard["sole_source_ratio"].round(3),
        Articulation=leaderboard["articulation_point"].map(lambda x: "⚠️" if x == 1 else ""),
        AwardValue=leaderboard["total_award_value"].map(lambda v: f"${v:,.0f}"),
    )[
        ["Rank", "Vendor", "Tier", "Model", "Baseline", "Iso",
         "Agencies", "NAICS", "CritNAICS", "SoleSrc", "Articulation", "AwardValue"]
    ]
    st.dataframe(display, hide_index=True, use_container_width=True, height=440)

    recall = eval_report.get("recall_at_k", {})
    if recall:
        st.caption(
            f"📊 Held-out test Recall@10 — supervised: "
            f"**{recall.get('model', {}).get('recall@10', 0):.2f}** | "
            f"betweenness: {recall.get('baseline', {}).get('recall@10', 0):.2f} | "
            f"IsolationForest: {recall.get('iso', {}).get('recall@10', 0):.2f} | "
            f"Spearman vs true coverage_drop: {eval_report.get('spearman_vs_true_coverage_drop', {}).get('model', 0):.2f}"
        )


# ---------------------------------------------------------------------------
# SECTION 2 — Stress Test (with interactive supply graph)
# ---------------------------------------------------------------------------
st.markdown('<div class="cp-section">// 02 · STRESS-TEST SIMULATOR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cp-section-cap">Remove a vendor from the live supply graph and observe coverage collapse. The interactive graph below renders the vendor\'s local subgraph; affected (agency × NAICS) pairs glow red after simulation.</div>',
    unsafe_allow_html=True,
)

vendor_options = scores_500["vendor_name"].tolist()
selected_vendor = st.selectbox(
    "Target vendor",
    options=vendor_options,
    index=0,
    key="vendor_select",
)
col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    if st.button("⚠ Simulate Failure", type="primary"):
        st.session_state["stress_target"] = selected_vendor
        st.session_state["stress_animate"] = True

stress_target = st.session_state.get("stress_target", selected_vendor)
stress = get_stress(stress_target) if stress_target else None
explain = get_explain(stress_target) if stress_target else None


def render_supply_graph(
    vendor_name: str, stress_data: dict | None, vendor_features: dict | None, height: int = 460
) -> str:
    """Render the vendor's local subgraph as a pyvis HTML snippet."""
    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#06090d",
        font_color="#d8e2ec",
        directed=False,
        notebook=False,
    )
    net.barnes_hut(
        gravity=-12000, central_gravity=0.3, spring_length=160, spring_strength=0.04
    )

    # Center: the vendor
    is_chokepoint = stress_data and stress_data.get("coverage_drop", 0) > 0.05
    vendor_color = "#ff3b3b" if is_chokepoint else "#00ff9c"
    n_pairs = stress_data["pairs_served"] if stress_data else (
        vendor_features.get("agency_count", 0) * vendor_features.get("naics_count", 0)
        if vendor_features else 1
    )
    vendor_size = min(60, 22 + (n_pairs ** 0.5))
    net.add_node(
        f"V::{vendor_name}",
        label=vendor_name,
        title=f"VENDOR · {vendor_name}",
        color=vendor_color,
        size=vendor_size,
        shape="dot",
        borderWidth=3,
        font={"size": 16, "color": "#ffffff", "face": "JetBrains Mono"},
    )

    if not stress_data:
        return net.generate_html(notebook=False)

    # Pull top vulnerable NAICS for emphasis
    vulnerable = set(stress_data.get("top_vulnerable_naics", []))

    # Use vendor features to build a synthetic representation if we don't
    # have neighbor-resolution at this layer. Show agency_count agencies and
    # the named NAICS that lost coverage.
    agencies_n = vendor_features.get("agency_count", 0) if vendor_features else 0
    naics_n = vendor_features.get("naics_count", 0) if vendor_features else 0

    # Plot up to 10 agency placeholders
    for i in range(min(10, agencies_n)):
        net.add_node(
            f"A::agency_{i}",
            label=f"AGENCY {i+1}",
            title=f"Sub-agency placeholder {i+1}",
            color="#3a6ea5",
            size=18,
            shape="square",
            font={"size": 11, "color": "#a0c4e8"},
        )
        net.add_edge(f"V::{vendor_name}", f"A::agency_{i}", color="#2a3a4d", width=1)

    # NAICS: emphasize vulnerable ones in red, others muted
    n_vuln = len(vulnerable)
    other_naics = max(0, min(12, naics_n - n_vuln))
    for desc in list(vulnerable)[:8]:
        short = desc if len(desc) <= 32 else desc[:30] + "…"
        net.add_node(
            f"N::{desc}",
            label=short,
            title=f"NAICS LOST · {desc}",
            color="#ff3b3b",
            size=22,
            shape="triangle",
            font={"size": 11, "color": "#ffb8b8"},
        )
        net.add_edge(
            f"V::{vendor_name}", f"N::{desc}",
            color="#ff3b3b", width=2.5,
            title="Coverage lost on removal",
        )
    for i in range(other_naics):
        net.add_node(
            f"N::other_{i}",
            label=f"NAICS {i+1}",
            title=f"NAICS placeholder {i+1}",
            color="#88aabc",
            size=12,
            shape="triangle",
            font={"size": 10, "color": "#88aabc"},
        )
        net.add_edge(f"V::{vendor_name}", f"N::other_{i}", color="#2a3a4d", width=1)

    return net.generate_html(notebook=False)


if stress:
    drop_pct = stress["coverage_drop"] * 100
    crit_drop_pct = stress["critical_coverage_drop"] * 100

    # Animated reveal (only if button was just pressed)
    if st.session_state.get("stress_animate"):
        placeholder = st.empty()
        for line in [
            "▶ Initiating contingency simulation…",
            "▶ Removing vendor node from supply graph…",
            "▶ Recomputing (agency × NAICS) coverage…",
            "▶ Identifying zero-supplier pairs…",
        ]:
            placeholder.markdown(
                f'<div class="cp-alert">{line}</div>', unsafe_allow_html=True
            )
            time.sleep(0.15)
        placeholder.empty()
        st.session_state["stress_animate"] = False

    # Top alert box
    if drop_pct >= 20 or crit_drop_pct >= 10:
        st.markdown(
            f"""
            <div class="cp-alert">
              <strong>⚠ CHOKEPOINT CONFIRMED</strong> · {stress['vendor_name']} ·
              {stress['pairs_lost']} of {stress['pairs_served']} served pairs collapse to zero suppliers
              ({drop_pct:.1f}% overall · {crit_drop_pct:.1f}% defense-critical).
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif drop_pct >= 5:
        st.markdown(
            f"""
            <div class="cp-warn">
              <strong>↯ ELEVATED RISK</strong> · {stress['vendor_name']} ·
              {drop_pct:.1f}% coverage drop on removal ({stress['pairs_lost']}/{stress['pairs_served']} pairs).
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="cp-ok">
              <strong>✓ STABLE</strong> · {stress['vendor_name']} ·
              {drop_pct:.1f}% coverage drop (limited structural impact).
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coverage Drop", f"{drop_pct:.1f}%")
    m2.metric("Critical-NAICS Drop", f"{crit_drop_pct:.1f}%")
    m3.metric("NAICS Affected", stress["naics_affected"])
    m4.metric("Agencies Impacted", stress["agencies_impacted"])

    # Interactive supply graph
    st.markdown("**Local supply graph** — vendor in center, blue squares = sub-agencies, triangles = NAICS. Red triangles + edges = coverage lost on removal.")
    vendor_row = scores_500.loc[scores_500["vendor_name"] == stress["vendor_name"]]
    vendor_features = vendor_row.iloc[0].to_dict() if not vendor_row.empty else None
    graph_html = render_supply_graph(stress["vendor_name"], stress, vendor_features)
    components.html(graph_html, height=470, scrolling=False)

    if stress["top_vulnerable_naics"]:
        st.markdown("**Vulnerable NAICS losing coverage:**")
        for desc in stress["top_vulnerable_naics"]:
            st.markdown(f"- {desc}")


# ---------------------------------------------------------------------------
# SECTION 3 — Explain Card
# ---------------------------------------------------------------------------
st.markdown('<div class="cp-section">// 03 · VENDOR EXPLAIN CARD</div>', unsafe_allow_html=True)
st.markdown('<div class="cp-section-cap">Why the model flagged this vendor — feature contributions (z-scored value × global importance), templated rationale.</div>', unsafe_allow_html=True)

if explain:
    cA, cB = st.columns([1, 2])
    with cA:
        tier = explain["risk_tier"]
        tier_color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(tier, "⚪")
        st.markdown(f"### {tier_color} {tier} RISK")
        st.metric("Model score", f"{explain['model_score']:.3f}")
        st.markdown(
            f'<div class="cp-ok"><em>{explain["explanation_text"]}</em></div>',
            unsafe_allow_html=True,
        )

    with cB:
        contribs = pd.DataFrame(explain["feature_contributions"])
        # Take top 10 by absolute contribution magnitude
        contribs = contribs.iloc[:10].copy()
        contribs = contribs.sort_values("contribution", ascending=True)
        fig = go.Figure(
            go.Bar(
                x=contribs["contribution"],
                y=contribs["feature"],
                orientation="h",
                marker=dict(
                    color=contribs["contribution"],
                    colorscale=[[0, "#3a6ea5"], [0.5, "#11181f"], [1, "#ff3b3b"]],
                    cmid=0,
                    line=dict(color="#1c2630", width=1),
                ),
                hovertemplate="<b>%{y}</b><br>contribution: %{x:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#06090d",
            plot_bgcolor="#06090d",
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(family="JetBrains Mono", color="#d8e2ec", size=11),
            xaxis=dict(title="Contribution (z-score × importance)", zerolinecolor="#1c2630", gridcolor="#11181f"),
            yaxis=dict(title=None, gridcolor="#11181f"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# SECTION 4 — Defense-Critical NAICS Chokepoints
# ---------------------------------------------------------------------------
st.markdown('<div class="cp-section">// 04 · DEFENSE-CRITICAL NAICS CHOKEPOINTS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cp-section-cap">Restricted to the DoD Critical Technology Areas NAICS list — aerospace, missiles, naval propulsion, microelectronics, ordnance, guidance. These vendors sole-source national-security capabilities.</div>',
    unsafe_allow_html=True,
)

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
    st.dataframe(crit_df, hide_index=True, use_container_width=True, height=440)
else:
    st.info("No critical-NAICS chokepoints in current eval report.")


# ---------------------------------------------------------------------------
# SECTION 5 — Sub-agency × critical-NAICS risk heatmap
# ---------------------------------------------------------------------------
st.markdown('<div class="cp-section">// 05 · CONCENTRATION HEATMAP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cp-section-cap">Supplier count per (sub-agency × defense-critical NAICS). Red cells = sole-source. Empty cells = no supplier in that pair today.</div>',
    unsafe_allow_html=True,
)

heatmap_data = api_get("/heatmap/critical") or {}
if heatmap_data and heatmap_data.get("agencies"):
    import numpy as _np
    z = _np.array(
        [
            [
                (v if v is not None else _np.nan) for v in row
            ]
            for row in heatmap_data["supplier_counts"]
        ],
        dtype=float,
    )
    # Custom colorscale: red at 1 (sole-source), amber at 2-3, green at 5+
    colorscale = [
        [0.0, "#ff3b3b"],
        [0.15, "#ffb020"],
        [0.40, "#44b389"],
        [1.0, "#00ff9c"],
    ]
    naics_short = [
        f"{c} · {d[:34] + '…' if len(d) > 35 else d}"
        for c, d in zip(heatmap_data["naics_codes"], heatmap_data["naics_descriptions"])
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=naics_short,
            y=heatmap_data["agencies"],
            colorscale=colorscale,
            zmin=1, zmax=10,
            hovertemplate="<b>%{y}</b><br>%{x}<br>suppliers: %{z:.0f}<extra></extra>",
            colorbar=dict(
                title="Suppliers",
                tickfont=dict(color="#d8e2ec"),
                titlefont=dict(color="#d8e2ec"),
            ),
            xgap=2, ygap=2,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#06090d",
        plot_bgcolor="#06090d",
        height=520,
        margin=dict(l=10, r=10, t=10, b=160),
        font=dict(family="JetBrains Mono", color="#d8e2ec", size=10),
        xaxis=dict(tickangle=-50, side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    n_sole = int(_np.nansum((z == 1).astype(int)))
    st.caption(
        f"🔴 {n_sole} sole-source (agency × critical-NAICS) pairs identified in the live graph. "
        f"Each red cell is a single vendor whose disappearance zero-supplies that pair for that sub-agency."
    )


# ---------------------------------------------------------------------------
# SECTION 6 — Money / risk flow Sankey
# ---------------------------------------------------------------------------
st.markdown('<div class="cp-section">// 06 · MONEY · RISK · FLOW</div>', unsafe_allow_html=True)
st.markdown('<div class="cp-section-cap">Top chokepoint vendors and the contract-value flow that pools at them. Width = total contract value.</div>', unsafe_allow_html=True)

# Build a synthetic Sankey from /score top-10 + their agency/NAICS counts.
# (Per-edge dollar values aren't surfaced by the API today; we approximate
# using total_award_value distributed evenly across agency_count.)
top10 = scores_500.head(10)
if not top10.empty:
    label_to_idx: dict[str, int] = {}
    labels: list[str] = []
    colors: list[str] = []

    def idx_of(name: str, color: str) -> int:
        if name not in label_to_idx:
            label_to_idx[name] = len(labels)
            labels.append(name)
            colors.append(color)
        return label_to_idx[name]

    src, dst, vals, link_colors = [], [], [], []

    # All-vendors aggregate source for left side
    sources_left = ["DoD CONTRACT POOL"]
    for s in sources_left:
        idx_of(s, "#1c2630")
    for _, row in top10.iterrows():
        src.append(idx_of("DoD CONTRACT POOL", "#1c2630"))
        dst.append(idx_of(row["vendor_name"], "#ff3b3b" if row["risk_tier"] == "HIGH" else "#ffb020"))
        vals.append(float(row["total_award_value"]))
        link_colors.append("rgba(255,59,59,0.18)" if row["risk_tier"] == "HIGH" else "rgba(255,176,32,0.18)")

    # Vendor -> "Critical NAICS exposure" or "Other NAICS"
    for _, row in top10.iterrows():
        crit_share = float(row.get("critical_naics_count", 0)) / max(1, float(row["naics_count"]))
        crit_val = float(row["total_award_value"]) * crit_share
        other_val = float(row["total_award_value"]) - crit_val
        if crit_val > 0:
            src.append(idx_of(row["vendor_name"], "#ff3b3b"))
            dst.append(idx_of("DEFENSE-CRITICAL NAICS", "#ff3b3b"))
            vals.append(crit_val)
            link_colors.append("rgba(255,59,59,0.35)")
        if other_val > 0:
            src.append(idx_of(row["vendor_name"], "#ffb020"))
            dst.append(idx_of("OTHER NAICS", "#88aabc"))
            vals.append(other_val)
            link_colors.append("rgba(136,170,188,0.18)")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18, thickness=14,
                line=dict(color="#1c2630", width=0.5),
                label=labels, color=colors,
            ),
            link=dict(source=src, target=dst, value=vals, color=link_colors),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#06090d",
        plot_bgcolor="#06090d",
        font=dict(family="JetBrains Mono", color="#d8e2ec", size=11),
        height=460, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# SECTION 7 — Eval Transparency
# ---------------------------------------------------------------------------
st.markdown('<div class="cp-section">// 07 · MODEL EVALUATION</div>', unsafe_allow_html=True)
with st.expander("Methodology, held-out recall, bootstrap CIs"):
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
                    row[f"R@{k}"] = f"{v:.2f}  [{ci['lo']:.2f}, {ci['hi']:.2f}]"
                else:
                    row[f"R@{k}"] = f"{v:.2f}"
            rows.append(row)
        st.markdown("**Held-out test recall@k (95% bootstrap CI, n=1000):**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    spearman = eval_report.get("spearman_vs_true_coverage_drop", {})
    if spearman:
        st.markdown("**Spearman vs true coverage_drop (test split):**")
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
        "Labels generated by simulating vendor removal on the bipartite "
        "(agency × NAICS) supply graph — N-1 contingency analysis. "
        "Train/test split (75/25, seed 42) on the candidate pool of top-1000 vendors "
        "by graph footprint. GradientBoosting trained on the train portion only; all "
        "metrics reported on the held-out test portion."
    )


st.markdown(
    '<div class="footer">CHOKEPOINT · SIC × DS³ × BOW CAPITAL · DEFENSE HACKATHON 2026 · BUILT SOLO</div>',
    unsafe_allow_html=True,
)
