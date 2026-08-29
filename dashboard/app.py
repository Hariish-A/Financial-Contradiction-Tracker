"""
dashboard/app.py
-----------------
Milestone 6 — Main Streamlit Dashboard for the Financial Contradiction Tracker.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.database import init_db
from dashboard.data_fetcher import (
    fetch_all_credibility_scores,
    fetch_contradictions_df,
    fetch_executives,
    fetch_predictions_df,
    fetch_summary_stats,
    run_semantic_search,
    verify_prediction,
    fetch_pending_reviews,
    fetch_review_history,
    submit_review_decision,
)

# Ensure database tables and migrations are initialized on dashboard launch
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Contradiction Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Dark Premium Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #111827;
    --bg-card:       #1a2236;
    --bg-card-hover: #1f2d45;
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-orange: #f59e0b;
    --accent-red:    #ef4444;
    --accent-purple: #8b5cf6;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --border:        #1e293b;
    --border-bright: #334155;
}

/* ── Base ── */
html, body, [data-testid="stApp"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
    font-weight: 600 !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--accent-blue) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] {
    color: var(--accent-blue) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Components
# ─────────────────────────────────────────────────────────────────────────────

def _tier_color(tier: str) -> str:
    return {
        "LOW RISK":    "#10b981",
        "MEDIUM RISK": "#f59e0b",
        "HIGH RISK":   "#ef4444",
    }.get(tier, "#94a3b8")


def _type_color(ctype: str) -> str:
    return {
        "HARD":    "#ef4444",
        "SOFT":    "#f59e0b",
        "OMISSION":"#8b5cf6",
    }.get(ctype, "#94a3b8")


def _type_icon(ctype: str) -> str:
    return {"HARD": "🔴", "SOFT": "🟡", "OMISSION": "🟣"}.get(ctype, "⚪")


def stat_card(label: str, value, icon: str = "", delta: str = None, color: str = "#3b82f6"):
    delta_html = f'<p style="color:#94a3b8;font-size:0.75rem;margin:0;">{delta}</p>' if delta else ""
    st.markdown(f"""
<div style="
    background: #1a2236;
    border: 1px solid #1e293b;
    border-left: 3px solid {color};
    border-radius: 12px;
    padding: 20px;
    transition: all 0.2s;
">
    <p style="color:#94a3b8;font-size:0.75rem;font-weight:600;
              text-transform:uppercase;letter-spacing:0.08em;margin:0 0 8px 0;">
        {icon} {label}
    </p>
    <p style="color:#f1f5f9;font-size:2rem;font-weight:800;margin:0 0 4px 0;
              line-height:1.1;font-family:'Inter',sans-serif;">{value}</p>
    {delta_html}
</div>
""", unsafe_allow_html=True)


def contradiction_card(row: dict):
    ctype  = row.get("contradiction_type", "")
    score  = row.get("score", 0)
    color  = _type_color(ctype)
    icon   = _type_icon(ctype)
    qa     = f"{row.get('quarter_a','')} FY{row.get('year_a','')}"
    qb     = f"{row.get('quarter_b','')} FY{row.get('year_b','')}"
    text_a = (row.get("statement_a_text") or "")[:300]
    text_b = (row.get("statement_b_text") or "")[:300]

    # Extra detail for omissions
    omission_detail = ""
    if ctype == "OMISSION":
        details   = row.get("details", {})
        topic     = details.get("topic", "—")
        prior_qs  = ", ".join(details.get("prior_quarters", []))
        omit_q    = details.get("omitted_quarter", "—")
        omission_detail = f"""
<div style="margin-top:10px;padding:10px 14px;background:#0a0e1a;border-radius:8px;border:1px solid #1e293b;">
    <p style="color:#8b5cf6;font-size:0.75rem;font-weight:600;margin:0 0 4px 0;">
        DROPPED TOPIC
    </p>
    <p style="color:#f1f5f9;font-size:0.9rem;font-weight:500;margin:0 0 4px 0;">"{topic}"</p>
    <p style="color:#94a3b8;font-size:0.75rem;margin:0;">
        Mentioned in: {prior_qs} → Absent in: {omit_q}
    </p>
</div>
"""

    st.markdown(f"""
<div style="
    background:#1a2236;
    border:1px solid #1e293b;
    border-top: 3px solid {color};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <span style="
            background:{color}22;color:{color};
            border:1px solid {color}44;
            border-radius:20px;padding:4px 14px;
            font-size:0.75rem;font-weight:700;letter-spacing:0.08em;
        ">{icon} {ctype}</span>
        <span style="color:#94a3b8;font-size:0.8rem;font-weight:500;">
            Score: <strong style="color:{color};">{score:.2f}</strong>
        </span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 40px 1fr;gap:10px;align-items:start;">
        <div style="background:#0d1526;border-radius:8px;padding:12px;border:1px solid #1e293b;">
            <p style="color:#94a3b8;font-size:0.7rem;font-weight:600;margin:0 0 6px 0;
                       text-transform:uppercase;letter-spacing:0.08em;">
                📅 {qa}
            </p>
            <p style="color:#f1f5f9;font-size:0.85rem;line-height:1.5;margin:0;">{text_a}…</p>
        </div>
        <div style="display:flex;align-items:center;justify-content:center;height:100%;">
            <span style="font-size:1.2rem;color:{color};">→</span>
        </div>
        <div style="background:#0d1526;border-radius:8px;padding:12px;border:1px solid {color}44;">
            <p style="color:#94a3b8;font-size:0.7rem;font-weight:600;margin:0 0 6px 0;
                       text-transform:uppercase;letter-spacing:0.08em;">
                📅 {qb}
            </p>
            <p style="color:#f1f5f9;font-size:0.85rem;line-height:1.5;margin:0;">{text_b}…</p>
        </div>
    </div>
    {omission_detail}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 24px 0;border-bottom:1px solid #1e293b;margin-bottom:24px;">
        <div style="font-size:1.6rem;font-weight:800;color:#f1f5f9;line-height:1.2;">
            📊 ContraGuard
        </div>
        <div style="color:#3b82f6;font-size:0.75rem;font-weight:600;
                    letter-spacing:0.1em;text-transform:uppercase;margin-top:4px;">
            Financial Contradiction Tracker
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary stats in sidebar
    stats = fetch_summary_stats()
    st.markdown("### 📋 Database Summary")
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:8px;">
        <div style="background:#1a2236;border-radius:8px;padding:10px 14px;
                    border:1px solid #1e293b;display:flex;justify-content:space-between;">
            <span style="color:#94a3b8;font-size:0.82rem;">Companies</span>
            <span style="color:#f1f5f9;font-weight:700;">{stats['companies']}</span>
        </div>
        <div style="background:#1a2236;border-radius:8px;padding:10px 14px;
                    border:1px solid #1e293b;display:flex;justify-content:space-between;">
            <span style="color:#94a3b8;font-size:0.82rem;">Executives</span>
            <span style="color:#f1f5f9;font-weight:700;">{stats['executives']}</span>
        </div>
        <div style="background:#1a2236;border-radius:8px;padding:10px 14px;
                    border:1px solid #1e293b;display:flex;justify-content:space-between;">
            <span style="color:#94a3b8;font-size:0.82rem;">Statements</span>
            <span style="color:#f1f5f9;font-weight:700;">{stats['statements']:,}</span>
        </div>
        <div style="background:#1a2236;border-radius:8px;padding:10px 14px;
                    border:1px solid #1e293b;display:flex;justify-content:space-between;">
            <span style="color:#94a3b8;font-size:0.82rem;">Contradictions</span>
            <span style="color:#ef4444;font-weight:700;">{stats['total_contradictions']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔧 Filters")
    exec_df   = fetch_executives()
    exec_opts = {"All Executives": None}
    if not exec_df.empty:
        for _, r in exec_df.iterrows():
            label = f"{r['name']} ({r['role']})"
            exec_opts[label] = int(r["id"])

    selected_exec_label = st.selectbox(
        "Executive", options=list(exec_opts.keys()), key="exec_filter"
    )
    selected_exec_id = exec_opts[selected_exec_label]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#475569;font-size:0.72rem;padding-top:16px;
                border-top:1px solid #1e293b;line-height:1.6;">
        <strong style="color:#64748b;">Bloomberg tracks keywords.</strong><br>
        This system detects <em>reasoning-level</em> contradictions.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 24px 0;">
    <h1 style="font-size:2.4rem;font-weight:800;color:#f1f5f9;margin:0;line-height:1.2;">
        Financial Contradiction Tracker
    </h1>
    <p style="color:#64748b;font-size:1rem;margin:8px 0 0 0;font-weight:400;">
        Detecting reasoning-level contradictions in Indian earnings call transcripts ·
        Scoring executive credibility across 8 quarters
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Top KPI Strip
# ─────────────────────────────────────────────────────────────────────────────
stats = fetch_summary_stats()
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    stat_card("Transcripts Ingested", stats["transcripts"], "📄", color="#3b82f6")
with k2:
    stat_card("Statements Extracted", f"{stats['statements']:,}", "🗣️", color="#06b6d4")
with k3:
    stat_card("Hard Contradictions", stats["hard"], "🔴", color="#ef4444")
with k4:
    stat_card("Soft Contradictions", stats["soft"], "🟡", color="#f59e0b")
with k5:
    stat_card("Omissions Detected", stats["omissions"], "🟣", color="#8b5cf6")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5 Main Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆  Executive Scorecard",
    "⚡  Contradiction Timeline",
    "🔍  Semantic Search",
    "✅  Prediction Verification",
    "⚖️  Review Queue",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Executive Scorecard
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    scores = fetch_all_credibility_scores()

    if not scores:
        st.info("No credibility data yet. Run `python run_credibility.py --extract-predictions` and `--score` first.", icon="ℹ️")
    else:
        # Filter to selected exec if one chosen
        display_scores = scores
        if selected_exec_id is not None:
            display_scores = [s for s in scores if s["executive_id"] == selected_exec_id]

        # ── Chart ────────────────────────────────────────────────────────────
        df_scores = pd.DataFrame(display_scores)
        df_scores["label"] = df_scores["name"] + "\n" + df_scores["company"]
        df_scores["color"] = df_scores["risk_tier"].map({
            "LOW RISK":    "#10b981",
            "MEDIUM RISK": "#f59e0b",
            "HIGH RISK":   "#ef4444",
        })

        fig = go.Figure()

        for tier, grp in df_scores.groupby("risk_tier"):
            fig.add_trace(go.Bar(
                x=grp["label"],
                y=grp["credibility_score"],
                name=tier,
                marker_color=grp["color"],
                text=[f"{v:.0f}" for v in grp["credibility_score"]],
                textposition="outside",
                textfont=dict(size=13, color="#f1f5f9", family="Inter"),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Score: %{y:.1f}<br>"
                    "<extra></extra>"
                ),
            ))

        fig.add_hline(y=70, line_dash="dot", line_color="#10b981",
                      annotation_text="LOW RISK threshold (70)",
                      annotation_position="right",
                      annotation_font_color="#10b981")
        fig.add_hline(y=50, line_dash="dot", line_color="#f59e0b",
                      annotation_text="MEDIUM RISK threshold (50)",
                      annotation_position="right",
                      annotation_font_color="#f59e0b")

        fig.update_layout(
            title=dict(
                text="Executive Credibility Scores",
                font=dict(size=18, color="#f1f5f9", family="Inter"),
            ),
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(
                tickfont=dict(size=11, color="#94a3b8"),
                gridcolor="#1e293b",
                linecolor="#1e293b",
            ),
            yaxis=dict(
                range=[0, 115],
                tickfont=dict(size=11, color="#94a3b8"),
                gridcolor="#1e293b",
                linecolor="#1e293b",
                title="Credibility Score (0–100)",
                title_font=dict(color="#64748b"),
            ),
            barmode="group",
            showlegend=True,
            legend=dict(
                bgcolor="#1a2236",
                bordercolor="#1e293b",
                borderwidth=1,
                font=dict(color="#94a3b8"),
            ),
            margin=dict(t=60, b=80, l=60, r=60),
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Score Cards ──────────────────────────────────────────────────────
        st.markdown("### 📋 Individual Executive Breakdown")
        for r in display_scores:
            tier_col = _tier_color(r["risk_tier"])
            pred_html = ""
            if r["verified_predictions"] > 0:
                acc = r.get("direction_accuracy_pct") or 0
                pred_html = f"""
<div style="margin-top:10px;display:flex;gap:12px;flex-wrap:wrap;">
    <span style="background:#1e293b;border-radius:6px;padding:4px 12px;color:#94a3b8;font-size:0.78rem;">
        ✓ {r['direction_correct']} correct · ✗ {r['direction_wrong']} wrong
    </span>
    <span style="background:#1e293b;border-radius:6px;padding:4px 12px;color:#94a3b8;font-size:0.78rem;">
        Direction Accuracy: <strong style="color:#f1f5f9;">{acc:.1f}%</strong>
    </span>
</div>
"""

            st.markdown(f"""
<div style="
    background:#1a2236;
    border:1px solid #1e293b;
    border-left:4px solid {tier_col};
    border-radius:12px;
    padding:20px 24px;
    margin-bottom:12px;
">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
        <div>
            <h3 style="color:#f1f5f9;font-size:1.1rem;font-weight:700;margin:0 0 4px 0;">
                {r['name']} &nbsp;<span style="color:#64748b;font-weight:400;font-size:0.9rem;">({r['role']})</span>
            </h3>
            <p style="color:#64748b;font-size:0.82rem;margin:0;">{r['company']}</p>
        </div>
        <div style="text-align:right;">
            <div style="font-size:2.2rem;font-weight:800;color:{tier_col};line-height:1;">
                {r['credibility_score']:.0f}
            </div>
            <div style="font-size:0.7rem;color:{tier_col};font-weight:600;
                        letter-spacing:0.08em;">{r['risk_tier']}</div>
        </div>
    </div>
    <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;">
        <span style="background:#ef444422;border:1px solid #ef444444;border-radius:6px;
                     padding:4px 12px;color:#ef4444;font-size:0.78rem;font-weight:600;">
            🔴 {r['hard_contradictions']} HARD
        </span>
        <span style="background:#f59e0b22;border:1px solid #f59e0b44;border-radius:6px;
                     padding:4px 12px;color:#f59e0b;font-size:0.78rem;font-weight:600;">
            🟡 {r['soft_contradictions']} SOFT
        </span>
        <span style="background:#8b5cf622;border:1px solid #8b5cf644;border-radius:6px;
                     padding:4px 12px;color:#8b5cf6;font-size:0.78rem;font-weight:600;">
            🟣 {r['omission_contradictions']} OMISSION
        </span>
        <span style="background:#1e293b;border-radius:6px;
                     padding:4px 12px;color:#64748b;font-size:0.78rem;">
            {r['total_statements']} statements · {r['total_predictions']} predictions
        </span>
    </div>
    {pred_html}
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Contradiction Timeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### ⚡ Contradiction Timeline")
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem;margin-top:-8px;margin-bottom:20px;'>"
        "Side-by-side comparison of contradicting executive statements across quarters."
        "</p>",
        unsafe_allow_html=True,
    )

    # Filters row
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        type_filter = st.multiselect(
            "Contradiction Type",
            options=["HARD", "SOFT", "OMISSION"],
            default=["HARD", "SOFT", "OMISSION"],
            key="timeline_type_filter",
        )
    with f2:
        min_score = st.slider(
            "Minimum Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key="timeline_score_filter",
        )
    with f3:
        st.markdown("<br>", unsafe_allow_html=True)

    contradictions_df = fetch_contradictions_df(selected_exec_id)

    if contradictions_df.empty:
        st.info("No contradictions found. Run the contradiction pipeline first.", icon="ℹ️")
    else:
        filtered = contradictions_df[
            contradictions_df["contradiction_type"].isin(type_filter) &
            (contradictions_df["score"] >= min_score)
        ]

        if filtered.empty:
            st.warning("No contradictions match the current filters.", icon="⚠️")
        else:
            # Donut chart summary
            col_chart, col_list = st.columns([1, 2])
            with col_chart:
                type_counts = filtered["contradiction_type"].value_counts()
                fig_donut = go.Figure(go.Pie(
                    labels=type_counts.index.tolist(),
                    values=type_counts.values.tolist(),
                    hole=0.65,
                    marker_colors=["#ef4444", "#f59e0b", "#8b5cf6"],
                    textinfo="label+percent",
                    textfont=dict(color="#f1f5f9", size=12),
                ))
                fig_donut.update_layout(
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=220,
                    annotations=[dict(
                        text=f"<b>{len(filtered)}</b><br>total",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=16, color="#f1f5f9", family="Inter"),
                    )],
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            with col_list:
                st.markdown("<br>", unsafe_allow_html=True)
                for ctype in ["HARD", "SOFT", "OMISSION"]:
                    cnt = (filtered["contradiction_type"] == ctype).sum()
                    c   = _type_color(ctype)
                    i   = _type_icon(ctype)
                    st.markdown(f"""
<div style="background:#1a2236;border:1px solid #1e293b;border-left:3px solid {c};
            border-radius:8px;padding:10px 16px;margin-bottom:8px;
            display:flex;justify-content:space-between;">
    <span style="color:#94a3b8;font-size:0.9rem;">{i} {ctype} contradictions</span>
    <strong style="color:{c};font-size:1rem;">{cnt}</strong>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"**Showing {len(filtered)} contradiction(s)**")

            for _, row in filtered.head(50).iterrows():
                contradiction_card(row.to_dict())


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Semantic Search
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔍 Semantic Statement Search")
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem;margin-top:-8px;margin-bottom:20px;'>"
        "Search any executive's past statements by topic using FAISS vector similarity. "
        "Results are ranked by semantic closeness, not keyword matches."
        "</p>",
        unsafe_allow_html=True,
    )

    if exec_df.empty:
        st.info("No executives in database yet. Run the full pipeline first.", icon="ℹ️")
    else:
        s1, s2 = st.columns([2, 1])
        with s1:
            search_query = st.text_input(
                "Search Query",
                placeholder="e.g. rural segment growth, margin expansion, CAPEX guidance…",
                key="semantic_query",
            )
        with s2:
            search_exec_opts = {
                f"{r['name']} ({r['role']})": int(r["id"])
                for _, r in exec_df.iterrows()
            }
            search_exec_label = st.selectbox(
                "Executive", options=list(search_exec_opts.keys()), key="search_exec"
            )
            search_exec_id = search_exec_opts[search_exec_label]

        top_k = st.slider("Number of results", 3, 15, 7, key="search_top_k")

        if st.button("🔍 Search Statements", key="search_btn"):
            if not search_query.strip():
                st.warning("Please enter a search query.", icon="⚠️")
            else:
                with st.spinner("Building FAISS index and searching…"):
                    try:
                        results = run_semantic_search(search_exec_id, search_query, top_k)
                        if not results:
                            st.info("No results found. Make sure embeddings are backfilled (`--backfill`).", icon="ℹ️")
                        else:
                            st.markdown(f"**{len(results)} result(s) for:** *\"{search_query}\"*")
                            st.markdown("---")
                            for i, res in enumerate(results, 1):
                                sim_pct = int(res["score"] * 100)
                                sim_color = "#10b981" if sim_pct >= 70 else "#f59e0b" if sim_pct >= 50 else "#64748b"
                                sent_icon = {"positive": "📈", "negative": "📉", "neutral": "➖"}.get(
                                    (res.get("sentiment") or "").lower(), "•"
                                )
                                st.markdown(f"""
<div style="
    background:#1a2236;
    border:1px solid #1e293b;
    border-radius:12px;
    padding:18px 20px;
    margin-bottom:10px;
">
    <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:10px;">
        <div style="display:flex;gap:10px;align-items:center;">
            <span style="
                background:#0a0e1a;
                border:1px solid #334155;
                border-radius:6px;
                padding:3px 10px;
                color:#94a3b8;
                font-size:0.75rem;
                font-weight:600;
            ">#{i}</span>
            <span style="color:#64748b;font-size:0.82rem;">
                📅 {res['quarter']} FY{res['year']}
            </span>
            <span style="color:#64748b;font-size:0.82rem;">
                {sent_icon} {(res.get('sentiment') or 'neutral').capitalize()}
            </span>
        </div>
        <div style="
            background:{sim_color}22;
            border:1px solid {sim_color}55;
            border-radius:20px;
            padding:4px 14px;
            color:{sim_color};
            font-size:0.8rem;
            font-weight:700;
        ">
            {sim_pct}% match
        </div>
    </div>
    <p style="color:#f1f5f9;font-size:0.9rem;line-height:1.65;margin:0;">
        {res['text']}
    </p>
</div>
""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Search failed: {e}", icon="🚨")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: Prediction Verification
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ✅ Prediction Verification")
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem;margin-top:-8px;margin-bottom:20px;'>"
        "Record actual financial outcomes against extracted predictions to calculate "
        "direction accuracy and update credibility scores."
        "</p>",
        unsafe_allow_html=True,
    )

    pred_df = fetch_predictions_df(selected_exec_id)

    if pred_df.empty:
        st.info(
            "No predictions found. Run `python run_credibility.py --extract-predictions` first.",
            icon="ℹ️",
        )
    else:
        # Stats row
        total_p    = len(pred_df)
        verified_p = int((pred_df["verified"] == 1).sum())
        pending_p  = total_p - verified_p

        v1, v2, v3 = st.columns(3)
        with v1:
            stat_card("Total Predictions", total_p, "📊", color="#3b82f6")
        with v2:
            stat_card("Verified", verified_p, "✅", color="#10b981")
        with v3:
            stat_card("Pending", pending_p, "⏳", color="#f59e0b")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Pending predictions ─────────────────────────────────────────────
        pending_df = pred_df[pred_df["verified"] != 1].copy()
        if not pending_df.empty:
            st.markdown("#### ⏳ Pending Predictions — Enter Actual Values")
            st.markdown(
                "<p style='color:#64748b;font-size:0.82rem;margin-bottom:16px;'>"
                "Once the quarterly results are published, enter the actual figure "
                "and click <strong>Save</strong> to verify the prediction."
                "</p>",
                unsafe_allow_html=True,
            )
            for _, pred in pending_df.iterrows():
                p_col1, p_col2, p_col3 = st.columns([3, 2, 1])
                with p_col1:
                    st.markdown(f"""
                    <div style="background:#1a2236;border:1px solid #1e293b;border-radius:10px;
                                padding:14px 18px;">
                        <p style="color:#64748b;font-size:0.72rem;font-weight:600;
                                  text-transform:uppercase;margin:0 0 4px 0;letter-spacing:0.06em;">
                            ID #{pred['id']} · {pred.get('executive_name','?')} · {pred['quarter']}
                        </p>
                        <p style="color:#f1f5f9;font-size:0.88rem;margin:0 0 6px 0;font-weight:500;">
                            {pred['metric'].replace('_',' ').title()}
                        </p>
                        <div style="display:flex;gap:8px;">
                            <span style="background:#1e293b;border-radius:6px;padding:2px 10px;
                                         color:#94a3b8;font-size:0.78rem;">
                                Predicted: <strong>{pred['predicted_value']}</strong>
                            </span>
                            <span style="background:#1e293b;border-radius:6px;padding:2px 10px;
                                         color:#94a3b8;font-size:0.78rem;">
                                Direction: {pred['direction'].upper()}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with p_col2:
                    actual_val = st.number_input(
                        f"Actual value",
                        key=f"actual_{pred['id']}",
                        value=None,
                        format="%.2f",
                        label_visibility="collapsed",
                        placeholder="Enter actual %",
                    )
                with p_col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Save", key=f"save_{pred['id']}"):
                        if actual_val is None:
                            st.warning("Enter a value first.")
                        else:
                            verify_prediction(int(pred["id"]), float(actual_val))
                            st.success(f"✓ Prediction #{pred['id']} verified!", icon="✅")
                            st.rerun()

        # ── Verified predictions table ──────────────────────────────────────
        verified_df = pred_df[pred_df["verified"] == 1].copy()
        if not verified_df.empty:
            st.markdown("---")
            st.markdown("#### ✅ Verified Predictions History")

            display_cols = ["executive_name", "quarter", "metric",
                            "predicted_value", "direction", "actual_value"]
            display_cols = [c for c in display_cols if c in verified_df.columns]
            show_df = verified_df[display_cols].copy()
            show_df.columns = ["Executive", "Quarter", "Metric",
                                "Predicted", "Direction", "Actual"][:len(display_cols)]

            # Color-code by direction correctness
            def _highlight(row):
                pred = row.get("Predicted")
                actual = row.get("Actual")
                direction = str(row.get("Direction", "")).lower()
                if pred is None or actual is None:
                    return [""] * len(row)
                delta = actual - pred
                if ((direction == "up" and delta > 0) or
                        (direction == "down" and delta < 0) or
                        (direction == "stable" and abs(delta) <= abs(pred) * 0.05)):
                    return ["background-color: rgba(16,185,129,0.1)"] * len(row)
                return ["background-color: rgba(239,68,68,0.1)"] * len(row)

            st.dataframe(
                show_df.style.apply(_highlight, axis=1),
                use_container_width=True,
                height=350,
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: Review Queue
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### ⚖️ HARD Contradiction Review Queue")
    st.markdown(
        "<p style='color:#64748b;font-size:0.9rem;margin-top:-8px;margin-bottom:20px;'>"
        "Human-in-the-loop review queue for candidate HARD contradictions. "
        "Approval deducts 20 credibility points; rejection preserves executive credibility."
        "</p>",
        unsafe_allow_html=True,
    )

    pending_df = fetch_pending_reviews()
    history_df = fetch_review_history()

    if selected_exec_id is not None and not pending_df.empty and "executive_name" in pending_df.columns:
        sel_name = exec_df[exec_df["id"] == selected_exec_id]["name"].iloc[0] if not exec_df.empty else ""
        if sel_name:
            pending_df = pending_df[pending_df["executive_name"] == sel_name]

    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        stat_card("Pending Reviews", len(pending_df), "⏳", color="#f59e0b")
    with r_col2:
        stat_card("History Records", len(history_df), "📜", color="#3b82f6")

    st.markdown("<br>", unsafe_allow_html=True)

    if pending_df.empty:
        st.success("🎉 No pending HARD contradiction candidates awaiting review!", icon="✅")
    else:
        st.markdown(f"#### ⏳ Candidates Awaiting Decision ({len(pending_df)})")

        for _, item in pending_df.iterrows():
            c_id = int(item["contradiction_id"])
            thread_id = item.get("graph_thread_id", f"hard_{item['statement_a_id']}_{item['statement_b_id']}_v1_0")
            nli_score = float(item.get("nli_score", 0.0))
            llm_v = item.get("llm_verdict")
            llm_c = item.get("llm_confidence")
            llm_exp = item.get("llm_explanation")
            src = item.get("decision_source", "NLI")

            llm_info = ""
            if llm_v:
                conf_str = f"{float(llm_c)*100:.0f}%" if llm_c is not None else "N/A"
                llm_info = f" | LLM Verdict: <strong style='color:#f59e0b;'>{str(llm_v).upper()}</strong> ({conf_str} confidence) — <em>{llm_exp}</em>"

            st.markdown(f"""
<div style="background:#1a2236;border:1px solid #334155;border-left:4px solid #f59e0b;border-radius:12px;padding:20px;margin-bottom:16px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div>
            <span style="color:#f1f5f9;font-weight:700;font-size:1.05rem;">{item.get('executive_name','?')} ({item.get('executive_role','?')})</span>
            <span style="color:#94a3b8;font-size:0.85rem;margin-left:8px;">— {item.get('company_name','?')}</span>
        </div>
        <span style="background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b44;border-radius:20px;padding:3px 12px;font-size:0.75rem;font-weight:700;">
            PENDING REVIEW (Source: {src})
        </span>
    </div>
    
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
        <div style="background:#0d1526;border-radius:8px;padding:12px;border:1px solid #1e293b;">
            <p style="color:#94a3b8;font-size:0.75rem;font-weight:600;margin:0 0 4px 0;">📅 STATEMENT A ({item.get('quarter_a','')} FY{item.get('year_a','')})</p>
            <p style="color:#f1f5f9;font-size:0.88rem;margin:0;line-height:1.5;">"{item.get('statement_a_text','')}"</p>
        </div>
        <div style="background:#0d1526;border-radius:8px;padding:12px;border:1px solid #ef444444;">
            <p style="color:#94a3b8;font-size:0.75rem;font-weight:600;margin:0 0 4px 0;">📅 STATEMENT B ({item.get('quarter_b','')} FY{item.get('year_b','')})</p>
            <p style="color:#f1f5f9;font-size:0.88rem;margin:0;line-height:1.5;">"{item.get('statement_b_text','')}"</p>
        </div>
    </div>
    
    <div style="background:#0a0e1a;border-radius:8px;padding:10px 14px;border:1px solid #1e293b;margin-bottom:14px;font-size:0.82rem;color:#94a3b8;">
        <strong>Evidence:</strong> NLI Contradiction Probability = <span style="color:#ef4444;font-weight:700;">{nli_score:.2f}</span>
        {llm_info}
        <br><span style="color:#64748b;">Expected Penalty on Approval: -20 points</span>
    </div>
</div>
""", unsafe_allow_html=True)

            rc1, rc2, rc3, rc4 = st.columns([2, 2, 1, 1])
            with rc1:
                rev_name = st.text_input("Reviewer Name", value="Audit Manager", key=f"rev_name_{c_id}")
            with rc2:
                rev_notes = st.text_input("Review Notes", placeholder="Reason for decision…", key=f"rev_notes_{c_id}")
            with rc3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✅ Confirm", key=f"confirm_{c_id}"):
                    submit_review_decision(thread_id, c_id, approved=True, reviewer_name=rev_name, review_notes=rev_notes)
                    st.success(f"Candidate #{c_id} APPROVED! Credibility penalty applied.", icon="✅")
                    st.rerun()
            with rc4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("❌ Reject", key=f"reject_{c_id}"):
                    submit_review_decision(thread_id, c_id, approved=False, reviewer_name=rev_name, review_notes=rev_notes)
                    st.info(f"Candidate #{c_id} REJECTED.", icon="ℹ️")
                    st.rerun()

            st.markdown("---")

    # ── History ───────────────────────────────────────────────────────────────
    if not history_df.empty:
        st.markdown("#### 📜 Review Decision History")
        h_cols = ["contradiction_id", "company_name", "executive_name", "review_status",
                  "reviewer_name", "reviewed_at", "decision_source", "review_notes"]
        h_cols = [c for c in h_cols if c in history_df.columns]
        h_show = history_df[h_cols].copy()
        st.dataframe(h_show, use_container_width=True, height=260)
