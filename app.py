"""
RFQ Digital Twin Dashboard
===========================
Combines:
  1. Historical Overview - what actually happened in the last 3 months
  2. Live RFQ Scorer - run a single hypothetical RFQ through all 4 trained models
  3. What-If Simulation - the actual "digital twin" layer: a discrete-event queue
     model advances a clock, holds real people for real durations, and carries
     backlog forward across weeks. Simulated RFQs are scored by your REAL trained
     models as they move, so changing arrival volume or headcount shows how the
     learned system would respond.
  4. Turnaround Predictor - queue-aware estimate for one new RFQ
  5. Live Intake Simulator - simulated intake event, or live Airtable read

Run locally with:  streamlit run app.py
Requires these files in the same folder: rfq_sim.py, rfq_dataset.csv,
resource_pool.csv, complexity_model.pkl, delay_risk_model.pkl,
win_probability_model.pkl, priority_score_model.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
try:
    from pyairtable import Api
    AIRTABLE_AVAILABLE = True
except ImportError:
    AIRTABLE_AVAILABLE = False
import matplotlib.pyplot as plt

# Discrete-event queue engine (rfq_sim.py must sit in this same folder)
from rfq_sim import simulate, BASELINE_RESOURCES, STAGE_LABELS

# Make matplotlib charts match the blueprint theme instead of rendering white
plt.rcParams.update({
    "figure.facecolor": "#16283F",
    "axes.facecolor": "#16283F",
    "axes.edgecolor": "#2A4A6E",
    "axes.labelcolor": "#F2F4F7",
    "text.color": "#F2F4F7",
    "xtick.color": "#8FA3BD",
    "ytick.color": "#8FA3BD",
    "grid.color": "#2A4A6E",
    "font.family": "sans-serif",
    "legend.facecolor": "#16283F",
    "legend.edgecolor": "#2A4A6E",
    "legend.labelcolor": "#F2F4F7",
})

st.set_page_config(page_title="RFQ Digital Twin", layout="wide")

# ---------------------------------------------------------------------------
# Visual theme: engineering blueprint / schematic design system
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bp-bg: #0F1F33;
    --bp-panel: #16283F;
    --bp-panel-light: #1C3350;
    --bp-line: #2A4A6E;
    --bp-brass: #E0A855;
    --bp-brass-bright: #FFC670;
    --bp-cyan: #5FD8E8;
    --bp-text: #F2F4F7;
    --bp-text-dim: #8FA3BD;
}

.stApp {
    background-color: var(--bp-bg);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--bp-text);
}

h1, h2, h3, h4 {
    font-family: 'Fredoka', sans-serif !important;
    color: var(--bp-brass-bright) !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

h1 {
    font-size: 2.6rem !important;
    border-bottom: 3px solid var(--bp-brass);
    padding-bottom: 0.5rem;
    text-shadow: 0 0 18px rgba(255, 198, 112, 0.35);
}

h2, h3 {
    font-size: 1.5rem !important;
}

p, span, div, label {
    font-family: 'Inter', sans-serif;
}

/* Section divider styled as a clean blueprint rule */
hr {
    border: none;
    height: 1px;
    background-color: var(--bp-line);
    margin: 1.5rem 0;
}

/* Metric cards styled like schematic readouts */
[data-testid="stMetric"] {
    background: var(--bp-panel);
    border: 1px solid var(--bp-line);
    border-top: 2px solid var(--bp-brass);
    border-radius: 6px;
    padding: 1rem 1.1rem;
}

[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--bp-text-dim) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

[data-testid="stMetricValue"] {
    font-family: 'Fredoka', sans-serif !important;
    color: var(--bp-cyan) !important;
    font-weight: 600 !important;
}

/* Tabs styled like a clean instrument panel selector */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: var(--bp-text-dim) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--bp-brass-bright) !important;
    border-bottom: 2px solid var(--bp-brass) !important;
}

/* Buttons: brass fill, cyan glow on hover */
.stButton > button {
    background-color: var(--bp-brass);
    color: #0F1F33;
    font-family: 'Fredoka', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 4px;
}
.stButton > button:hover {
    background-color: var(--bp-brass-bright);
    box-shadow: 0 0 0 2px var(--bp-cyan);
    color: #0F1F33;
}

/* Sidebar / containers */
section[data-testid="stSidebar"] {
    background-color: var(--bp-panel);
    border-right: 1px solid var(--bp-line);
}

/* Dataframes / tables */
[data-testid="stDataFrame"] {
    border: 1px solid var(--bp-line);
    border-radius: 6px;
}

/* Progress bar in cyan */
.stProgress > div > div > div {
    background-color: var(--bp-cyan);
}

/* Captions in dimmer tone */
[data-testid="stCaptionContainer"] {
    color: var(--bp-text-dim) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Input widgets */
.stSelectbox, .stNumberInput, .stSlider {
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data and models (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("rfq_dataset.csv", parse_dates=[
        "Date_Received", "Due_Date", "Doc_Review_Start", "Doc_Review_End",
        "Eng_Review_Start", "Eng_Review_End", "Quote_Start", "Quote_End",
        "Approval_Start", "Approval_End", "Submit_Date"
    ])
    return df

@st.cache_resource
def load_models():
    models = {}
    for name, filename in [
        ("complexity", "complexity_model.pkl"),
        ("delay_risk", "delay_risk_model.pkl"),
        ("win_probability", "win_probability_model.pkl"),
        ("priority", "priority_score_model.pkl"),
    ]:
        try:
            with open(filename, "rb") as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            models[name] = None
    return models

@st.cache_data(ttl=300)
def load_airtable_rfqs():
    """Shared Airtable loader - used by both the Turnaround Predictor's
    auto-fill option and the Live Intake Simulator's Live mode."""
    api = Api(st.secrets["AIRTABLE_TOKEN"])
    table = api.table(st.secrets["AIRTABLE_BASE_ID"], "RFQs")
    records = table.all()
    rows = []
    for r in records:
        fields = r["fields"]
        fields["_record_id"] = r["id"]
        rows.append(fields)
    live_df = pd.DataFrame(rows)
    rename_map = {
        "RFQ ID": "RFQ_ID", "Customer Tier": "Customer_Tier", "Quote Type": "Quote_Type",
        "Rough Value Estimate": "Rough_Value_Estimate", "Current Stage": "Current_Stage",
    }
    return live_df.rename(columns={k: v for k, v in rename_map.items() if k in live_df.columns})

def get_airtable_backlog():
    """Returns (success, backlog_dict_or_error_message).
    backlog_dict has: backlog_eng, backlog_quote, backlog_approval, total_in_progress, has_stage_detail
    """
    if not AIRTABLE_AVAILABLE:
        return False, "pyairtable isn't installed. Add it to requirements.txt and reboot."
    if "AIRTABLE_TOKEN" not in st.secrets or "AIRTABLE_BASE_ID" not in st.secrets:
        return False, "Airtable secrets aren't configured (see the Live Intake Simulator tab for setup steps)."
    try:
        live_df = load_airtable_rfqs()
    except Exception as e:
        return False, f"Couldn't connect to Airtable: {e}"

    if live_df.empty:
        return False, "Connected, but the RFQs table is empty."

    if "Status" in live_df.columns:
        total_in_progress = int((live_df["Status"] == "In Progress").sum())
    else:
        total_in_progress = 0

    if "Current_Stage" in live_df.columns:
        stage_counts = live_df["Current_Stage"].value_counts()
        backlog_eng = int(stage_counts.get("Engineering Review", 0))
        backlog_quote = int(stage_counts.get("Quoting", 0))
        backlog_approval = int(stage_counts.get("Internal Approval", 0))
        has_stage_detail = True
    else:
        # Fallback: split total in-progress evenly across the three stages
        split = total_in_progress / 3
        backlog_eng = backlog_quote = backlog_approval = round(split)
        has_stage_detail = False

    return True, {
        "backlog_eng": backlog_eng, "backlog_quote": backlog_quote,
        "backlog_approval": backlog_approval, "total_in_progress": total_in_progress,
        "has_stage_detail": has_stage_detail,
    }

try:
    df = load_data()
    data_ok = True
except FileNotFoundError:
    data_ok = False

models = load_models()
missing_models = [k for k, v in models.items() if v is None]

st.title("RFQ Digital Twin Dashboard")
st.caption("Contract manufacturing RFQ workflow — historical performance, live scoring, and AI-driven what-if simulation")

if not data_ok:
    st.error("rfq_dataset.csv not found in this folder. Place it alongside app.py and refresh.")
    st.stop()

if missing_models:
    st.warning(
        f"Missing model file(s): {', '.join(missing_models)}. "
        f"Those sections will be disabled until the .pkl files are in this folder."
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Historical Overview", "🔍 Live RFQ Scorer", "⚙️ What-If Simulation",
    "📅 Turnaround Predictor", "🔴 Live Intake Simulator"
])

# ===========================================================================
# TAB 1: HISTORICAL OVERVIEW
# ===========================================================================
with tab1:
    closed = df[df["Outcome"].isin(["Won", "Loss"])].copy()
    win_rate = (closed["Outcome"] == "Won").mean() if len(closed) else 0
    df["Total_Cycle_Days"] = (df["Submit_Date"] - df["Date_Received"]).dt.days
    df["Eng_Review_Days"] = (df["Eng_Review_End"] - df["Eng_Review_Start"]).dt.days
    completion_pct = (df["Status"].str.startswith("Closed")).mean()
    in_progress_count = (df["Status"] == "In Progress").sum()
    awaiting_decision_count = (df["Status"] == "Submitted - Awaiting Decision").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total RFQs (3 months)", len(df))
    col2.metric("Currently In Progress", int(in_progress_count),
                help="RFQs still moving through the workflow (not yet submitted) as of the end of this data period")
    col3.metric("Awaiting Customer Decision", int(awaiting_decision_count),
                help="Quotes already submitted, waiting to hear back Won/Loss")

    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Completion Rate", f"{completion_pct:.0%}",
        help="% of RFQs that reached a final Won/Loss decision within the period"
    )
    col5.metric("Win Rate", f"{win_rate:.0%}")
    col6.metric("Avg. Cycle Time", f"{df['Total_Cycle_Days'].mean():.1f} days")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("RFQs by Industry")
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        df["Industry"].value_counts().plot(kind="bar", ax=ax, color="#5FD8E8")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Pipeline Status")
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        status_counts = df["Status"].value_counts()
        status_counts.plot(kind="pie", ax=ax, autopct="%1.0f%%", ylabel="", textprops={"fontsize": 8})
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Win Rate by Complexity")
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        closed["Won_Flag"] = (closed["Outcome"] == "Won").astype(int)
        closed.groupby("Complexity")["Won_Flag"].mean().plot(kind="bar", ax=ax, color="#E0A855")
        ax.set_ylabel("Win Rate")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with right2:
        st.subheader("Eng Review Bottleneck — Weekly Avg")
        weekly_eng = df.set_index("Eng_Review_Start").resample("W")["Eng_Review_Days"].mean()
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        weekly_eng.plot(ax=ax, marker="o", markersize=4, color="#5FD8E8")
        ax.axvspan(pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-22"), color="#E0A855", alpha=0.2, label="Arrival spike")
        ax.set_ylabel("Days")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    st.caption("Watch the week of June 15: arrival volume spiked and Engineering Review durations climbed with it.")

# ===========================================================================
# TAB 2: LIVE RFQ SCORER
# ===========================================================================
with tab2:
    st.subheader("Score a hypothetical RFQ through all four models")
    st.caption(
        "Fill in what you'd know at intake — the models fill in the rest, "
        "the same way they would for a real incoming RFQ."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        industry = st.selectbox("Industry", ["Aerospace", "Defense", "Commercial", "Medical"])
        customer_tier = st.selectbox("Customer Tier", ["Strategic", "Standard", "Opportunistic"])
        quote_type = st.selectbox("Quote Type", ["New Quote", "Repeat Quote", "Revision Quote"])
    with c2:
        rough_value = st.number_input("Rough Value Estimate ($)", min_value=1000, value=100000, step=5000)
        num_processes = st.slider("Number of Manufacturing Processes", 1, 9, 4)
        tolerance_class = st.selectbox("Tolerance Class", ["Standard", "Tight", "Ultra-Tight"])
    with c3:
        certifications = st.selectbox(
            "Certifications Required",
            ["None", "ISO9001", "AS9100", "ITAR", "AS9100+ITAR", "ISO13485", "FDA", "ISO13485+FDA"]
        )
        special_materials = st.selectbox("Special Materials", ["No", "Yes"])
        num_parts = st.slider("Number of Unique Part Numbers", 1, 30, 5)
        supplier_quote_required = st.selectbox("Requires an Outsourced Supplier Quote?", ["No", "Yes"],
                                                 help="e.g. outsourced plating, exotic material sourcing, specialty process")

    if st.button("Score this RFQ", type="primary"):
        if models["priority"]:
            pr_input = pd.DataFrame([{"Customer_Tier": customer_tier, "Rough_Value_Estimate": rough_value, "Industry": industry}])
            priority_pred = models["priority"].predict(pr_input)[0]
        else:
            priority_pred = None

        if models["complexity"]:
            cx_input = pd.DataFrame([{
                "Industry": industry, "Customer_Tier": customer_tier, "Rough_Value_Estimate": rough_value,
                "Technical_Risk_Flag": "Yes" if tolerance_class == "Ultra-Tight" else "No",
                "Doc_Complete_Flag": "Complete", "Num_Manufacturing_Processes": num_processes,
                "Tolerance_Class": tolerance_class, "Certifications_Required": certifications,
                "Quote_Type": quote_type, "Special_Materials_Flag": special_materials,
                "Num_Unique_Part_Numbers": num_parts,
            }])
            complexity_pred = models["complexity"].predict(cx_input)[0]
        else:
            complexity_pred = "Medium"  # fallback assumption if model missing

        if models["delay_risk"]:
            dr_input = pd.DataFrame([{
                "Complexity": complexity_pred, "Review_Cycles": 1, "Days_Elapsed_At_Approval": 12,
                "Days_Remaining_At_Approval": 10, "Priority_Score": priority_pred if priority_pred else 50,
                "Industry": industry, "Supplier_Quote_Required": supplier_quote_required,
            }])
            delay_pred = models["delay_risk"].predict(dr_input)[0]
        else:
            delay_pred = None

        est_cost = rough_value * 0.65
        margin_pct = 25
        quoted_price = est_cost / (1 - margin_pct / 100)
        if models["win_probability"]:
            win_input = pd.DataFrame([{
                "Complexity": complexity_pred, "Customer_Tier": customer_tier, "Quote_Type": quote_type,
                "Quoted_Price": quoted_price, "Margin_Pct": margin_pct, "Industry": industry,
            }])
            win_pred = models["win_probability"].predict(win_input)[0]
        else:
            win_pred = None

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Priority", f"{priority_pred:.0f}" if priority_pred is not None else "N/A")
        r2.metric("Predicted Complexity", complexity_pred)
        r3.metric("Predicted Delay Risk", delay_pred if delay_pred is not None else "N/A")
        r4.metric("Predicted Win Probability", f"{win_pred:.0%}" if win_pred is not None else "N/A")

# ===========================================================================
# TAB 3: WHAT-IF SIMULATION (the digital twin core)
# ===========================================================================
with tab3:
    st.subheader("What-If Simulation")
    st.caption(
        "A discrete-event queue model runs RFQs through the workflow one at a time: each RFQ takes the "
        "first person who frees up, and waits when everyone at that stage is busy. Backlog carries forward "
        "week to week. Every simulated RFQ is scored by your REAL trained models as it moves — so this is "
        "the learned system responding to new conditions, not a generic queueing guess."
    )

    st.markdown("##### Adjust staffing across every stage, and arrival volume")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        num_pms = st.slider("Program Managers", 1, 20, BASELINE_RESOURCES["num_pms"],
                            help=f"Baseline: {BASELINE_RESOURCES['num_pms']}")
        num_doc_reviewers = st.slider("Documentation Reviewers", 1, 20, BASELINE_RESOURCES["num_doc_reviewers"],
                                      help=f"Baseline: {BASELINE_RESOURCES['num_doc_reviewers']}")
    with rc2:
        num_engineers = st.slider("Engineers", 1, 20, BASELINE_RESOURCES["num_engineers"],
                                   help=f"Baseline: {BASELINE_RESOURCES['num_engineers']}")
        num_analysts = st.slider("Quoting Analysts", 1, 20, BASELINE_RESOURCES["num_analysts"],
                                  help=f"Baseline: {BASELINE_RESOURCES['num_analysts']}")
    with rc3:
        num_approvers = st.slider("Approvers / Leadership", 1, 20, BASELINE_RESOURCES["num_approvers"],
                                   help=f"Baseline: {BASELINE_RESOURCES['num_approvers']}")
        arrival_multiplier = st.slider("Arrival Volume Multiplier", 0.5, 2.0, 1.0, step=0.05,
                                        help="1.0x = normal historical volume (~16 RFQs/week). "
                                             "Try 1.10x — a 10% increase is enough to tip the pipeline.")
    sc1, sc2 = st.columns(2)
    with sc1:
        num_weeks = st.slider("Weeks to Simulate", 4, 24, 12,
                              help="Longer runs let backlog build, which is where queueing effects show up. "
                                   "12+ weeks recommended.")
    with sc2:
        skip_eng_for_low = st.checkbox(
            "Route low-complexity quotes around full Engineering Review",
            help="Tests the recommendation directly: low-complexity RFQs skip the engineering queue entirely."
        )

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running the twin forward…"):
            baseline, base_sum = simulate(
                models=models, num_weeks=num_weeks, arrival_multiplier=1.0,
                skip_eng_for_low=False, seed=1, **BASELINE_RESOURCES
            )
            scenario, scen_sum = simulate(
                models=models, num_weeks=num_weeks, arrival_multiplier=arrival_multiplier,
                skip_eng_for_low=skip_eng_for_low, seed=1,
                num_pms=num_pms, num_doc_reviewers=num_doc_reviewers, num_engineers=num_engineers,
                num_analysts=num_analysts, num_approvers=num_approvers,
            )

        st.divider()

        # --- Stability warning: the single most important output ---
        bottleneck_util = scen_sum["utilization"][scen_sum["bottleneck_stage"]]
        worst = STAGE_LABELS[scen_sum["bottleneck_stage"]]
        if not scen_sum["stable"]:
            st.error(
                f"**{worst} is over capacity** ({bottleneck_util:.0%} utilization). Above 100%, work arrives "
                f"faster than it can be cleared and the backlog grows without limit — the cycle times below "
                f"are still rising when the simulation ends, so treat them as a floor, not a forecast."
            )
        elif bottleneck_util > 0.90:
            st.warning(
                f"**{worst} is running hot** ({bottleneck_util:.0%} utilization). Above ~90%, small increases "
                f"in volume cause large jumps in turnaround."
            )

        st.markdown(
            f"#### Baseline ({BASELINE_RESOURCES['num_pms']} PM / "
            f"{BASELINE_RESOURCES['num_doc_reviewers']} Doc / {BASELINE_RESOURCES['num_engineers']} Eng / "
            f"{BASELINE_RESOURCES['num_analysts']} Quoting / {BASELINE_RESOURCES["num_approvers"]} Approvers, "
            f"normal volume) vs. Your Scenario"
        )
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("RFQs Simulated", scen_sum["rfqs_simulated"],
                   delta=scen_sum["rfqs_simulated"] - base_sum["rfqs_simulated"])
        b2.metric("Avg. Cycle Time", f"{scen_sum['avg_cycle_days']:.1f} days",
                   delta=f"{scen_sum['avg_cycle_days'] - base_sum['avg_cycle_days']:+.1f} days",
                   delta_color="inverse")
        b3.metric("90th Percentile", f"{scen_sum['p90_cycle_days']:.1f} days",
                   delta=f"{scen_sum['p90_cycle_days'] - base_sum['p90_cycle_days']:+.1f} days",
                   delta_color="inverse",
                   help="9 in 10 quotes go out within this. The number you'd actually promise a customer.")
        b4.metric("% Missing 21-Day Target", f"{scen_sum['late_rate']:.0%}",
                   delta=f"{(scen_sum['late_rate'] - base_sum['late_rate']):+.0%}",
                   delta_color="inverse")

        b5, b6, b7 = st.columns(3)
        b5.metric("Avg. Time Spent Waiting", f"{scen_sum['avg_wait_days']:.1f} days",
                   help="Pure queue time — no one is working on the RFQ. This is the recoverable portion.")
        b6.metric("Constraint Stage", worst)
        high_delay = (scenario["delay_risk"] == "High").mean()
        base_high_delay = (baseline["delay_risk"] == "High").mean()
        b7.metric("% High Delay Risk (model)", f"{high_delay:.0%}",
                   delta=f"{(high_delay - base_high_delay):+.0%}", delta_color="inverse")

        # --- Utilization: where the capacity actually goes ---
        st.markdown("#### Capacity Utilization by Stage")
        util_df = pd.DataFrame({
            "Baseline": [base_sum["utilization"][s] for s in STAGE_LABELS],
            "Scenario": [scen_sum["utilization"][s] for s in STAGE_LABELS],
        }, index=[STAGE_LABELS[s] for s in STAGE_LABELS])

        ucol1, ucol2 = st.columns([3, 2])
        with ucol1:
            fig, ax = plt.subplots(figsize=(5, 3))
            util_df.plot(kind="barh", ax=ax, color=["#8FA3BD", "#5FD8E8"])
            ax.axvline(1.0, color="#E0A855", linestyle="--", linewidth=1.5)
            ax.set_xlabel("Share of available person-days used")
            ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with ucol2:
            st.dataframe(
                util_df.apply(lambda col: col.map(lambda v: f"{v:.0%}")),
                use_container_width=True
            )
            st.caption(
                "The dashed line is 100% — past it the queue grows without bound. The highest bar is your "
                "constraint; adding people anywhere else changes almost nothing, and once you relieve the "
                "constraint the next highest stage becomes the new one."
            )

        # --- Waiting vs working: the honest breakdown ---
        st.markdown("#### Where the Days Go — Waiting vs. Working")
        stage_cols = ["pm_wait_days", "doc_review_days", "eng_review_days", "quote_days", "approval_days"]
        stage_labels_list = ["PM Triage", "Doc Review", "Eng Review", "Quoting", "Approval"]
        stage_compare = pd.DataFrame({
            "Baseline": [baseline[c].mean() for c in stage_cols],
            "Scenario": [scenario[c].mean() for c in stage_cols],
        }, index=stage_labels_list)

        cchart1, cchart2 = st.columns([3, 2])
        with cchart1:
            fig, ax = plt.subplots(figsize=(5, 3))
            stage_compare.plot(kind="bar", ax=ax, color=["#8FA3BD", "#5FD8E8"])
            ax.set_ylabel("Avg. Days (queue + processing)")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with cchart2:
            st.dataframe(stage_compare.round(1), use_container_width=True)
            pct_waiting = (scen_sum["avg_wait_days"] / scen_sum["avg_cycle_days"]
                           if scen_sum["avg_cycle_days"] else 0)
            st.info(
                f"**{pct_waiting:.0%} of total turnaround is queue time** — RFQs sitting untouched, waiting "
                f"for a person to free up. That share is what capacity or routing changes can recover; the "
                f"rest is actual work content."
            )
            if scenario["supplier_wait_days"].mean() > 0.5:
                st.caption(
                    f"A further {scenario['supplier_wait_days'].mean():.1f} days on average is external "
                    f"supplier wait — internal staffing cannot touch it."
                )

        # --- Cycle time trend: shows backlog building ---
        st.markdown("#### Weekly Cycle Time Trend")
        fig, ax = plt.subplots(figsize=(7, 2.6))
        baseline.groupby("week")["total_cycle_days"].mean().plot(
            ax=ax, marker="o", markersize=4, label="Baseline", color="#8FA3BD")
        scenario.groupby("week")["total_cycle_days"].mean().plot(
            ax=ax, marker="o", markersize=4, label="Scenario", color="#5FD8E8")
        ax.axhline(21, color="#E0A855", linestyle="--", linewidth=1.2, label="21-day target")
        ax.set_ylabel("Days")
        ax.set_xlabel("Simulated Week")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "A rising line means backlog is accumulating faster than it clears — the pipeline hasn't reached "
            "steady state and turnaround is still deteriorating. A flat line means it has stabilized."
        )

        # --- Scenario comparison table: the leadership view ---
        with st.expander("Compare standard scenarios side by side"):
            st.caption("Each row is a full simulation run. This is the table to show leadership.")
            rows = []
            for label, kw in [
                ("Baseline", dict()),
                ("+10% volume, no action", dict(arrival_multiplier=1.10)),
                ("+10% volume, +1 engineer", dict(arrival_multiplier=1.10,
                                                  num_engineers=BASELINE_RESOURCES["num_engineers"] + 1)),
                ("+10% volume, low-complexity skips eng", dict(arrival_multiplier=1.10,
                                                                skip_eng_for_low=True)),
                ("+25% volume, no action", dict(arrival_multiplier=1.25)),
            ]:
                kwargs = {**BASELINE_RESOURCES, **kw}
                _, s = simulate(models=None, num_weeks=num_weeks, seed=1, **kwargs)
                rows.append({
                    "Scenario": label,
                    "Avg Days": round(s["avg_cycle_days"], 1),
                    "P90 Days": round(s["p90_cycle_days"], 1),
                    "% Late": f"{s['late_rate']:.0%}",
                    "Constraint": STAGE_LABELS[s["bottleneck_stage"]],
                    "Constraint Util": f"{s['utilization'][s['bottleneck_stage']]:.0%}",
                    "Stable": "Yes" if s["stable"] else "NO",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(
                "These rows run without model scoring so the comparison is fast — complexity is sampled "
                "from the historical mix instead of predicted."
            )

        st.caption(
            "Two things make this a digital twin rather than a report: the clock (RFQs occupy real people "
            "for real durations and queue when they can't) and the models (complexity, delay risk and win "
            "probability are predicted by the same trained models running in production)."
        )

# ===========================================================================
# TAB 4: TURNAROUND PREDICTOR (queue-aware, uses CURRENT backlog you enter)
# ===========================================================================
with tab4:
    st.subheader("Predicted Quote Turnaround for a New RFQ")
    st.caption(
        "This estimates when a new RFQ entering TODAY would realistically get its quote out, given what's "
        "already ahead of it in your current pipeline. Enter your live backlog numbers from Airtable below — "
        "the historical dataset is a closed record and can't tell us what's in the queue right now."
    )

    st.markdown("##### Step 1 — Current backlog (from your live Airtable base)")

    if "airtable_backlog_defaults" not in st.session_state:
        st.session_state.airtable_backlog_defaults = {"eng": 0, "quote": 0, "approval": 0}

    af1, af2 = st.columns([1, 3])
    with af1:
        pull_clicked = st.button("🔄 Pull backlog from Airtable")
    with af2:
        if pull_clicked:
            success, result = get_airtable_backlog()
            if success:
                st.session_state.airtable_backlog_defaults = {
                    "eng": result["backlog_eng"], "quote": result["backlog_quote"],
                    "approval": result["backlog_approval"],
                }
                if result["has_stage_detail"]:
                    st.success(
                        f"Pulled from Airtable: {result['total_in_progress']} RFQs in progress, "
                        f"split by Current Stage."
                    )
                else:
                    st.info(
                        f"Pulled {result['total_in_progress']} in-progress RFQs from Airtable, but no "
                        f"'Current Stage' field was found, so the count was split evenly across "
                        f"Engineering, Quoting, and Approval as an estimate. Add a 'Current Stage' "
                        f"field to your Airtable base for a precise per-stage breakdown."
                    )
            else:
                st.error(result)

    defaults = st.session_state.airtable_backlog_defaults
    bl1, bl2, bl3 = st.columns(3)
    with bl1:
        backlog_eng = st.number_input(
            "RFQs currently in/waiting for Engineering Review",
            min_value=0, value=defaults["eng"], step=1
        )
        num_engineers_live = st.number_input("Engineers currently available", min_value=1, value=2, step=1)
    with bl2:
        backlog_quote = st.number_input(
            "RFQs currently in/waiting for Quoting",
            min_value=0, value=defaults["quote"], step=1
        )
        num_analysts_live = st.number_input("Quoting analysts currently available", min_value=1, value=3, step=1)
    with bl3:
        backlog_approval = st.number_input(
            "RFQs currently in/waiting for Approval",
            min_value=0, value=defaults["approval"], step=1
        )
        num_approvers_live = st.number_input("Approvers currently available", min_value=1, value=2, step=1)

    st.markdown("##### Step 2 — New RFQ details")
    tp_elapsed = st.number_input(
        "Business days already elapsed since this RFQ was received (0 if brand new)",
        min_value=0, value=0, step=1,
        help="Leave at 0 for a brand-new RFQ. If this RFQ is already in progress, "
             "enter how many business days have passed since intake to see its completion %."
    )
    t1, t2, t3 = st.columns(3)
    with t1:
        tp_industry = st.selectbox("Industry", ["Aerospace", "Defense", "Commercial", "Medical"], key="tp_ind")
        tp_tier = st.selectbox("Customer Tier", ["Strategic", "Standard", "Opportunistic"], key="tp_tier")
        tp_quote_type = st.selectbox("Quote Type", ["New Quote", "Repeat Quote", "Revision Quote"], key="tp_qt")
    with t2:
        tp_value = st.number_input("Rough Value Estimate ($)", min_value=1000, value=100000, step=5000, key="tp_val")
        tp_processes = st.slider("Number of Manufacturing Processes", 1, 9, 4, key="tp_proc")
        tp_tolerance = st.selectbox("Tolerance Class", ["Standard", "Tight", "Ultra-Tight"], key="tp_tol")
    with t3:
        tp_certs = st.selectbox(
            "Certifications Required",
            ["None", "ISO9001", "AS9100", "ITAR", "AS9100+ITAR", "ISO13485", "FDA", "ISO13485+FDA"],
            key="tp_cert"
        )
        tp_materials = st.selectbox("Special Materials", ["No", "Yes"], key="tp_mat")
        tp_parts = st.slider("Number of Unique Part Numbers", 1, 30, 5, key="tp_parts")
    tp_supplier_required = st.selectbox("Requires an Outsourced Supplier Quote?", ["No", "Yes"], key="tp_supplier",
                                          help="e.g. outsourced plating, exotic material sourcing, specialty process — "
                                               "this is an EXTERNAL wait, independent of your internal staffing.")
    tp_supplier_days = 0
    if tp_supplier_required == "Yes":
        tp_supplier_days = st.number_input("Expected supplier response time (business days)", min_value=1, value=5, step=1,
                                             help="If unknown, use 5-7 days as a typical estimate, "
                                                  "or check your supplier's historical response time.")

    # Historical average per-RFQ processing time by complexity, used to convert
    # "how many RFQs are ahead of me" into "how many days that queue represents"
    HIST_STAGE_DAYS = {
        "Low":    {"eng": 1.5, "quote": 1.5, "approval": 2.0},
        "Medium": {"eng": 3.5, "quote": 2.5, "approval": 3.0},
        "High":   {"eng": 7.5, "quote": 4.5, "approval": 4.5},
    }

    if st.button("Predict Turnaround", type="primary"):
        # 1. Predict this RFQ's complexity with the real trained model
        if models["complexity"]:
            cx_input = pd.DataFrame([{
                "Industry": tp_industry, "Customer_Tier": tp_tier, "Rough_Value_Estimate": tp_value,
                "Technical_Risk_Flag": "Yes" if tp_tolerance == "Ultra-Tight" else "No",
                "Doc_Complete_Flag": "Complete", "Num_Manufacturing_Processes": tp_processes,
                "Tolerance_Class": tp_tolerance, "Certifications_Required": tp_certs,
                "Quote_Type": tp_quote_type, "Special_Materials_Flag": tp_materials,
                "Num_Unique_Part_Numbers": tp_parts,
            }])
            tp_complexity = models["complexity"].predict(cx_input)[0]
        else:
            tp_complexity = "Medium"

        qt_mult = {"New Quote": 1.0, "Repeat Quote": 0.45, "Revision Quote": 0.65}[tp_quote_type]

        # 2. Queue wait = (RFQs ahead of you / people available) x typical time each one occupies that stage
        eng_wait = (backlog_eng / num_engineers_live) * HIST_STAGE_DAYS[tp_complexity]["eng"]
        quote_wait = (backlog_quote / num_analysts_live) * HIST_STAGE_DAYS[tp_complexity]["quote"]
        approval_wait = (backlog_approval / num_approvers_live) * HIST_STAGE_DAYS[tp_complexity]["approval"]

        # 3. This RFQ's own processing time once it reaches the front of each queue
        own_eng = HIST_STAGE_DAYS[tp_complexity]["eng"] * qt_mult
        own_quote = HIST_STAGE_DAYS[tp_complexity]["quote"] * qt_mult + tp_supplier_days
        own_approval = HIST_STAGE_DAYS[tp_complexity]["approval"]
        own_doc = 1.5  # runs in parallel with Eng Review, rarely the bottleneck

        total_days = eng_wait + own_eng + quote_wait + own_quote + approval_wait + own_approval
        predicted_date = pd.Timestamp.today().normalize() + pd.tseries.offsets.BDay(int(round(total_days)))

        # 4. Predict delay risk and win probability for this RFQ using the real models
        priority_pred = 50
        if models["priority"]:
            pr_input = pd.DataFrame([{"Customer_Tier": tp_tier, "Rough_Value_Estimate": tp_value, "Industry": tp_industry}])
            priority_pred = models["priority"].predict(pr_input)[0]

        delay_pred = None
        if models["delay_risk"]:
            dr_input = pd.DataFrame([{
                "Complexity": tp_complexity, "Review_Cycles": 1,
                "Days_Elapsed_At_Approval": int(eng_wait + own_eng + quote_wait + own_quote),
                "Days_Remaining_At_Approval": 5, "Priority_Score": priority_pred, "Industry": tp_industry,
                "Supplier_Quote_Required": tp_supplier_required,
            }])
            delay_pred = models["delay_risk"].predict(dr_input)[0]

        win_pred = None
        if models["win_probability"]:
            est_cost = tp_value * 0.65
            quoted_price = est_cost / 0.75
            win_input = pd.DataFrame([{
                "Complexity": tp_complexity, "Customer_Tier": tp_tier, "Quote_Type": tp_quote_type,
                "Quoted_Price": quoted_price, "Margin_Pct": 25, "Industry": tp_industry,
            }])
            win_pred = models["win_probability"].predict(win_input)[0]

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Complexity", tp_complexity)
        r2.metric("Estimated Turnaround", f"{total_days:.1f} business days")
        r3.metric("Estimated Quote Date", predicted_date.strftime("%b %d, %Y"))

        completion_pct = float(np.clip(tp_elapsed / total_days, 0, 1)) if total_days > 0 else 0
        st.markdown(f"##### Completion: {completion_pct:.0%}")
        st.progress(completion_pct)
        if tp_elapsed > total_days:
            st.error(
                f"This RFQ has already exceeded its predicted turnaround ({tp_elapsed} days elapsed vs. "
                f"{total_days:.1f} predicted) — it's running late relative to the model's estimate."
            )
        else:
            st.caption(f"{tp_elapsed} of {total_days:.1f} predicted business days elapsed, "
                       f"~{max(0, total_days - tp_elapsed):.1f} days remaining.")

        r4, r5 = st.columns(2)
        r4.metric("Predicted Delay Risk", delay_pred if delay_pred else "N/A")
        r5.metric("Predicted Win Probability", f"{win_pred:.0%}" if win_pred is not None else "N/A")

        st.markdown("##### Breakdown")
        breakdown = pd.DataFrame({
            "Stage": ["Doc Review (parallel)", "Eng Review — queue wait", "Eng Review — this RFQ",
                      "Quoting — queue wait", "Quoting — this RFQ (incl. supplier wait)",
                      "Approval — queue wait", "Approval — this RFQ"],
            "Days": [own_doc, eng_wait, own_eng, quote_wait, own_quote, approval_wait, own_approval],
        })
        st.dataframe(breakdown, hide_index=True, use_container_width=True)

        if eng_wait > own_eng * 1.5:
            st.warning(
                f"Queue wait at Engineering Review ({eng_wait:.1f} days) exceeds this RFQ's own processing "
                f"time ({own_eng:.1f} days) — Engineering is the bottleneck for this quote's turnaround, "
                f"not the RFQ's own complexity."
            )
        if tp_supplier_required == "Yes" and tp_supplier_days >= 5:
            st.info(
                f"{tp_supplier_days} days of this turnaround depends on an outside supplier's response time — "
                f"this is an EXTERNAL delay that adding internal staff can't fix. Consider whether a "
                f"preliminary/budgetary quote could go out to the customer before the supplier quote is final."
            )

# ===========================================================================
# TAB 5: LIVE INTAKE SIMULATOR (simulated demo OR live Airtable connection)
# ===========================================================================
with tab5:
    st.subheader("Live Intake Simulator")

    intake_mode = st.radio(
        "Mode",
        ["🔴 Simulated Intake", "🟢 Live from Airtable"],
        horizontal=True,
        help="Simulated mode demonstrates the concept with generated data. Live mode reads your real Airtable base."
    )

    # -----------------------------------------------------------------------
    # SIMULATED MODE (unchanged concept demo)
    # -----------------------------------------------------------------------
    if intake_mode == "🔴 Simulated Intake":
        st.markdown(
            "In a full production deployment, this is what would happen automatically: an **Airtable Automation** fires "
            "the moment a new RFQ record is created, sending its fields to an API endpoint that runs it through "
            "all four trained models instantly and writes the predictions back — no one has to open a notebook "
            "or manually re-score anything. This tab simulates that event so you can see the concept live."
        )
        st.caption("Click the button below to simulate a brand-new RFQ landing in Airtable right now.")

        if "intake_log" not in st.session_state:
            st.session_state.intake_log = []

        if st.button("🔴 Simulate New RFQ Created in Airtable", type="primary"):
            rng = np.random.default_rng()
            industries = ["Aerospace", "Defense", "Commercial", "Medical"]
            tiers = ["Strategic", "Standard", "Opportunistic"]
            quote_types = ["New Quote", "Repeat Quote", "Revision Quote"]

            industry = rng.choice(industries)
            tier = rng.choice(tiers, p=[0.25, 0.55, 0.20])
            quote_type = rng.choice(quote_types, p=[0.45, 0.35, 0.20])
            rough_value = float(np.round(rng.lognormal(mean=10.5, sigma=0.9), -2))
            num_processes = int(np.clip(rng.poisson(3.5) + 1, 1, 9))
            tolerance_class = rng.choice(["Standard", "Tight", "Ultra-Tight"], p=[0.35, 0.40, 0.25])
            certifications = rng.choice(["None", "AS9100", "ITAR", "ISO13485", "FDA"])
            special_materials = rng.choice(["Yes", "No"], p=[0.35, 0.65])
            num_parts = int(np.clip(rng.poisson(num_processes * 1.3) + 1, 1, 30))
            supplier_required = rng.choice(["Yes", "No"], p=[0.4, 0.6])
            rfq_id = f"RFQ-{rng.integers(3000, 9999)}"

            priority_pred = None
            if models["priority"]:
                pr_input = pd.DataFrame([{"Customer_Tier": tier, "Rough_Value_Estimate": rough_value, "Industry": industry}])
                priority_pred = models["priority"].predict(pr_input)[0]

            complexity_pred = "Medium"
            if models["complexity"]:
                cx_input = pd.DataFrame([{
                    "Industry": industry, "Customer_Tier": tier, "Rough_Value_Estimate": rough_value,
                    "Technical_Risk_Flag": "Yes" if tolerance_class == "Ultra-Tight" else "No",
                    "Doc_Complete_Flag": "Complete", "Num_Manufacturing_Processes": num_processes,
                    "Tolerance_Class": tolerance_class, "Certifications_Required": certifications,
                    "Quote_Type": quote_type, "Special_Materials_Flag": special_materials,
                    "Num_Unique_Part_Numbers": num_parts,
                }])
                complexity_pred = models["complexity"].predict(cx_input)[0]

            delay_pred = None
            if models["delay_risk"]:
                dr_input = pd.DataFrame([{
                    "Complexity": complexity_pred, "Review_Cycles": 0, "Days_Elapsed_At_Approval": 0,
                    "Days_Remaining_At_Approval": 20, "Priority_Score": priority_pred if priority_pred else 50,
                    "Industry": industry, "Supplier_Quote_Required": supplier_required,
                }])
                delay_pred = models["delay_risk"].predict(dr_input)[0]

            win_pred = None
            if models["win_probability"]:
                est_cost = rough_value * 0.65
                quoted_price = est_cost / 0.75
                win_input = pd.DataFrame([{
                    "Complexity": complexity_pred, "Customer_Tier": tier, "Quote_Type": quote_type,
                    "Quoted_Price": quoted_price, "Margin_Pct": 25, "Industry": industry,
                }])
                win_pred = models["win_probability"].predict(win_input)[0]

            st.session_state.intake_log.insert(0, {
                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                "RFQ_ID": rfq_id, "Industry": industry, "Customer_Tier": tier, "Quote_Type": quote_type,
                "Rough_Value": f"${rough_value:,.0f}", "Priority": f"{priority_pred:.0f}" if priority_pred is not None else "N/A",
                "Complexity": complexity_pred, "Delay_Risk": delay_pred if delay_pred else "N/A",
                "Win_Probability": f"{win_pred:.0%}" if win_pred is not None else "N/A",
                "Supplier_Quote": supplier_required,
            })

            st.success(f"New RFQ **{rfq_id}** detected and scored in real time — see the feed below.")

        if st.session_state.intake_log:
            st.markdown("##### Live Intake Feed (most recent first)")
            st.dataframe(pd.DataFrame(st.session_state.intake_log), hide_index=True, use_container_width=True)
            if st.button("Clear feed"):
                st.session_state.intake_log = []
                st.rerun()
        else:
            st.info("No simulated intake events yet — click the button above to generate one.")

    # -----------------------------------------------------------------------
    # LIVE AIRTABLE MODE
    # -----------------------------------------------------------------------
    else:
        st.caption(
            "Reads real RFQ records from your Airtable base, scores each one with your trained models, "
            "and lets you drill into any single RFQ for an estimated turnaround."
        )

        if not AIRTABLE_AVAILABLE:
            st.error("The `pyairtable` package isn't installed. Add `pyairtable` to requirements.txt and reboot the app.")
            st.stop()

        if "AIRTABLE_TOKEN" not in st.secrets or "AIRTABLE_BASE_ID" not in st.secrets:
            st.warning(
                "Airtable credentials aren't configured yet. On Streamlit Cloud: go to your app -> **Settings -> Secrets** "
                "and add:\n\n"
                "```\nAIRTABLE_TOKEN = \"your_token_here\"\nAIRTABLE_BASE_ID = \"your_base_id_here\"\n```\n\n"
                "Never put the token directly in app.py or any file committed to GitHub — secrets set this way are "
                "stored securely by Streamlit and are never visible in your public repo."
            )
            st.stop()

        try:
            live_df = load_airtable_rfqs()
        except Exception as e:
            st.error(f"Couldn't connect to Airtable: {e}")
            st.stop()

        if live_df.empty:
            st.info("Connected to Airtable, but the RFQs table is empty.")
            st.stop()

        # Normalize expected column names (Airtable fields commonly use spaces)
        rename_map = {
            "RFQ ID": "RFQ_ID", "Industry": "Industry", "Customer Tier": "Customer_Tier",
            "Quote Type": "Quote_Type", "Rough Value Estimate": "Rough_Value_Estimate",
            "Status": "Status", "Complexity": "Complexity", "Outcome": "Outcome",
        }
        live_df = live_df.rename(columns={k: v for k, v in rename_map.items() if k in live_df.columns})

        # --- KPIs ---
        total_rfqs = len(live_df)
        in_progress_count = (live_df["Status"] == "In Progress").sum() if "Status" in live_df.columns else 0
        new_quote_count = (live_df["Quote_Type"] == "New Quote").sum() if "Quote_Type" in live_df.columns else 0
        won_count = (live_df["Outcome"] == "Won").sum() if "Outcome" in live_df.columns else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total RFQs (Airtable)", total_rfqs)
        k2.metric("Currently In Progress", int(in_progress_count))
        k3.metric("New Quotes", int(new_quote_count))
        k4.metric("Won", int(won_count))

        if "Rough_Value_Estimate" in live_df.columns:
            st.metric("Total Pipeline Value", f"${live_df['Rough_Value_Estimate'].sum():,.0f}")

        st.markdown("##### All RFQs")
        st.dataframe(live_df, use_container_width=True, hide_index=True)

        # --- Click into a specific RFQ for turnaround estimate ---
        st.markdown("##### Estimate Turnaround for a Specific RFQ")
        id_col = "RFQ_ID" if "RFQ_ID" in live_df.columns else "_record_id"
        selected_id = st.selectbox("Select an RFQ", live_df[id_col].astype(str).tolist())
        selected_row = live_df[live_df[id_col].astype(str) == selected_id].iloc[0]

        def safe_val(row, key, default):
            """Airtable fields that are blank come through as NaN/None even when the
            key exists, so a plain .get(key, default) won't catch them. This does."""
            val = row.get(key, default) if hasattr(row, "get") else default
            if val is None:
                return default
            try:
                if pd.isna(val):
                    return default
            except (TypeError, ValueError):
                pass
            return val

        HIST_STAGE_DAYS_LIVE = {
            "Low":    {"eng": 1.5, "quote": 1.5, "approval": 2.0},
            "Medium": {"eng": 3.5, "quote": 2.5, "approval": 3.0},
            "High":   {"eng": 7.5, "quote": 4.5, "approval": 4.5},
        }

        row_complexity = safe_val(selected_row, "Complexity", None)
        if not row_complexity:
            if models["complexity"]:
                cx_input = pd.DataFrame([{
                    "Industry": safe_val(selected_row, "Industry", "Commercial"),
                    "Customer_Tier": safe_val(selected_row, "Customer_Tier", "Standard"),
                    "Rough_Value_Estimate": safe_val(selected_row, "Rough_Value_Estimate", 50000),
                    "Technical_Risk_Flag": "No", "Doc_Complete_Flag": "Complete",
                    "Num_Manufacturing_Processes": 4, "Tolerance_Class": "Tight",
                    "Certifications_Required": "None",
                    "Quote_Type": safe_val(selected_row, "Quote_Type", "New Quote"),
                    "Special_Materials_Flag": "No", "Num_Unique_Part_Numbers": 8,
                }])
                row_complexity = models["complexity"].predict(cx_input)[0]
            else:
                row_complexity = "Medium"
            st.caption(f"Complexity not set on this record — predicted as **{row_complexity}** by the trained model.")

        qt_mult = {"New Quote": 1.0, "Repeat Quote": 0.45, "Revision Quote": 0.65}.get(
            safe_val(selected_row, "Quote_Type", "New Quote"), 1.0
        )

        # Backlog proxy: current in-progress RFQs elsewhere in the base, split evenly
        # across the three resource-constrained stages as a simplifying assumption
        backlog_other = max(0, in_progress_count - 1)
        default_eng, default_quote, default_approval = 2, 3, 2
        eng_wait = (backlog_other / 3 / default_eng) * HIST_STAGE_DAYS_LIVE[row_complexity]["eng"]
        quote_wait = (backlog_other / 3 / default_quote) * HIST_STAGE_DAYS_LIVE[row_complexity]["quote"]
        approval_wait = (backlog_other / 3 / default_approval) * HIST_STAGE_DAYS_LIVE[row_complexity]["approval"]

        own_eng = HIST_STAGE_DAYS_LIVE[row_complexity]["eng"] * qt_mult
        own_quote = HIST_STAGE_DAYS_LIVE[row_complexity]["quote"] * qt_mult
        own_approval = HIST_STAGE_DAYS_LIVE[row_complexity]["approval"]
        total_days = eng_wait + own_eng + quote_wait + own_quote + approval_wait + own_approval
        predicted_date = pd.Timestamp.today().normalize() + pd.tseries.offsets.BDay(int(round(total_days)))

        tr1, tr2, tr3 = st.columns(3)
        tr1.metric("Complexity", row_complexity)
        tr2.metric("Estimated Turnaround", f"{total_days:.1f} business days")
        tr3.metric("Estimated Quote Date", predicted_date.strftime("%b %d, %Y"))
        st.caption(
            "Backlog wait is estimated by splitting all other currently-in-progress RFQs evenly across "
            "Engineering, Quoting, and Approval — for a precise number tied to actual per-stage backlog, "
            "use the Turnaround Predictor tab instead."
        )
