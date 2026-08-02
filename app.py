"""
RFQ Digital Twin Dashboard
===========================
Combines:
  1. Historical Overview - what actually happened in the last 3 months
  2. Live RFQ Scorer - run a single hypothetical RFQ through all 4 trained models
  3. What-If Simulation - the actual "digital twin" layer: simulated RFQs are
     scored by your REAL trained models as they move through the workflow, so
     changing arrival volume or engineering headcount shows how the real,
     learned system would respond - not a generic queueing guess.

Run locally with:  streamlit run app.py
Requires these files in the same folder: rfq_dataset.csv, resource_pool.csv,
complexity_model.pkl, delay_risk_model.pkl, win_probability_model.pkl,
priority_score_model.pkl
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
    st.warning(f"Missing model file(s): {', '.join(missing_models)}. Those sections will be disabled until the .pkl files are in this folder.")

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
    col4.metric("Completion Rate", f"{completion_pct:.0%}", help="% of RFQs that reached a final Won/Loss decision within the period")
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
    st.caption("Fill in what you'd know at intake — the models fill in the rest, the same way they would for a real incoming RFQ.")

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
        certifications = st.selectbox("Certifications Required", ["None", "ISO9001", "AS9100", "ITAR", "AS9100+ITAR", "ISO13485", "FDA", "ISO13485+FDA"])
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
        "Simulated RFQs are generated with the same distributions as your historical data, then scored by "
        "your REAL trained models as they move through the workflow — this is what makes it a digital twin "
        "rather than a generic queueing simulation."
    )

    st.markdown("##### Adjust staffing across every stage, and arrival volume")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        num_pms = st.slider("Program Managers", 1, 6, 3, help="Baseline: 3 (from resource_pool.csv)")
        num_doc_reviewers = st.slider("Documentation Reviewers", 1, 6, 2, help="Baseline: 2")
    with rc2:
        num_engineers = st.slider("Engineers", 1, 6, 2, help="Baseline: 2")
        num_analysts = st.slider("Quoting Analysts", 1, 6, 3, help="Baseline: 3")
    with rc3:
        num_approvers = st.slider("Approvers / Leadership", 1, 6, 2, help="Baseline: 2")
        arrival_multiplier = st.slider("Arrival Volume Multiplier", 0.5, 3.0, 1.0, step=0.1,
                                        help="1.0x = normal historical volume. 2.0x = double the RFQs arriving.")
    num_weeks = st.slider("Weeks to Simulate", 4, 16, 8)

    BASELINE_RESOURCES = dict(num_pms=3, num_doc_reviewers=2, num_engineers=2, num_analysts=3, num_approvers=2)

    def simulate(num_pms, num_doc_reviewers, num_engineers, num_analysts, num_approvers,
                 arrival_multiplier, num_weeks, seed=1):
        rng = np.random.default_rng(seed)
        industries = ["Aerospace", "Defense", "Commercial", "Medical"]
        industry_mix = [0.28, 0.17, 0.35, 0.20]
        tiers = ["Strategic", "Standard", "Opportunistic"]
        tier_mix = [0.25, 0.55, 0.20]
        quote_types = ["New Quote", "Repeat Quote", "Revision Quote"]
        qt_mix = [0.45, 0.35, 0.20]

        base_weekly_rate = 16 * arrival_multiplier  # ~16/week matches historical baseline
        # separate queue trackers per stage - each resource type queues independently
        pm_queue_load, doc_queue_load, eng_queue_load, quote_queue_load, approval_queue_load = {}, {}, {}, {}, {}
        results = []

        def stage_wait(queue_load, num_people, week, penalty_weight=0.8):
            person = rng.integers(0, num_people)
            key = (person, week)
            penalty = queue_load.get(key, 0) * penalty_weight
            queue_load[key] = queue_load.get(key, 0) + 1
            return penalty

        for week in range(num_weeks):
            n_arrivals = rng.poisson(base_weekly_rate)
            for _ in range(n_arrivals):
                industry = rng.choice(industries, p=industry_mix)
                tier = rng.choice(tiers, p=tier_mix)
                quote_type = rng.choice(quote_types, p=qt_mix)
                rough_value = float(np.round(rng.lognormal(mean=10.5, sigma=0.9), -2))
                num_processes = int(np.clip(rng.poisson(3.5) + 1, 1, 9))
                tolerance_class = rng.choice(["Standard", "Tight", "Ultra-Tight"], p=[0.35, 0.40, 0.25])
                certifications = rng.choice(["None", "AS9100", "ITAR", "ISO13485", "FDA"])
                special_materials = rng.choice(["Yes", "No"], p=[0.35, 0.65])
                num_parts = int(np.clip(rng.poisson(num_processes * 1.3) + 1, 1, 30))

                # PM triage queue - every RFQ passes through PM review before Doc/Eng Review can start
                pm_wait = stage_wait(pm_queue_load, num_pms, week, penalty_weight=0.3)

                # score complexity with the REAL trained model
                if models["complexity"]:
                    cx_input = pd.DataFrame([{
                        "Industry": industry, "Customer_Tier": tier, "Rough_Value_Estimate": rough_value,
                        "Technical_Risk_Flag": "Yes" if tolerance_class == "Ultra-Tight" else "No",
                        "Doc_Complete_Flag": "Complete", "Num_Manufacturing_Processes": num_processes,
                        "Tolerance_Class": tolerance_class, "Certifications_Required": certifications,
                        "Quote_Type": quote_type, "Special_Materials_Flag": special_materials,
                        "Num_Unique_Part_Numbers": num_parts,
                    }])
                    complexity = models["complexity"].predict(cx_input)[0]
                else:
                    complexity = "Medium"

                qt_mult = {"New Quote": 1.0, "Repeat Quote": 0.45, "Revision Quote": 0.65}[quote_type]

                # Documentation Review (parallel branch, short but still resource-constrained)
                doc_wait = stage_wait(doc_queue_load, num_doc_reviewers, week, penalty_weight=0.3)
                doc_duration = max(0.5, 1.5 + doc_wait + rng.normal(0, 0.4))

                # Engineering Review
                base_eng_days = {"Low": 1.5, "Medium": 3.5, "High": 7.5}[complexity] * qt_mult
                eng_wait = stage_wait(eng_queue_load, num_engineers, week, penalty_weight=0.8)
                eng_duration = max(1, base_eng_days + eng_wait + rng.normal(0, 0.7))

                # Supplier quote wait - external delay, independent of internal staffing
                supplier_prob = 0.55 if special_materials == "Yes" else 0.20
                supplier_required = rng.choice(["Yes", "No"], p=[min(supplier_prob, 0.9), 1 - min(supplier_prob, 0.9)])
                supplier_wait = 0.0
                if supplier_required == "Yes":
                    base_wait = {"New Quote": 6.0, "Repeat Quote": 2.0, "Revision Quote": 3.5}[quote_type]
                    supplier_wait = max(0.5, base_wait + rng.normal(0, 2.0))

                # Quoting
                base_quote_days = {"Low": 1.5, "Medium": 2.5, "High": 4.5}[complexity] * qt_mult
                quote_wait = stage_wait(quote_queue_load, num_analysts, week, penalty_weight=0.6)
                quote_duration = max(0.5, base_quote_days + quote_wait + supplier_wait + rng.normal(0, 0.5))

                # Internal Approval
                base_approval_days = 2.5
                approval_wait = stage_wait(approval_queue_load, num_approvers, week, penalty_weight=0.5)
                approval_duration = max(1, base_approval_days + approval_wait + rng.normal(0, 0.4))

                total_cycle_days = pm_wait + max(doc_duration, eng_duration) + quote_duration + approval_duration

                # score delay risk + win probability with the REAL trained models
                priority_pred = 50
                if models["priority"]:
                    pr_input = pd.DataFrame([{"Customer_Tier": tier, "Rough_Value_Estimate": rough_value, "Industry": industry}])
                    priority_pred = models["priority"].predict(pr_input)[0]

                delay_risk = "Medium"
                if models["delay_risk"]:
                    dr_input = pd.DataFrame([{
                        "Complexity": complexity, "Review_Cycles": 1,
                        "Days_Elapsed_At_Approval": int(total_cycle_days - approval_duration),
                        "Days_Remaining_At_Approval": max(0, 20 - int(total_cycle_days - approval_duration)),
                        "Priority_Score": priority_pred, "Industry": industry,
                        "Supplier_Quote_Required": supplier_required,
                    }])
                    delay_risk = models["delay_risk"].predict(dr_input)[0]

                win_prob = 0.4
                if models["win_probability"]:
                    est_cost = rough_value * 0.65
                    quoted_price = est_cost / 0.75
                    win_input = pd.DataFrame([{
                        "Complexity": complexity, "Customer_Tier": tier, "Quote_Type": quote_type,
                        "Quoted_Price": quoted_price, "Margin_Pct": 25, "Industry": industry,
                    }])
                    win_prob = models["win_probability"].predict(win_input)[0]

                results.append({
                    "week": week, "industry": industry, "complexity": complexity,
                    "pm_wait_days": pm_wait, "doc_review_days": doc_duration, "eng_review_days": eng_duration,
                    "quote_days": quote_duration, "approval_days": approval_duration,
                    "supplier_wait_days": supplier_wait, "total_cycle_days": total_cycle_days,
                    "delay_risk": delay_risk, "win_probability": win_prob,
                })

        return pd.DataFrame(results)

    if st.button("Run Simulation", type="primary"):
        baseline = simulate(**BASELINE_RESOURCES, arrival_multiplier=1.0, num_weeks=num_weeks, seed=1)
        scenario = simulate(num_pms=num_pms, num_doc_reviewers=num_doc_reviewers, num_engineers=num_engineers,
                             num_analysts=num_analysts, num_approvers=num_approvers,
                             arrival_multiplier=arrival_multiplier, num_weeks=num_weeks, seed=1)

        st.divider()
        st.markdown("#### Baseline (3 PM / 2 Doc / 2 Eng / 3 Quoting / 2 Approvers, normal volume) vs. Your Scenario")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("RFQs Simulated", len(scenario), delta=f"{len(scenario) - len(baseline)}")
        b2.metric("Avg. Total Cycle Time", f"{scenario['total_cycle_days'].mean():.1f} days",
                   delta=f"{scenario['total_cycle_days'].mean() - baseline['total_cycle_days'].mean():.1f} days", delta_color="inverse")
        high_delay_pct = (scenario["delay_risk"] == "High").mean()
        base_high_delay_pct = (baseline["delay_risk"] == "High").mean()
        b3.metric("% High Delay Risk", f"{high_delay_pct:.0%}", delta=f"{(high_delay_pct - base_high_delay_pct):.0%}",
