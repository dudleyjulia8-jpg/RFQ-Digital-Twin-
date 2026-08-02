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
    --bp-brass-bright: #F2C275;
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
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}

h1 {
    border-bottom: 2px solid var(--bp-brass);
    padding-bottom: 0.5rem;
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
