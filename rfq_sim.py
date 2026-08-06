"""
rfq_sim.py -- discrete-event queue engine for the RFQ Digital Twin.

Drop this file next to app.py. It replaces the (person, week) penalty proxy with
a real clock: each person holds a "next free" timestamp, an RFQ starts a stage
only when both the RFQ is ready AND someone is free, and backlog carries forward
across weeks. That carryover is what produces the nonlinear cliff -- past ~100%
utilization the queue grows without bound, exactly as it does in real life.

Units are BUSINESS DAYS. Stage times below are PERSON-DAYS OF WORK per RFQ, not
elapsed calendar time. Supplier wait is elapsed-only and consumes nobody.

Depends only on numpy (already in your requirements).
"""

import numpy as np
import pandas as pd

DAYS_PER_WEEK = 5

# ---------------------------------------------------------------------------
# Baseline staffing. Recalibrated for stability at ~16 RFQs/week -- see the
# note at the bottom of this file for the arithmetic.
# ---------------------------------------------------------------------------
BASELINE_RESOURCES = dict(
    num_pms=2,
    num_doc_reviewers=5,
    num_engineers=10,
    num_analysts=7,
    num_approvers=2,
)

BASE_WEEKLY_ARRIVALS = 16          # ~215 RFQs per quarter, matching your dataset

# Person-days of work per RFQ at each stage.
WORK_DAYS = {
    "pm_review":  {"Low": 0.5, "Medium": 0.5, "High": 0.5},
    "doc_review": {"Low": 0.8, "Medium": 1.2, "High": 2.0},
    "eng_review": {"Low": 1.5, "Medium": 3.5, "High": 7.5},   # your original values
    "quoting":    {"Low": 1.5, "Medium": 2.5, "High": 4.5},   # your original values
    "approval":   {"Low": 0.5, "Medium": 0.5, "High": 0.5},
}

# Repeat and revision quotes are less work. Applies to eng review and quoting only.
QT_MULTIPLIER = {"New Quote": 1.0, "Repeat Quote": 0.45, "Revision Quote": 0.65}

STAGE_TO_RESOURCE = {
    "pm_review": "num_pms",
    "doc_review": "num_doc_reviewers",
    "eng_review": "num_engineers",
    "quoting": "num_analysts",
    "approval": "num_approvers",
}


def _claim(servers, ready_at, duration):
    """
    Give this RFQ to whoever frees up first.

    servers  -- array of times at which each person becomes available
    ready_at -- when the RFQ is ready to enter this stage
    returns  -- (finish_time, wait_days)

    The single important line is `start = max(ready_at, servers[idx])`.
    When the second term wins, the RFQ queued.
    """
    idx = int(np.argmin(servers))
    start = max(ready_at, servers[idx])
    finish = start + duration
    servers[idx] = finish
    return finish, start - ready_at


def simulate(models=None, num_weeks=8, arrival_multiplier=1.0, seed=1,
             skip_eng_for_low=False, promised_days=21, **resources):
    """
    Run the twin forward.

    models -- your loaded models dict, or None to skip scoring (faster, and
              lets you test the queue engine on its own).
    resources -- num_pms, num_doc_reviewers, num_engineers, num_analysts,
                 num_approvers. Anything omitted falls back to BASELINE_RESOURCES.

    Returns (DataFrame of one row per RFQ, summary dict).
    """
    cap = {**BASELINE_RESOURCES, **resources}
    rng = np.random.default_rng(seed)

    horizon = num_weeks * DAYS_PER_WEEK
    servers = {stage: np.zeros(cap[STAGE_TO_RESOURCE[stage]]) for stage in WORK_DAYS}
    busy = {stage: 0.0 for stage in WORK_DAYS}

    # ---- arrivals: Poisson process, so gaps are exponential ----
    rate_per_day = (BASE_WEEKLY_ARRIVALS * arrival_multiplier) / DAYS_PER_WEEK
    arrivals, clock = [], 0.0
    while True:
        clock += rng.exponential(1.0 / rate_per_day)
        if clock >= horizon:
            break
        arrivals.append(clock)

    industries = ["Aerospace", "Defense", "Commercial", "Medical"]
    industry_mix = [0.28, 0.17, 0.35, 0.20]
    tiers = ["Strategic", "Standard", "Opportunistic"]
    tier_mix = [0.25, 0.55, 0.20]
    quote_types = ["New Quote", "Repeat Quote", "Revision Quote"]
    qt_mix = [0.45, 0.35, 0.20]

    results = []
    for arrival in arrivals:
        industry = rng.choice(industries, p=industry_mix)
        tier = rng.choice(tiers, p=tier_mix)
        quote_type = rng.choice(quote_types, p=qt_mix)
        rough_value = float(np.round(rng.lognormal(mean=10.5, sigma=0.9), -2))
        num_processes = int(np.clip(rng.poisson(3.5) + 1, 1, 9))
        tolerance_class = rng.choice(["Standard", "Tight", "Ultra-Tight"], p=[0.35, 0.40, 0.25])
        certifications = rng.choice(["None", "AS9100", "ITAR", "ISO13485", "FDA"])
        special_materials = rng.choice(["Yes", "No"], p=[0.35, 0.65])
        num_parts = int(np.clip(rng.poisson(num_processes * 1.3) + 1, 1, 30))

        # --- complexity from the REAL trained model, before any timing ---
        if models and models.get("complexity"):
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
            complexity = rng.choice(["Low", "Medium", "High"], p=[0.40, 0.35, 0.25])

        qt_mult = QT_MULTIPLIER[quote_type]
        waits = {}

        def work(stage, scale=1.0):
            base = WORK_DAYS[stage][complexity] * scale
            return max(0.1, base * rng.normal(1.0, 0.20))   # 20% variability

        # 1. PM triage
        d = work("pm_review"); busy["pm_review"] += d
        t, waits["pm_review"] = _claim(servers["pm_review"], arrival, d)

        # 2. Doc Review and Eng Review run in PARALLEL off PM Review
        d = work("doc_review"); busy["doc_review"] += d
        doc_done, waits["doc_review"] = _claim(servers["doc_review"], t, d)

        if skip_eng_for_low and complexity == "Low":
            eng_done, waits["eng_review"] = t, 0.0
            eng_touch = 0.0
        else:
            eng_touch = work("eng_review", qt_mult); busy["eng_review"] += eng_touch
            eng_done, waits["eng_review"] = _claim(servers["eng_review"], t, eng_touch)

        t = max(doc_done, eng_done)      # slower branch governs

        # 3. Supplier quote: EXTERNAL elapsed wait, consumes none of your people
        supplier_prob = 0.55 if special_materials == "Yes" else 0.20
        supplier_required = "Yes" if rng.random() < supplier_prob else "No"
        supplier_wait = 0.0
        if supplier_required == "Yes":
            base_wait = {"New Quote": 6.0, "Repeat Quote": 2.0, "Revision Quote": 3.5}[quote_type]
            supplier_wait = max(0.5, base_wait + rng.normal(0, 2.0))
            t += supplier_wait

        # 4. Quoting
        d = work("quoting", qt_mult); busy["quoting"] += d
        t, waits["quoting"] = _claim(servers["quoting"], t, d)

        # 5. Internal approval
        d = work("approval"); busy["approval"] += d
        t, waits["approval"] = _claim(servers["approval"], t, d)

        total_cycle_days = t - arrival

        # --- score delay risk / priority / win with the REAL trained models ---
        priority_pred = 50
        if models and models.get("priority"):
            pr_input = pd.DataFrame([{"Customer_Tier": tier,
                                      "Rough_Value_Estimate": rough_value,
                                      "Industry": industry}])
            priority_pred = models["priority"].predict(pr_input)[0]

        delay_risk = "Medium"
        if models and models.get("delay_risk"):
            elapsed_at_approval = int(total_cycle_days - waits["approval"])
            dr_input = pd.DataFrame([{
                "Complexity": complexity, "Review_Cycles": 1,
                "Days_Elapsed_At_Approval": elapsed_at_approval,
                "Days_Remaining_At_Approval": max(0, promised_days - elapsed_at_approval),
                "Priority_Score": priority_pred, "Industry": industry,
                "Supplier_Quote_Required": supplier_required,
            }])
            delay_risk = models["delay_risk"].predict(dr_input)[0]

        win_prob = 0.4
        if models and models.get("win_probability"):
            quoted_price = (rough_value * 0.65) / 0.75
            win_input = pd.DataFrame([{
                "Complexity": complexity, "Customer_Tier": tier, "Quote_Type": quote_type,
                "Quoted_Price": quoted_price, "Margin_Pct": 25, "Industry": industry,
            }])
            win_prob = models["win_probability"].predict(win_input)[0]

        results.append({
            "week": int(arrival // DAYS_PER_WEEK),
            "arrival_day": round(arrival, 2),
            "industry": industry, "complexity": complexity, "quote_type": quote_type,
            "pm_wait_days": waits["pm_review"],
            "doc_review_days": waits["doc_review"] + WORK_DAYS["doc_review"][complexity],
            "eng_review_days": waits["eng_review"] + eng_touch,
            "eng_wait_days": waits["eng_review"],
            "quote_days": waits["quoting"] + WORK_DAYS["quoting"][complexity] * qt_mult,
            "approval_days": waits["approval"] + WORK_DAYS["approval"][complexity],
            "supplier_wait_days": supplier_wait,
            "total_wait_days": sum(waits.values()),
            "total_cycle_days": total_cycle_days,
            "late": total_cycle_days > promised_days,
            "delay_risk": delay_risk,
            "win_probability": win_prob,
        })

    sim_df = pd.DataFrame(results)

    util = {stage: busy[stage] / (cap[STAGE_TO_RESOURCE[stage]] * horizon)
            for stage in WORK_DAYS}
    bottleneck = max(util, key=util.get)
    summary = {
        "rfqs_simulated": len(sim_df),
        "avg_cycle_days": float(sim_df["total_cycle_days"].mean()) if len(sim_df) else 0.0,
        "p90_cycle_days": float(sim_df["total_cycle_days"].quantile(0.9)) if len(sim_df) else 0.0,
        "avg_wait_days": float(sim_df["total_wait_days"].mean()) if len(sim_df) else 0.0,
        "late_rate": float(sim_df["late"].mean()) if len(sim_df) else 0.0,
        "utilization": util,
        "bottleneck_stage": bottleneck,
        "stable": util[bottleneck] < 1.0,
    }
    return sim_df, summary


STAGE_LABELS = {
    "pm_review": "PM Triage", "doc_review": "Doc Review", "eng_review": "Eng Review",
    "quoting": "Quoting", "approval": "Approval",
}


if __name__ == "__main__":
    print("Queue engine self-test (no models -- random complexity)\n")
    header = f"{'Scenario':<34}{'Avg':>7}{'P90':>7}{'Late':>7}{'EngUtil':>9}{'Stable':>8}"
    print(header); print("-" * len(header))
    for label, kw in [
        ("Baseline (16 RFQ/wk)",            dict()),
        ("+10% volume",                     dict(arrival_multiplier=1.10)),
        ("+10% volume, +1 engineer",        dict(arrival_multiplier=1.10, num_engineers=11)),
        ("+10% volume, low skips eng",      dict(arrival_multiplier=1.10, skip_eng_for_low=True)),
        ("+25% volume, no action",          dict(arrival_multiplier=1.25)),
        ("-1 engineer, normal volume",      dict(num_engineers=9)),
    ]:
        _, s = simulate(num_weeks=16, **kw)
        print(f"{label:<34}{s['avg_cycle_days']:>7.1f}{s['p90_cycle_days']:>7.1f}"
              f"{s['late_rate']:>6.0%}{s['utilization']['eng_review']:>9.0%}"
              f"{str(s['stable']):>8}")

    print("\nBaseline utilization by stage:")
    _, s = simulate(num_weeks=16)
    for stage, u in sorted(s["utilization"].items(), key=lambda kv: -kv[1]):
        print(f"  {STAGE_LABELS[stage]:<12} {u:>5.0%}"
              + ("   <-- BOTTLENECK" if stage == s["bottleneck_stage"] else ""))
