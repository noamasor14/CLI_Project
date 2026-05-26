import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive Load Index Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color constants ───────────────────────────────────────────────────────────
COLOR_LOW    = "#2ecc71"
COLOR_MED    = "#f39c12"
COLOR_HIGH   = "#e74c3c"
COLOR_MAP    = {"Low": COLOR_LOW, "Medium": COLOR_MED, "High": COLOR_HIGH}

def color_cli(value):
    try:
        v = float(value)
    except Exception:
        return ""

    if v <= 40:
        return "background-color:#e6f4ea;color:#137333"

    elif v <= 70:
        return "background-color:#fff4e5;color:#b06000"

    else:
        return "background-color:#fdecea;color:#a50e0e"

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("outputs", "cli_dashboard_data.csv")
FEAT_PATH    = os.path.join("outputs", "v7_feature_importance_best_variant.csv")
COMP_PATH    = os.path.join("outputs", "v7_model_comparison.csv")

@st.cache_data
def load_data():
    df   = pd.read_csv(DATA_PATH)
    feat = pd.read_csv(FEAT_PATH)
    comp = pd.read_csv(COMP_PATH)
    return df, feat, comp

df, feat_df, comp_df = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🧠 CLI Dashboard")
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Participant View", "Task / Segment Comparison", "Model Results", "Project Information"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption("Final Project · Industrial Engineering & Management")

# ── Helper: CLI zone background shapes ───────────────────────────────────────
def cli_zone_shapes():
    return [
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0,   y1=40,  fillcolor=COLOR_LOW,  opacity=0.08, line_width=0),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=40,  y1=70,  fillcolor=COLOR_MED,  opacity=0.08, line_width=0),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=70,  y1=100, fillcolor=COLOR_HIGH, opacity=0.08, line_width=0),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if section == "Overview":
    st.title("Cognitive Load Index — Overview")
    st.markdown("Summary statistics across all participants, sessions, and windows.")

    # ── KPI cards ────────────────────────────────────────────────────────────
    n_participants = df["participant_id"].nunique()
    n_windows      = len(df)
    n_segments     = df.groupby(["participant_id", "session", "segment"]).ngroups
    avg_cli        = df["CLI"].mean()
    pct_high       = (df["CLI_category"] == "High").mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participants",  n_participants)
    c2.metric("Windows",       f"{n_windows:,}")
    c3.metric("Segments",      n_segments)
    c4.metric("Avg CLI",       f"{avg_cli:.1f}")
    c5.metric("% High Load",   f"{pct_high:.1f}%")

    st.markdown("---")

    # ── Category distribution ─────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Window Distribution by Category")
        cat_counts = df["CLI_category"].value_counts().reindex(["Low", "Medium", "High"])
        fig_pie = go.Figure(go.Pie(
            labels=cat_counts.index,
            values=cat_counts.values,
            marker_colors=[COLOR_MAP[c] for c in cat_counts.index],
            hole=0.4,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("CLI Distribution (all windows)")
        fig_hist = go.Figure()
        fig_hist.add_vrect(x0=0,  x1=40,  fillcolor=COLOR_LOW,  opacity=0.10, line_width=0)
        fig_hist.add_vrect(x0=40, x1=70,  fillcolor=COLOR_MED,  opacity=0.10, line_width=0)
        fig_hist.add_vrect(x0=70, x1=100, fillcolor=COLOR_HIGH, opacity=0.10, line_width=0)
        fig_hist.add_trace(go.Histogram(x=df["CLI"], nbinsx=40, marker_color="#5b9bd5", opacity=0.85))
        fig_hist.add_vline(x=avg_cli, line_dash="dash", line_color="black",
                           annotation_text=f"Mean={avg_cli:.1f}", annotation_position="top right")
        fig_hist.update_layout(xaxis_title="CLI", yaxis_title="Windows",
                               margin=dict(t=20, b=40, l=40, r=20), height=320,
                               xaxis_range=[0, 100])
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ── Participant summary table ──────────────────────────────────────────────
    st.subheader("Participant-Level Summary")
    part_summary = (
        df.groupby("participant_id")
        .agg(
            Sessions  = ("session",      "nunique"),
            Windows   = ("window_idx",   "count"),
            Avg_CLI   = ("CLI",          "mean"),
            Pct_High  = ("CLI_category", lambda x: (x == "High").mean() * 100),
            Pct_Low   = ("CLI_category", lambda x: (x == "Low").mean()  * 100),
        )
        .reset_index()
        .rename(columns={"participant_id": "Participant", "Avg_CLI": "Avg CLI",
                         "Pct_High": "% High", "Pct_Low": "% Low"})
    )
    part_summary["Avg CLI"] = part_summary["Avg CLI"].round(1)
    part_summary["% High"]  = part_summary["% High"].round(1)
    part_summary["% Low"]   = part_summary["% Low"].round(1)

    # color Avg CLI column
    def color_cli(val):
        if val <= 40:
            return f"background-color: {COLOR_LOW}33"
        elif val <= 70:
            return f"background-color: {COLOR_MED}33"
        else:
            return f"background-color: {COLOR_HIGH}33"

    st.dataframe(
        part_summary.style.applymap(color_cli, subset=["Avg CLI"]),
        use_container_width=True,
        hide_index=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PARTICIPANT VIEW
# ─────────────────────────────────────────────────────────────────────────────
elif section == "Participant View":
    st.title("Participant View")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        participants = sorted(df["participant_id"].unique())
        selected_pid = st.selectbox("Participant", participants)
    with col_sel2:
        sessions = sorted(df.loc[df["participant_id"] == selected_pid, "session"].unique())
        selected_ses = st.selectbox("Session", sessions)

    pdata = df[(df["participant_id"] == selected_pid) & (df["session"] == selected_ses)].copy()
    pdata = pdata.sort_values(["segment", "window_idx"]).reset_index(drop=True)
    pdata["window_global"] = range(len(pdata))

    if pdata.empty:
        st.warning("No data for this selection.")
        st.stop()

    st.markdown("---")

    # ── CLI time-series ───────────────────────────────────────────────────────
    st.subheader(f"CLI Time-Series — {selected_pid} / {selected_ses}")

    # segment boundary lines
    seg_boundaries = pdata.groupby("segment")["window_global"].min().to_dict()

    fig_ts = go.Figure()
    # colored zone backgrounds
    for sh in cli_zone_shapes():
        fig_ts.add_shape(**sh)

    # trace colored by category
    for cat, grp in pdata.groupby("CLI_category"):
        fig_ts.add_trace(go.Scatter(
            x=grp["window_global"], y=grp["CLI"],
            mode="markers",
            marker=dict(color=COLOR_MAP[cat], size=4, opacity=0.7),
            name=cat,
        ))

    # segment marker lines + labels
    for seg, xpos in seg_boundaries.items():
        fig_ts.add_vline(x=xpos, line_dash="dot", line_color="gray", line_width=1)
        fig_ts.add_annotation(x=xpos, y=103, text=seg, showarrow=False,
                               textangle=-45, font=dict(size=9, color="gray"),
                               xanchor="left", yanchor="bottom")

    fig_ts.add_hline(y=40, line_dash="dash", line_color=COLOR_LOW,  line_width=1)
    fig_ts.add_hline(y=70, line_dash="dash", line_color=COLOR_HIGH, line_width=1)
    fig_ts.update_layout(
        xaxis_title="Window index", yaxis_title="CLI",
        yaxis_range=[0, 108],
        legend_title="Category",
        margin=dict(t=40, b=60, l=50, r=20),
        height=420,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # ── Stats + interpretation ────────────────────────────────────────────────
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        avg  = pdata["CLI"].mean()
        p_lo = (pdata["CLI_category"] == "Low").mean()    * 100
        p_me = (pdata["CLI_category"] == "Medium").mean() * 100
        p_hi = (pdata["CLI_category"] == "High").mean()   * 100

        st.subheader("Session Statistics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg CLI",    f"{avg:.1f}")
        m2.metric("% Low",      f"{p_lo:.1f}%")
        m3.metric("% Medium",   f"{p_me:.1f}%")
        m4.metric("% High",     f"{p_hi:.1f}%")

        # dominant category
        dom = pdata["CLI_category"].value_counts().idxmax()
        dom_color = COLOR_MAP[dom]
        st.markdown(
            f'<div style="background-color:{dom_color}22;border-left:5px solid {dom_color};'
            f'padding:14px;border-radius:4px;margin-top:12px;">'
            f'<b>Overall assessment:</b> Dominant cognitive load is '
            f'<span style="color:{dom_color};font-weight:bold;">{dom}</span>.<br>'
            f'Average CLI = <b>{avg:.1f}</b> / 100. '
            + ("High cognitive demand — monitor for overload risk."
               if dom == "High" else
               "Moderate cognitive demand — within manageable range."
               if dom == "Medium" else
               "Low cognitive demand — participant appears at ease.")
            + "</div>",
            unsafe_allow_html=True,
        )

    with col_s2:
        st.subheader("Segment-Level Summary")
        seg_tbl = (
            pdata.groupby("segment")
            .agg(
                Windows    = ("window_idx",          "count"),
                Avg_CLI    = ("CLI",                 "mean"),
                Pct_High   = ("CLI_category",        lambda x: (x == "High").mean() * 100),
                NASA_Score = ("Weighted_NASA_score", "mean"),
            )
            .reset_index()
            .rename(columns={"segment": "Segment", "Avg_CLI": "Avg CLI",
                             "Pct_High": "% High", "NASA_Score": "NASA Score"})
        )
        seg_tbl["Avg CLI"]    = seg_tbl["Avg CLI"].round(1)
        seg_tbl["% High"]     = seg_tbl["% High"].round(1)
        seg_tbl["NASA Score"] = seg_tbl["NASA Score"].round(1)
        st.dataframe(seg_tbl.style.applymap(color_cli, subset=["Avg CLI"]),
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TASK / SEGMENT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif section == "Task / Segment Comparison":
    st.title("Task / Segment Comparison")
    st.markdown("Mean CLI per segment type, averaged across all participants and sessions.")

    seg_agg = (
        df.groupby("segment")
        .agg(
            Mean_CLI   = ("CLI",                 "mean"),
            Pct_High   = ("CLI_category",        lambda x: (x == "High").mean() * 100),
            Pct_Low    = ("CLI_category",        lambda x: (x == "Low").mean()  * 100),
            Windows    = ("window_idx",          "count"),
            NASA_Score = ("Weighted_NASA_score", "mean"),
        )
        .reset_index()
        .rename(columns={"segment": "Segment", "Mean_CLI": "Mean CLI",
                         "Pct_High": "% High", "Pct_Low": "% Low",
                         "NASA_Score": "Avg NASA"})
    )

    sort_col = st.selectbox("Sort by", ["Mean CLI", "% High", "Avg NASA", "Windows"], index=0)
    seg_agg  = seg_agg.sort_values(sort_col, ascending=False)

    # category color for each bar
    def cat_color(v):
        if v <= 40:   return COLOR_LOW
        elif v <= 70: return COLOR_MED
        else:         return COLOR_HIGH

    bar_colors = [cat_color(v) for v in seg_agg["Mean CLI"]]

    fig_bar = go.Figure(go.Bar(
        x=seg_agg["Segment"], y=seg_agg["Mean CLI"],
        marker_color=bar_colors,
        text=seg_agg["Mean CLI"].round(1),
        textposition="outside",
    ))
    fig_bar.add_hline(y=40, line_dash="dash", line_color=COLOR_LOW,  annotation_text="Low/Med boundary",  annotation_position="right")
    fig_bar.add_hline(y=70, line_dash="dash", line_color=COLOR_HIGH, annotation_text="Med/High boundary", annotation_position="right")
    fig_bar.update_layout(
        xaxis_title="Segment", yaxis_title="Mean CLI",
        yaxis_range=[0, 100],
        margin=dict(t=20, b=80, l=50, r=20),
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Segment Statistics Table")

    display_tbl = seg_agg.copy()
    for col in ["Mean CLI", "% High", "% Low", "Avg NASA"]:
        display_tbl[col] = display_tbl[col].round(1)

    st.dataframe(
        display_tbl.style.applymap(color_cli, subset=["Mean CLI"]),
        use_container_width=True,
        hide_index=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif section == "Model Results":
    st.title("Model Results")

    # ── Model info cards ──────────────────────────────────────────────────────
    st.subheader("Selected Model Configuration")
    ci1, ci2, ci3 = st.columns(3)
    ci1.info("**Algorithm**\n\nGradient Boosting Classifier\nn_estimators=100, random_state=42")
    ci2.info("**Training Strategy**\n\nLOPO Cross-Validation\n24 folds (one per participant)")
    ci3.info("**Data Filter**\n\nv3 Noisy-Label Filter\n(NASA-TLX threshold ±60/40)")

    st.markdown("---")

    # ── Performance metrics ───────────────────────────────────────────────────
    st.subheader("Performance Metrics — Selected Variant (Variant A)")
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Accuracy",      "0.541")
    pm2.metric("Balanced Acc",  "0.551")
    pm3.metric("F1 (macro)",    "0.536")
    pm4.metric("ROC-AUC",       "0.568")
    pm5.metric("Recall High",   "0.643")

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.subheader("Feature Importance (Variant A — Original Features)")
    feat_sorted = feat_df.sort_values("importance", ascending=True)
    group_colors = {
        "TEMP": "#e67e22", "EDA": "#3498db", "HRV": "#9b59b6"
    }
    bar_clrs = [group_colors.get(g, "#95a5a6") for g in feat_sorted["group"]]

    fig_feat = go.Figure(go.Bar(
        x=feat_sorted["importance"],
        y=feat_sorted["feature"],
        orientation="h",
        marker_color=bar_clrs,
        text=feat_sorted["importance"].round(3),
        textposition="outside",
    ))
    fig_feat.update_layout(
        xaxis_title="Mean Importance", yaxis_title="",
        margin=dict(t=10, b=40, l=200, r=80),
        height=max(300, len(feat_sorted) * 32),
    )
    # legend for groups
    for grp, clr in group_colors.items():
        if grp in feat_sorted["group"].values:
            fig_feat.add_trace(go.Bar(x=[None], y=[None], name=grp, marker_color=clr, showlegend=True))
    fig_feat.update_layout(legend_title="Signal group")
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")

    # ── Variant comparison table ───────────────────────────────────────────────
    st.subheader("Variant Comparison")
    st.caption("All variants use v3 noisy-label filter. Variant A selected (no B/C improvement ≥ 0.005 LOPO AUC).")

    display_comp = comp_df.copy()
    # highlight selected row
    def highlight_selected(row):
        if "A" in str(row.get("Variant", "")):
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_comp.style.apply(highlight_selected, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Metric radar ──────────────────────────────────────────────────────────
    st.subheader("Metric Radar — Selected Variant")
    metrics_labels = ["Accuracy", "Balanced Acc", "F1 (macro)", "ROC-AUC", "Recall High"]
    metrics_vals   = [0.541, 0.551, 0.536, 0.568, 0.643]
    metrics_vals_c = metrics_vals + [metrics_vals[0]]
    metrics_labels_c = metrics_labels + [metrics_labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=metrics_vals_c,
        theta=metrics_labels_c,
        fill="toself",
        fillcolor="rgba(91,155,213,0.2)",
        line_color="#5b9bd5",
        name="Variant A",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=380,
        margin=dict(t=20, b=20, l=60, r=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PROJECT INFORMATION
# ─────────────────────────────────────────────────────────────────────────────
elif section == "Project Information":
    st.title("Project Information")

    # ── Description ───────────────────────────────────────────────────────────
    st.subheader("Project Description")
    st.markdown("""
**Predicting and Preventing Burnout Using Cognitive Load Analysis**

This project develops a Cognitive Load Index (CLI) from physiological signals (EDA, HRV, skin temperature)
collected in the Anders et al. dataset. 24 participants performed easy and hard cognitive tasks
across two laboratory sessions (Lab1, Lab2). The CLI provides a continuous, personalized measure of
cognitive demand that can be used for early burnout risk monitoring.
""")

    st.markdown("---")

    # ── CLI formula ───────────────────────────────────────────────────────────
    st.subheader("CLI Definition & Interpretation")
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        st.markdown(r"""
**Formula**

$$\text{CLI} = P(\text{High}) \times 100$$

where $P(\text{High})$ is the Gradient Boosting Classifier's predicted probability
that a 30-second window belongs to the High cognitive load class.
""")
    with col_f2:
        st.markdown(
            f'<div style="margin-top:12px;">'
            f'<div style="background:{COLOR_LOW}22;border-left:5px solid {COLOR_LOW};padding:10px;margin-bottom:8px;border-radius:4px;">'
            f'<b>0 – 40 · Low</b> — relaxed, low cognitive demand</div>'
            f'<div style="background:{COLOR_MED}22;border-left:5px solid {COLOR_MED};padding:10px;margin-bottom:8px;border-radius:4px;">'
            f'<b>41 – 70 · Medium</b> — moderate demand, within sustainable range</div>'
            f'<div style="background:{COLOR_HIGH}22;border-left:5px solid {COLOR_HIGH};padding:10px;border-radius:4px;">'
            f'<b>71 – 100 · High</b> — high demand, monitor for overload</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Analysis progression ──────────────────────────────────────────────────
    st.subheader("Analysis Progression")
    progression = pd.DataFrame([
        {"Notebook": "01", "Description": "Initial CLI pipeline — baseline LOPO, GBT model"},
        {"Notebook": "02", "Description": "Improved pipeline — v2 baseline model (GBT n=100, random_state=42)"},
        {"Notebook": "03", "Description": "NASA-TLX exploration — questionnaire scores vs. predicted CLI"},
        {"Notebook": "04", "Description": "v3 noisy-label filter — remove ambiguous segments from training"},
        {"Notebook": "05", "Description": "Personal calibration — per-participant Low-session normalization"},
        {"Notebook": "06", "Description": "Relative features — delta and ratio features ablation study"},
        {"Notebook": "07", "Description": "v7 final model — v3 filter + relative features 2×2 design → Variant A selected"},
    ])
    st.dataframe(progression, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Limitations ───────────────────────────────────────────────────────────
    st.subheader("Limitations")
    limitations = [
        "Small sample: 24 participants, two laboratory sessions — results may not generalize to field settings.",
        "Binary labels derived from task type (*easy / *hard); no continuous ground-truth cognitive load signal.",
        "LOPO ROC-AUC of 0.568 indicates modest generalisation across unseen participants.",
        "v2 normalization uses per-participant z-score across train and test windows together — an optimistic calibration assumption.",
        "Wild sessions excluded due to ambiguous labels and partial recordings.",
        "No real-time streaming; the CLI is computed post-hoc from pre-computed windowed features.",
        "Physiological signals are sensitive to sensor placement, movement artefacts, and individual baselines.",
    ]
    for lim in limitations:
        st.markdown(f"- {lim}")

    st.markdown("---")

    # ── Dataset info ──────────────────────────────────────────────────────────
    st.subheader("Dataset")
    col_d1, col_d2 = st.columns(2)
    col_d1.markdown("""
| Property | Value |
|---|---|
| Source | Anders et al. physiological dataset |
| Participants | 24 (UN_101 – UN_124) |
| Sessions used | Lab1, Lab2 |
| Signals | EDA, HRV, Skin Temperature |
| Window size | 30 s (feature pre-computed) |
| Total windows | 18,659 |
| Total segments | 407 |
""")
    col_d2.markdown("""
| Label | Segments |
|---|---|
| Low (0) | relaxation_video, video_baseline, *_easy |
| High (1) | *_hard |
| Excluded | Wild sessions |

**Noisy-label filter (v3)**
- Low window + NASA-TLX > 60 → removed from training
- High window + NASA-TLX < 40 → removed from training
- Segments without NASA match → kept (conservative)
""")

print("Dashboard created successfully.")
print()
print("Generated files:")
print("  app.py")
print("  outputs/cli_dashboard_data.csv")
print("  outputs/v7_model_comparison.csv")
print("  outputs/v7_lopo_predictions_best_variant.csv")
print("  outputs/v7_feature_importance_best_variant.csv")
