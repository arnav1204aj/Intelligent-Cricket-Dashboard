# app.py
import streamlit as st
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from math import isfinite

# -------------------- Helpers --------------------
def load_bin(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_val(d, key):
    """Return a numeric value from d[key]. If it's a dict, take a numeric inner value.
       If missing or non-numeric, return np.nan."""
    if d is None:
        return np.nan
    v = d.get(key, np.nan)
    if isinstance(v, dict):
        # try explicit common keys first
        for k in ("value", "val", "intent", "score", list(v.keys())[0] if v else None):
            if k and k in v and isinstance(v[k], (int, float)) and isfinite(v[k]):
                return float(v[k])
        # else pick first numeric entry
        for vv in v.values():
            if isinstance(vv, (int, float)) and isfinite(vv):
                return float(vv)
        return np.nan
    # numeric
    if isinstance(v, (int, float)) and isfinite(v):
        return float(v)
    # sometimes v could be string number
    try:
        return float(v)
    except Exception:
        return np.nan

def percentile_better(value, arr, higher_is_better=True):
    """Return percentage (0-100) of entries in arr that the current value is 'better than'.
       For ties count half. NaNs in arr are ignored. If arr empty return np.nan.
    """
    arr = np.array([x for x in arr if x is not None and not np.isnan(x)])
    if arr.size == 0 or np.isnan(value):
        return np.nan
    if higher_is_better:
        less = np.sum(arr < value)
        equal = np.sum(arr == value)
        score = (less + 0.5 * equal) / arr.size
    else:
        # lower is better -> count how many are greater than value
        greater = np.sum(arr > value)
        equal = np.sum(arr == value)
        score = (greater + 0.5 * equal) / arr.size
    return float(score * 100.0)

def metric_label_html(label, tooltip_template, value):
    import numpy as np
    val_text = "—" if np.isnan(value) else round(value, 2)
    tooltip = tooltip_template.format(val=val_text)

    return f"""
    <style>
    .tooltip {{
      position: relative;
      display: inline-block;
      cursor: pointer;
      z-index: 9999; /* bring above everything */
    }}
    .tooltip .tooltiptext {{
      visibility: hidden;
      width: 240px;
      background-color: #333;
      color: #fff;
      text-align: center;
      border-radius: 6px;
      padding: 6px;
      position: absolute;
      z-index: 9999; /* keep on top */
      bottom: 125%; /* position above the icon */
      left: 50%;
      transform: translateX(-50%);
      opacity: 0;
      transition: opacity 0.3s;
    }}
    /* show when the icon itself is clicked/tapped */
    .tooltip:focus-within .tooltiptext {{
      visibility: visible;
      opacity: 1;
    }}
    </style>
    <div style="display:flex; align-items:center; color:white;">
      <span style="margin-right:6px;">{label}</span>
      <div class="tooltip">
        <span tabindex="0" style="font-weight:bold; font-size:18px; outline:none;">ℹ️</span>
        <span class="tooltiptext">{tooltip}</span>
      </div>
    </div>
    """





def colored_bar_html(pct):
    """Return an HTML snippet drawing a horizontal red->green gradient bar (full gradient across 0..100),
    and mask the right side so only 'pct'% of the gradient is visible as the filled portion.
    Score text is displayed in white on top of the bar.
    """
    if pct is None or np.isnan(pct):
        pct = 0.0
    pct = float(max(0.0, min(100.0, pct)))

    html = f"""
    <div style="position:relative; width:100%; height:22px;
                border-radius:6px; overflow:hidden;
                box-shadow: inset 0 -2px 0 rgba(0,0,0,0.08);">
      <!-- full gradient background (covers entire bar from 0..100) -->
      <div style="position:absolute; inset:0;
                  background: linear-gradient(90deg, #e74c3c 0%, #f1c40f 50%, #2ecc71 100%);
                  z-index:0;"></div>

      <!-- mask (covers the unfilled/right portion) -->
      <div style="position:absolute; top:0; bottom:0; left:{pct}%; right:0;
                  background:#eee; z-index:1;
                  border-top-right-radius:6px; border-bottom-right-radius:6px;"></div>

      <!-- score text centered above the bar (white) -->
      <div style="position:absolute; left:50%; top:50%;
                  transform: translate(-50%, -50%);
                  z-index:2; color:black; font-weight:700; font-size:20px;
                  text-shadow: 0 1px 2px rgba(0,0,0,0.6); padding:0px 8px; border-radius:4px;">
        {pct:.1f}
      </div>
    </div>
    """
    return html


# -------------------- Load data --------------------
DATA_DIR = "t20_decay"
required_files = ["fshots.bin", "intents.bin", "negative_dur.bin", "impact_stats.bin","360.bin",'vectorsnorm.bin','rlist.bin']
for fn in required_files:
    if not os.path.exists(os.path.join(DATA_DIR, fn)):
        raise FileNotFoundError(f"Required file missing: {os.path.join(DATA_DIR, fn)}")

fshots = load_bin(os.path.join(DATA_DIR, "fshots.bin"))
intents = load_bin(os.path.join(DATA_DIR, "intents.bin"))
negative_dur = load_bin(os.path.join(DATA_DIR, "negative_dur.bin"))
impact_stats = load_bin(os.path.join(DATA_DIR, "impact_stats.bin"))
stats_360 = load_bin(os.path.join(DATA_DIR, "360.bin"))
vecdic = load_bin(os.path.join(DATA_DIR, "vectorsnorm.bin"))
# rlist = load_bin(os.path.join(DATA_DIR, "rlist.bin"))
# -------------------- Global batter list (intersection) --------------------
batter_list = sorted(
    set(fshots.keys())
    & set(intents.keys())
    & set(negative_dur.keys())
    & set(impact_stats.keys())
    & set(stats_360.keys())
    & set(vecdic.keys())
)

# -------------------- Precompute arrays for percentiles --------------------
# Build arrays aligned on batter_list for the metrics we will show.

def build_metric_arrays(b_list):
    ipace = []
    isp = []
    iov = []
    rel_pace_a = []
    rel_spin_a = []
    rel_overall_a = []
    # impact tab arrays
    negdur_a = []
    per_ball_a = []
    per_inn_a = []
    imp_improv_a = []
    score_360_a = []
    audacity_a = []
    for b in b_list:
        # intents
        ip = safe_val(intents[b], "pace")
        is_ = safe_val(intents[b], "spin")
        ipace.append(ip)
        isp.append(is_)
        iov.append(np.nanmean([x for x in (ip, is_) if not np.isnan(x)]) if not (np.isnan(ip) and np.isnan(is_)) else np.nan)

        # fshots -> reliability = 1 / fshots
        fs_p = safe_val(fshots[b], "pace")
        fs_s = safe_val(fshots[b], "spin")
        rel_p = np.nan if fs_p == 0 or np.isnan(fs_p) else 1.0 / fs_p
        rel_s = np.nan if fs_s == 0 or np.isnan(fs_s) else 1.0 / fs_s
        rel_pace_a.append(rel_p)
        rel_spin_a.append(rel_s)
        rel_overall_a.append(np.nanmean([x for x in (rel_p, rel_s) if not np.isnan(x)]) if not (np.isnan(rel_p) and np.isnan(rel_s)) else np.nan)

        # negative dur
        nd = safe_val(negative_dur, b) if isinstance(negative_dur, dict) and b in negative_dur else safe_val({"v":negative_dur.get(b)},"v") if isinstance(negative_dur, dict) else np.nan
        # simpler: negative_dur is dict[batter]=value so:
        try:
            nd = float(negative_dur[b])
        except Exception:
            nd = np.nan
        negdur_a.append(nd)

        # impact_stats: dict[b][metric]
        imp = impact_stats.get(b, {})
        per_ball = safe_val(imp, "per_ball_impact") if isinstance(imp, dict) else np.nan
        per_inn = safe_val(imp, "per_inn_impact") if isinstance(imp, dict) else np.nan
        imp_improv = safe_val(imp, "impact_improvement") if isinstance(imp, dict) else np.nan
        per_ball_a.append(per_ball)
        per_inn_a.append(per_inn)
        imp_improv_a.append(imp_improv)

        s360 = stats_360.get(b, {})
        audacity = safe_val(s360, "audacity") if isinstance(s360, dict) else np.nan
        score360 = safe_val(s360, "score360") if isinstance(s360, dict) else np.nan
        audacity_a.append(audacity)
        score_360_a.append(score360)
        

    return {
        "intent_pace": np.array(ipace, dtype=float),
        "intent_spin": np.array(isp, dtype=float),
        "intent_overall": np.array(iov, dtype=float),
        "rel_pace": np.array(rel_pace_a, dtype=float),
        "rel_spin": np.array(rel_spin_a, dtype=float),
        "rel_overall": np.array(rel_overall_a, dtype=float),
        "neg_dur": np.array(negdur_a, dtype=float),
        "per_ball": np.array(per_ball_a, dtype=float),
        "per_inn": np.array(per_inn_a, dtype=float),
        "imp_improv": np.array(imp_improv_a, dtype=float),
        "audacity":np.array(audacity_a,dtype=float),
        "score_360":np.array(score_360_a,dtype=float),
    }

metric_arrays = build_metric_arrays(batter_list)
n_batters = len(batter_list)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="T20 Batting Analytics", layout="wide")
st.markdown("""
    <style>
      body { background: #ffffff; }
      .big-title { font-family: 'Helvetica Neue', Arial, sans-serif; font-size:28px; font-weight:700; margin-bottom:6px;}
      .subtitle { color: #666; margin-bottom:18px; font-size:14px; }
      .metric-label { color:#444; font-weight:600; }
      .small { color:#666; font-size:0.9rem; }
      .card { background: #fff; border-radius:8px; padding:12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
      .metric-val { font-size:1.05rem; font-weight:700; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">T20 Batting Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intent / Reliability, Impact and 360 play metrics — search and compare players (2015-2025, 50% weight to latest 2 years performance)</div>', unsafe_allow_html=True)

# ---- Search + suggestion-like behavior ----
# single selectbox with built-in filtering
batter = st.selectbox("Search or select batter:", options=batter_list, index=0, key="batter_select")


    # Compute values for selected batter
    # intents
intent_pace = safe_val(intents[batter], "pace")
intent_spin = safe_val(intents[batter], "spin")
intent_overall = np.nanmean([x for x in (intent_pace, intent_spin) if not np.isnan(x)]) if not (np.isnan(intent_pace) and np.isnan(intent_spin)) else np.nan

# reliability
fs_p = safe_val(fshots[batter], "pace")
fs_s = safe_val(fshots[batter], "spin")
rel_pace = np.nan if fs_p == 0 or np.isnan(fs_p) else 1.0 / fs_p
rel_spin = np.nan if fs_s == 0 or np.isnan(fs_s) else 1.0 / fs_s
rel_overall = np.nanmean([x for x in (rel_pace, rel_spin) if not np.isnan(x)]) if not (np.isnan(rel_pace) and np.isnan(rel_spin)) else np.nan

intrel_pace = intent_pace * rel_pace if not (np.isnan(intent_pace) or np.isnan(rel_pace)) else np.nan
intrel_spin = intent_spin * rel_spin if not (np.isnan(intent_spin) or np.isnan(rel_spin)) else np.nan
intrel_overall = intent_overall * rel_overall if not (np.isnan(intent_overall) or np.isnan(rel_overall)) else np.nan

# impact metrics
neg_dur_v = float(negative_dur.get(batter, np.nan)) if batter in negative_dur else np.nan
imp = impact_stats.get(batter, {})
per_ball_v = safe_val(imp, "per_ball_impact")
per_inn_v = safe_val(imp, "per_inn_impact")
imp_improv_v = safe_val(imp, "impact_improvement")

# -------------------- Percentiles --------------------
# Prepare arrays for percentiles
ma = metric_arrays
p_intent_pace = percentile_better(intent_pace, ma["intent_pace"], higher_is_better=True)
p_intent_spin = percentile_better(intent_spin, ma["intent_spin"], higher_is_better=True)
p_intent_overall = percentile_better(intent_overall, ma["intent_overall"], higher_is_better=True)

p_rel_pace = percentile_better(rel_pace, ma["rel_pace"], higher_is_better=True)
p_rel_spin = percentile_better(rel_spin, ma["rel_spin"], higher_is_better=True)
p_rel_overall = percentile_better(rel_overall, ma["rel_overall"], higher_is_better=True)

p_intrel_pace = percentile_better(intrel_pace, ma["intent_pace"] * ma["rel_pace"], higher_is_better=True) if (ma["intent_pace"].size and ma["rel_pace"].size) else np.nan
p_intrel_spin = percentile_better(intrel_spin, ma["intent_spin"] * ma["rel_spin"], higher_is_better=True) if (ma["intent_spin"].size and ma["rel_spin"].size) else np.nan
p_intrel_overall = percentile_better(intrel_overall, ma["intent_overall"] * ma["rel_overall"], higher_is_better=True) if (ma["intent_overall"].size and ma["rel_overall"].size) else np.nan

# Impact tab percentiles
p_neg_dur = percentile_better(neg_dur_v, ma["neg_dur"], higher_is_better=False)  # lower is better
p_per_ball = percentile_better(per_ball_v, ma["per_ball"], higher_is_better=True)
p_per_inn = percentile_better(per_inn_v, ma["per_inn"], higher_is_better=True)
p_imp_improv = percentile_better(imp_improv_v, ma["imp_improv"], higher_is_better=True)

# Average percentiles for each tab
# Int-Rel average uses: intent_overall, rel_overall, intrel_overall
list_intrel_pct = [p for p in [p_intent_overall, p_rel_overall, p_intrel_overall] if not np.isnan(p)]
avg_intrel_pct = float(np.nanmean(list_intrel_pct)) if list_intrel_pct else np.nan

# Int Impact average uses: negative duration (inverted percentile), per_ball, per_inn, improvement
list_impact_pct = [p for p in [p_neg_dur, p_per_ball, p_per_inn, p_imp_improv] if not np.isnan(p)]
avg_impact_pct = float(np.nanmean(list_impact_pct)) if list_impact_pct else np.nan

audacity_v = safe_val(stats_360[batter], "audacity")
score360_v = safe_val(stats_360[batter], "score360")

p_audacity = percentile_better(audacity_v, ma["audacity"], higher_is_better=True)
p_score360 = percentile_better(score360_v, ma["score_360"], higher_is_better=True)

list_360_pct = [p for p in [p_audacity, p_score360] if not np.isnan(p)]
avg_360_pct = float(np.nanmean(list_360_pct)) if list_360_pct else np.nan
# -------------------- Display --------------------
# show small header with batter name
st.markdown(f"### {batter}", unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Int-Rel", "Int Impact", "360 Play","Info"])

# ─── Custom CSS for Metrics ─────────────────────────────────────────────
def metric_gradient_html(val, text_html):
    """
    val: numeric value (0-100 ideally)
    text_html: the HTML inside the div (value + % better)
    returns HTML with gradient background & masked fill portion
    """
    if np.isnan(val):
        val = 0.0  # treat NaN as 0 for mask
    
    pct = float(max(0.0, min(100.0, val)))

    # Full gradient background & mask logic
    return f"""
    <div class='metric-val' style='position:relative; overflow:hidden;'>
      <!-- full gradient background -->
      <div style="position:absolute; inset:0;
                  background: linear-gradient(90deg, #e74c3c 0%, #f1c40f 50%, #2ecc71 100%);
                  z-index:0;"></div>

      <!-- mask for unfilled portion -->
      <div style="position:absolute; top:0; bottom:0; left:{pct}%; right:0;
                  background:#f7f9fa; z-index:1;"></div>

      <!-- text on top -->
      <div style="position:relative; z-index:2;">
        {text_html}
      </div>
    </div>
    """


# ─── Styles ───────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-block {
    margin-bottom:10px;
}
.metric-name {
    font-size:0.95rem;
    font-weight:600;
    color:#444;
    margin-bottom:2px;
}
.metric-val {
    font-size:1.1rem;
    font-weight:700;
    color:#111;
    padding:4px 6px;
    border-radius:4px;
    display:inline-block;
}
.metric-val .small {
    font-size:0.8rem;
    color:#222;
    margin-left:4px;
}
.section-title {
    margin-bottom:8px;
    font-weight:700;
    font-size:1.1rem;
    color:#333;
}
</style>
""", unsafe_allow_html=True)

# ─── TAB 1 ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-title' style='color:white;'>Overall Int-Rel Profile</div>", unsafe_allow_html=True)
    st.markdown(colored_bar_html(round(avg_intrel_pct, 2)), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_label_html("Intent (Pace)",
            "If non striker plays with a SR of 100, batter plays with a SR of {val}", 100*intent_pace),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intent_pace,
            f"{'—' if np.isnan(intent_pace) else round(intent_pace,2)} "
            f"<span class='small'>({'' if np.isnan(p_intent_pace) else f'{round(p_intent_pace,2)}% better'})</span>"
        ), unsafe_allow_html=True)

        st.markdown(metric_label_html("Reliability (Pace)",
            "If non striker has a control% of 100, batter has a control % of {val}", 100*rel_pace),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_rel_pace,
            f"{'—' if np.isnan(rel_pace) else round(rel_pace,2)} "
            f"<span class='small'>({'' if np.isnan(p_rel_pace) else f'{round(p_rel_pace,2)}% better'})</span>"
        ), unsafe_allow_html=True)

        st.markdown(metric_label_html("Int-Rel (Pace)",
            "A combined score (Intent*Reliability), indicating controlled striking ability. Value={val}", intrel_pace),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intrel_pace,
            f"{'—' if np.isnan(intrel_pace) else round(intrel_pace,2)} "
            f"<span class='small'>({'' if np.isnan(p_intrel_pace) else f'{round(p_intrel_pace,2)}% better'})</span>"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(metric_label_html("Intent (Spin)",
            "If non striker plays with a SR of 100, batter plays with a SR of {val}", 100*intent_spin),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intent_spin,
            f"{'—' if np.isnan(intent_spin) else round(intent_spin,2)} "
            f"<span class='small'>({'' if np.isnan(p_intent_spin) else f'{round(p_intent_spin,2)}% better'})</span>"
        ), unsafe_allow_html=True)

        st.markdown(metric_label_html("Reliability (Spin)",
            "If non striker has a control% of 100, batter has a control % of {val}", 100*rel_spin),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_rel_spin,
            f"{'—' if np.isnan(rel_spin) else round(rel_spin,2)} "
            f"<span class='small'>({'' if np.isnan(p_rel_spin) else f'{round(p_rel_spin,2)}% better'})</span>"
        ), unsafe_allow_html=True)

        st.markdown(metric_label_html("Int-Rel (Spin)",
            "A combined score (Intent*Reliability), indicating controlled striking ability. Value={val}", intrel_spin),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intrel_spin,
            f"{'—' if np.isnan(intrel_spin) else round(intrel_spin,2)} "
            f"<span class='small'>({'' if np.isnan(p_intrel_spin) else f'{round(p_intrel_spin,2)}% better'})</span>"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(metric_label_html("Intent (Overall)",
            "If non striker plays with a SR of 100, batter plays with a SR of {val}", 100*intent_overall),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intent_overall,
            f"{'—' if np.isnan(intent_overall) else round(intent_overall,2)} "
            f"<span class='small'>({'' if np.isnan(p_intent_overall) else f'{round(p_intent_overall,2)}% better'})</span>"
        ), unsafe_allow_html=True)

        st.markdown(metric_label_html("Reliability (Overall)",
            "If non striker has a control% of 100, batter has a control % of {val}", 100*rel_overall),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_rel_overall,
            f"{'—' if np.isnan(rel_overall) else round(rel_overall,2)} "
            f"<span class='small'>({'' if np.isnan(p_rel_overall) else f'{round(p_rel_overall,2)}% better'})</span>"
        ), unsafe_allow_html=True)

        st.markdown(metric_label_html("Int-Rel (Overall)",
            "A combined score (Intent*Reliability), indicating controlled striking ability. Value={val}", intrel_overall),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intrel_overall,
            f"{'—' if np.isnan(intrel_overall) else round(intrel_overall,2)} "
            f"<span class='small'>({'' if np.isnan(p_intrel_overall) else f'{round(p_intrel_overall,2)}% better'})</span>"
        ), unsafe_allow_html=True)


# ─── TAB 2 ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title' style='color:white;'>Overall Impact Profile</div>", unsafe_allow_html=True)
    st.markdown(colored_bar_html(round(avg_impact_pct, 2)), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_label_html("Negative Duration",
            "Batter takes {val} balls to convert his knock into a positive impact innings", neg_dur_v),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_neg_dur,
            f"{'—' if np.isnan(neg_dur_v) else round(neg_dur_v,2)} "
            f"<span class='small'>({'' if np.isnan(p_neg_dur) else f'{round(p_neg_dur,2)}% better'})</span>"
        ), unsafe_allow_html=True)

    with c2:
        st.markdown(metric_label_html("Per Ball Impact",
            "Batter adds {val} extra runs per ball", per_ball_v),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_per_ball,
            f"{'—' if np.isnan(per_ball_v) else round(per_ball_v,2)} "
            f"<span class='small'>({'' if np.isnan(p_per_ball) else f'{round(p_per_ball,2)}% better'})</span>"
        ), unsafe_allow_html=True)

    with c3:
        st.markdown(metric_label_html("Per Inning Impact",
            "Batter adds {val} extra runs per innings", per_inn_v),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_per_inn,
            f"{'—' if np.isnan(per_inn_v) else round(per_inn_v,2)} "
            f"<span class='small'>({'' if np.isnan(p_per_inn) else f'{round(p_per_inn,2)}% better'})</span>"
        ), unsafe_allow_html=True)

    with c4:
        st.markdown(metric_label_html("Impact Improvement",
            "Batter improves runs added per ball by {val} in the later stages (last 25%) of his innings (Compared to first 25%)", imp_improv_v),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_imp_improv,
            f"{'—' if np.isnan(imp_improv_v) else round(imp_improv_v,2)} "
            f"<span class='small'>({'' if np.isnan(p_imp_improv) else f'{round(p_imp_improv,2)}% better'})</span>"
        ), unsafe_allow_html=True)


with tab3:
    st.markdown("<div class='section-title' style='color:white;'>Overall 360 Play Profile</div>", unsafe_allow_html=True)
    st.markdown(colored_bar_html(round(avg_360_pct, 2)), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_label_html("Avg Shot Difficulty",
            "Probability of an average batter playing similar shots as batter is 1/{val}", audacity_v),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_audacity,
            f"{'—' if np.isnan(audacity_v) else round(audacity_v,2)} "
            f"<span class='small'>({'' if np.isnan(p_audacity) else f'{round(p_audacity,2)}% better'})</span>"
        ), unsafe_allow_html=True)

    with c2:
        st.markdown(metric_label_html("360 Score",
            "An overall wagon wheel spread score indicating equalness in different regions. Score={val}", score360_v),
            unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_score360,
            f"{'—' if np.isnan(score360_v) else round(score360_v,2)} "
            f"<span class='small'>({'' if np.isnan(p_score360) else f'{round(p_score360,2)}% better'})</span>"
        ), unsafe_allow_html=True)


    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='color:white;'>Intelligent Wagon Wheel (Takes into account shot difficulty along with the runs scored)</div>", unsafe_allow_html=True)

    # checkbox for boundaries only
    

    # --- Plot wagon wheel ---
    max_magnitude = 35  # already loaded vectorsnorm

    list1 = vecdic[batter]['vectors']
    evs = vecdic[batter]['evs']

    normal_vecs, boundary_vecs = [], []

    for i in range(len(list1)):
        if np.linalg.norm(evs[i]) > np.linalg.norm(list1[i]):
            vec = evs[i]
            norm = np.linalg.norm(vec)
            clipped = (
                vec[0] if norm <= max_magnitude else max_magnitude * (vec[0] / norm),
                vec[1] if norm <= max_magnitude else max_magnitude * (vec[1] / norm)
            )
            if np.linalg.norm(list1[i]) in [4, 6]:
                boundary_vecs.append(clipped)
            else:
                normal_vecs.append(clipped)

    x_normal = [v[0] for v in normal_vecs]
    y_normal = [v[1] for v in normal_vecs]
    x_boundary = [v[0] for v in boundary_vecs]
    y_boundary = [v[1] for v in boundary_vecs]

    origin_x = np.zeros(len(normal_vecs) + len(boundary_vecs))
    origin_y = np.zeros(len(normal_vecs) + len(boundary_vecs))

    # smaller figure
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('green')
    ax.set_facecolor('green')
    ax.axis('off')

    # boundaries
    plt.quiver(
        origin_x[len(normal_vecs):], origin_y[len(normal_vecs):],
        x_boundary, y_boundary,
        angles='xy', scale_units='xy', scale=1, color='red', alpha=0.9,
        headwidth=1, headlength=0, label='Boundaries (4s/6s)'
    )

    # normal shots (only if checkbox is not checked)
   
    plt.quiver(
            origin_x[:len(normal_vecs)], origin_y[:len(normal_vecs)],
            x_normal, y_normal,
            angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7,
            headwidth=1, headlength=0
    )

    # Circular max magnitude boundary
    circle = plt.Circle((0, 0), max_magnitude, color='white', fill=False, linewidth=2)
    ax.add_artist(circle)

    # Batter facing arrow
    ax.arrow(0, -10, 0, 10, color='white', width=0.2, head_width=1, head_length=1, length_includes_head=True)
    ax.text(0, -12, "Batter Facing", color='white', fontsize=12, ha='center')

    ax.set_xlim(-max_magnitude-2, max_magnitude+2)
    ax.set_ylim(-max_magnitude-2, max_magnitude+2)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

    
    st.markdown(
        """
        <style>
        .responsive-img {
            width: 100%;
            max-width: 600px;  /* prevents overly large */
            aspect-ratio: 1 / 1; /* keep square shape */
            margin: auto;
        }
        @media (min-width: 900px) {
            .responsive-img {
                width: 50%; /* half width on large screens */
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render figure as HTML img tag for responsiveness
    import io
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.getvalue()

    st.markdown(
        f"<div class='responsive-img'><img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}' style='width:100%;height:100%;'></div>",
        unsafe_allow_html=True
    )
# with tab4:
#     st.markdown(
#         "<div class='section-title' style='color:white;'>Rankings</div>",
#         unsafe_allow_html=True
#     )

#     # --- compute ranking metrics for all batters in rlist ---
#     r_batters = [b for b in rlist if b in batter_list]  # restrict to batters we have data for

#     # Precompute their profile percentiles
#     rank_data = []
#     for b in r_batters:
#         # IntRel
#         ip = safe_val(intents[b], "pace")
#         is_ = safe_val(intents[b], "spin")
#         io = np.nanmean([x for x in (ip, is_) if not np.isnan(x)]) if not (np.isnan(ip) and np.isnan(is_)) else np.nan

#         fs_p = safe_val(fshots[b], "pace")
#         fs_s = safe_val(fshots[b], "spin")
#         rp = np.nan if fs_p == 0 or np.isnan(fs_p) else 1.0 / fs_p
#         rs = np.nan if fs_s == 0 or np.isnan(fs_s) else 1.0 / fs_s
#         ro = np.nanmean([x for x in (rp, rs) if not np.isnan(x)]) if not (np.isnan(rp) and np.isnan(rs)) else np.nan

#         intrel_o = io * ro if not (np.isnan(io) or np.isnan(ro)) else np.nan

#         p_io = percentile_better(io, ma["intent_overall"], higher_is_better=True)
#         p_ro = percentile_better(ro, ma["rel_overall"], higher_is_better=True)
#         p_intrel = percentile_better(intrel_o, ma["intent_overall"] * ma["rel_overall"], higher_is_better=True)

#         avg_intrel_pct_b = float(np.nanmean([p for p in [p_io, p_ro, p_intrel] if not np.isnan(p)]))

#         # Impact
#         nd = float(negative_dur.get(b, np.nan))
#         impb = impact_stats.get(b, {})
#         pb = safe_val(impb, "per_ball_impact")
#         pi = safe_val(impb, "per_inn_impact")
#         iim = safe_val(impb, "impact_improvement")

#         p_nd = percentile_better(nd, ma["neg_dur"], higher_is_better=False)
#         p_pb = percentile_better(pb, ma["per_ball"], higher_is_better=True)
#         p_pi = percentile_better(pi, ma["per_inn"], higher_is_better=True)
#         p_iim = percentile_better(iim, ma["imp_improv"], higher_is_better=True)

#         avg_impact_pct_b = float(np.nanmean([p for p in [p_nd, p_pb, p_pi, p_iim] if not np.isnan(p)]))

#         # 360
#         s360 = stats_360.get(b, {})
#         aud = safe_val(s360, "audacity")
#         sc360 = safe_val(s360, "score360")
#         p_aud = percentile_better(aud, ma["audacity"], higher_is_better=True)
#         p_sc360 = percentile_better(sc360, ma["score_360"], higher_is_better=True)
#         avg_360_pct_b = float(np.nanmean([p for p in [p_aud, p_sc360] if not np.isnan(p)]))

#         # T20 Batting Index
#         t20_index = float(np.nanmean([avg_intrel_pct_b, avg_impact_pct_b, avg_360_pct_b]))

#         rank_data.append({
#             "batter": b,
#             "Int-Rel Profile": avg_intrel_pct_b,
#             "Impact Profile": avg_impact_pct_b,
#             "360 Profile": avg_360_pct_b,
#             "T20 Batting Index": t20_index
#         })

#     import pandas as pd
#     df_rank = pd.DataFrame(rank_data)

#     # User selection
#     ranking_type = st.selectbox(
#         "Select ranking basis:",
#         options=["Int-Rel Profile", "Impact Profile", "360 Profile", "T20 Batting Index"]
#     )

#     # Sort by selected metric
#     df_rank = df_rank.sort_values(by=ranking_type, ascending=False).reset_index(drop=True)
#     df_rank["Rank"] = df_rank.index + 1

#     # Search player
#     search_player = st.text_input("Search player:")
#     if search_player:
#         df_show = df_rank[df_rank["batter"].str.contains(search_player, case=False, na=False)]
#     else:
#         df_show = df_rank

#     # Display
#     st.dataframe(
#         df_show[["Rank", "batter", ranking_type]].style.format({ranking_type: "{:.2f}"}),
#         use_container_width=True
#     )


with tab4:
    st.markdown(
        "<div class='section-title' style='color:white;'>Info & More Insights</div>",
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Unlocking All Corners of a Circle")
    st.markdown(
        "[Click here to read](https://arnavj.substack.com/p/unlocking-all-corners-of-a-circle?r=3er7j9)"
    )

    st.markdown("### 📝 Intent-Impact")
    st.markdown(
        "[Click here to read](https://open.substack.com/pub/arnavj/p/solving-the-intent-and-impact-equation?r=3er7j9&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)"
    )

    st.markdown("---")

    st.markdown(
        """
        💡 **For more interesting metrics and insights on cricket**,  
        follow my Substack here:  
        [https://arnavj.substack.com/](https://arnavj.substack.com/)
        """,
        unsafe_allow_html=True
    )    

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.caption("Click ℹ️ to see what each metric means")
st.caption("Percentiles shown are 'percentage of batters this player is better than'.")
st.caption("Profile score is mean of all metric percentiles")
   


