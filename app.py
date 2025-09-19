# app.py
import streamlit as st
import pickle
import os
import numpy as np
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
        {pct:.1f}%
      </div>
    </div>
    """
    return html


# -------------------- Load data --------------------
DATA_DIR = "t20_decay"
required_files = ["fshots.bin", "intents.bin", "negative_dur.bin", "impact_stats.bin"]
for fn in required_files:
    if not os.path.exists(os.path.join(DATA_DIR, fn)):
        raise FileNotFoundError(f"Required file missing: {os.path.join(DATA_DIR, fn)}")

fshots = load_bin(os.path.join(DATA_DIR, "fshots.bin"))
intents = load_bin(os.path.join(DATA_DIR, "intents.bin"))
negative_dur = load_bin(os.path.join(DATA_DIR, "negative_dur.bin"))
impact_stats = load_bin(os.path.join(DATA_DIR, "impact_stats.bin"))

# -------------------- Global batter list (intersection) --------------------
batter_list = sorted(
    set(fshots.keys())
    & set(intents.keys())
    & set(negative_dur.keys())
    & set(impact_stats.keys())
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
        "imp_improv": np.array(imp_improv_a, dtype=float)
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
st.markdown('<div class="subtitle">Intent / Reliability and Impact metrics — search and compare players</div>', unsafe_allow_html=True)

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

# -------------------- Display --------------------
# show small header with batter name
st.markdown(f"### {batter}", unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Int-Rel", "Int Impact"])

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
    st.markdown("<div class='section-title' style='color:white;'>Overall Int-Rel Score</div>", unsafe_allow_html=True)
    st.markdown(colored_bar_html(round(avg_intrel_pct, 2)), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-name' style='color:white;'>Intent (Pace)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intent_pace, f"{round(intent_pace,2) if not np.isnan(intent_pace) else '—'} <span class='small'>({'' if np.isnan(p_intent_pace) else f'{round(p_intent_pace,2)}% better'})</span>"), unsafe_allow_html=True)

        st.markdown("<div class='metric-name' style='color:white;'>Reliability (Pace)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_rel_pace, f"{round(rel_pace,2) if not np.isnan(rel_pace) else '—'} <span class='small'>({'' if np.isnan(p_rel_pace) else f'{round(p_rel_pace,2)}% better'})</span>"), unsafe_allow_html=True)

        st.markdown("<div class='metric-name' style='color:white;'>Int-Rel (Pace)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intrel_pace, f"{round(intrel_pace,2) if not np.isnan(intrel_pace) else '—'} <span class='small'>({'' if np.isnan(p_intrel_pace) else f'{round(p_intrel_pace,2)}% better'})</span>"), unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-name' style='color:white;'>Intent (Spin)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intent_spin, f"{round(intent_spin,2) if not np.isnan(intent_spin) else '—'} <span class='small'>({'' if np.isnan(p_intent_spin) else f'{round(p_intent_spin,2)}% better'})</span>"), unsafe_allow_html=True)

        st.markdown("<div class='metric-name' style='color:white;'>Reliability (Spin)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_rel_spin, f"{round(rel_spin,2) if not np.isnan(rel_spin) else '—'} <span class='small'>({'' if np.isnan(p_rel_spin) else f'{round(p_rel_spin,2)}% better'})</span>"), unsafe_allow_html=True)

        st.markdown("<div class='metric-name' style='color:white;'>Int-Rel (Spin)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intrel_spin, f"{round(intrel_spin,2) if not np.isnan(intrel_spin) else '—'} <span class='small'>({'' if np.isnan(p_intrel_spin) else f'{round(p_intrel_spin,2)}% better'})</span>"), unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-name' style='color:white;'>Intent (Overall)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intent_overall, f"{round(intent_overall,2) if not np.isnan(intent_overall) else '—'} <span class='small'>({'' if np.isnan(p_intent_overall) else f'{round(p_intent_overall,2)}% better'})</span>"), unsafe_allow_html=True)

        st.markdown("<div class='metric-name' style='color:white;'>Reliability (Overall)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_rel_overall, f"{round(rel_overall,2) if not np.isnan(rel_overall) else '—'} <span class='small'>({'' if np.isnan(p_rel_overall) else f'{round(p_rel_overall,2)}% better'})</span>"), unsafe_allow_html=True)

        st.markdown("<div class='metric-name' style='color:white;'>Int-Rel (Overall)</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_intrel_overall, f"{round(intrel_overall,2) if not np.isnan(intrel_overall) else '—'} <span class='small'>({'' if np.isnan(p_intrel_overall) else f'{round(p_intrel_overall,2)}% better'})</span>"), unsafe_allow_html=True)

# ─── TAB 2 ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title' style='color:white;'>Overall Impact Score</div>", unsafe_allow_html=True)
    st.markdown(colored_bar_html(round(avg_impact_pct, 2)), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-name' style='color:white;'>Negative Duration</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_neg_dur, f"{round(neg_dur_v,2) if not np.isnan(neg_dur_v) else '—'} <span class='small'>({'' if np.isnan(p_neg_dur) else f'{round(p_neg_dur,2)}% better'})</span>"), unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-name' style='color:white;'>Per Ball Impact</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_per_ball, f"{round(per_ball_v,2) if not np.isnan(per_ball_v) else '—'} <span class='small'>({'' if np.isnan(p_per_ball) else f'{round(p_per_ball,2)}% better'})</span>"), unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-name' style='color:white;'>Per Innings Impact</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_per_inn, f"{round(per_inn_v,2) if not np.isnan(per_inn_v) else '—'} <span class='small'>({'' if np.isnan(p_per_inn) else f'{round(p_per_inn,2)}% better'})</span>"), unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-name' style='color:white;'>Impact Improvement</div>", unsafe_allow_html=True)
        st.markdown(metric_gradient_html(p_imp_improv, f"{round(imp_improv_v,2) if not np.isnan(imp_improv_v) else '—'} <span class='small'>({'' if np.isnan(p_imp_improv) else f'{round(p_imp_improv,2)}% better'})</span>"), unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.caption("Percentiles shown are 'percentage of batters this player is better than'.")

