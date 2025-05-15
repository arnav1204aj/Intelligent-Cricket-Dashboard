import streamlit as st
st.set_page_config(
    page_title="🏏 Batter Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pickle
import pandas as pd
import numbers
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN  # (imported to satisfy any dependencies)
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
sns.set_theme(style="darkgrid")

# --- Entry Planner Data & Helpers ---
@st.cache_data
def load_planner_data():
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {
        'intent_dict':       load_pickle('data/intents.bin'),
        'p_intent':          load_pickle('data/paceintents.bin'),
        's_intent':          load_pickle('data/spinintents.bin'),
        'p_fshot':           load_pickle('data/pacefshots.bin'),
        's_fshot':           load_pickle('data/spinfshots.bin'),
        'fshot_dict':        load_pickle('data/fshots.bin'),
        'gchar':             load_pickle('data/ground_char.bin'),
        'phase_experience':  load_pickle('data/phase_breakdown.bin'),
        'negdur':            load_pickle('data/negative_dur.bin'),
        'bat_rel':           load_pickle('data/bat_relief.bin')
    }

planner = load_planner_data()
intent_dict      = planner['intent_dict']
p_intent         = planner['p_intent']
s_intent         = planner['s_intent']
p_fshot          = planner['p_fshot']
s_fshot          = planner['s_fshot']
fshot_dict       = planner['fshot_dict']
gchar            = planner['gchar']
phase_experience = planner['phase_experience']
negdur           = planner['negdur']
batter_stats     = planner['bat_rel']

# phase mapping
phase_mapping = {
    i: "Powerplay (1-6 overs)"    if i <= 6 else
       "Middle (7-11 overs)"      if i <= 11 else
       "Middle (12-16 overs)"     if i <= 16 else
       "Death (17-20 overs)"
    for i in range(1, 21)
}


def plot_intent_impact(batter_name):
    if batter_name not in batter_stats:
        print("Batter not found.")
        return None
    
    batter_data        = batter_stats[batter_name]
    counts             = batter_data["batter_ith_ball_count"]
    total_runs         = batter_data["batter_ith_ball_total_runs"]
    non_striker_runs   = batter_data["non_striker_ith_ball_total_runs"]
    
    valid_balls = sorted(i for i in counts if counts[i] >= 5)
    if not valid_balls:
        print("No data with count >= 5.")
        return None
    
    batter_rpb      = np.array([total_runs[i]/counts[i] for i in valid_balls])
    non_striker_rpb = np.array([non_striker_runs[i]/counts[i] for i in valid_balls])
    intent_impact   = np.cumsum(batter_rpb - non_striker_rpb)
    
    min_balls_no_negative = None
    for idx, val in enumerate(intent_impact):
        if val >= 0 and np.all(intent_impact[idx:] >= 0):
            min_balls_no_negative = valid_balls[idx]
            break

    # --- create fig & ax explicitly ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    sns.lineplot(
        x=valid_balls,
        y=intent_impact,
        linewidth=2.5,
        label=f'{batter_name} Intent Impact',
        color='royalblue',
        ax=ax
    )
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    if min_balls_no_negative is not None:
        ax.axvline(
            min_balls_no_negative,
            color='crimson',
            linestyle='--',
            linewidth=2,
            label=f'Neg. Impact Duration – {min_balls_no_negative} Balls'
        )
    ax.set_xlabel("Balls Faced", fontsize=14, fontweight='bold')
    ax.set_ylabel("Cumulative Intent Impact", fontsize=14, fontweight='bold')
    ax.set_title(f"Intent Impact Progression for {batter_name}", fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, loc="upper left")
    fig.tight_layout()
    
    return fig


def plot_int_wagons2(batter, minmag, max_magnitude, percentile):
    bin_file_path = f'{match_type}/vectorsnorm.bin'

    with open(bin_file_path, 'rb') as bin_file:
        vecdic = pickle.load(bin_file)

    vectors = vecdic[batter]['evs']
    list1 = vecdic[batter]['vectors']
    bf = len(vectors)
    normal_vecs = []
    boundary_vecs = []
    all_plotted_vecs = []

    for i in range(len(list1)):
        if np.linalg.norm(vectors[i]) > np.linalg.norm(list1[i]) and np.linalg.norm(list1[i]) >= minmag:
            vec = vectors[i]
            norm = np.linalg.norm(vec)
            clipped = (
                vec[0] if norm <= max_magnitude else max_magnitude * (vec[0] / norm),
                vec[1] if norm <= max_magnitude else max_magnitude * (vec[1] / norm)
            )
            all_plotted_vecs.append(clipped)
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

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('green')
    ax.set_facecolor('green')
    ax.axis('off')

    # Plot boundary shots first
    plt.quiver(
        origin_x[len(normal_vecs):], origin_y[len(normal_vecs):],
        x_boundary, y_boundary,
        angles='xy', scale_units='xy', scale=1, color='red', alpha=0.9,
        headwidth=1, headlength=0, label='Boundaries (4s/6s)'
    )

    # Plot normal shots
    plt.quiver(
        origin_x[:len(normal_vecs)], origin_y[:len(normal_vecs)],
        x_normal, y_normal,
        angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7,
        headwidth=1, headlength=0
    )

    # Circular max magnitude boundary
    circle = plt.Circle((0, 0), max_magnitude, color='white', fill=False, linewidth=2)
    ax.add_artist(circle)

    # Percentile circle
    if all_plotted_vecs:
        percentile_magnitude = np.percentile([np.linalg.norm([v[0], v[1]]) for v in all_plotted_vecs], percentile)
        percentile_magnitude = 7.93
        intersection_points = []

        for vec in all_plotted_vecs:
            norm = np.linalg.norm(vec)
            if norm >= percentile_magnitude:
                scale = percentile_magnitude / norm
                intersect_point = (vec[0] * scale, vec[1] * scale)
                intersection_points.append(intersect_point)

        if len(intersection_points) > 1:
            intersection_points = np.array(intersection_points)
            angles = np.arctan2(intersection_points[:, 1], intersection_points[:, 0])
            sorted_indices = np.argsort(angles)
            sorted_points = intersection_points[sorted_indices]

            arc_chain = [sorted_points[0]]
            arc_lines = []

            for i in range(1, len(sorted_points)):
                prev = sorted_points[i - 1]
                curr = sorted_points[i]
                dist = np.linalg.norm(curr - prev)

                if dist < 2*(50/np.sqrt(len(vectors))):
                    arc_chain.append(curr)
                else:
                    if len(arc_chain) >= 2:
                        arc_chain_np = np.array(arc_chain)
                        dists = np.linalg.norm(np.diff(arc_chain_np, axis=0), axis=1)
                        if np.sum(dists) >= 2:
                            arc_lines.append(arc_chain)
                    arc_chain = [curr]

            if len(arc_chain) >= 2:
                arc_chain_np = np.array(arc_chain)
                dists = np.linalg.norm(np.diff(arc_chain_np, axis=0), axis=1)
                if np.sum(dists) >= 2:
                    arc_lines.append(arc_chain)

            for chain in arc_lines:
                xs, ys = zip(*chain)
                ax.plot(xs, ys, color='black', linewidth=2)

    ax.arrow(0, -10, 0, 10, color='white', width=0.2, head_width=1, head_length=1, length_includes_head=True)
    ax.text(0, -12, "Batter Facing", color='white', fontsize=12, ha='center')

    ax.set_xlim(-max_magnitude, max_magnitude)
    ax.set_ylim(-max_magnitude, max_magnitude)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

    legend = ax.legend(loc='upper right', frameon=True, facecolor='white')
    black_line_legend = Line2D([0], [0], color='black', linewidth=2, label='Regions of Strength')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [black_line_legend], labels + ['Regions of Strength'],
              loc='upper right', frameon=True, facecolor='white')
    for text in legend.get_texts():
        text.set_color('black')

    plt.suptitle(f"{batter} Intelligent Wagon Wheel", fontsize=16, color='white', weight='bold')
    # The function shows the plot as-is
    plt.show()

def get_top_3_overs(batter, ground_name, num_spinners, num_pacers, n):
    acc = np.zeros(120)
    for s_ball in range(120):
        bfaced = 0
        intent = 0
        for ball in range(s_ball, 120):
            bfaced += 1
            overnum = (ball // 6) + 1
            phase = phase_mapping[overnum]
            paceweight = np.power(gchar[ground_name][overnum - 1] / 100, 1.5)
            spinweight = np.power(1 - gchar[ground_name][overnum - 1] / 100, 1.5)
            spin_prob = spinweight * (num_spinners / (num_spinners + num_pacers))
            pace_prob = paceweight * (num_pacers / (num_spinners + num_pacers))
            total_prob = pace_prob + spin_prob
            pace_prob /= total_prob
            spin_prob /= total_prob

            def get_metric(intent_data, fallback, key):
                try:
                    if intent_data['othbatballs'][overnum - 1] == 0 \
                    or intent_data['batballs'][overnum - 1] == 0 \
                    or intent_data['othbatruns'][overnum - 1] == 0:
                        return fallback[batter][key]['1-20']
                    return (intent_data['batruns'][overnum - 1] /
                            intent_data['batballs'][overnum - 1]) / \
                           (intent_data['othbatruns'][overnum - 1] /
                            intent_data['othbatballs'][overnum - 1])
                except:
                    return fallback[batter][key]['1-20']

            def get_fshot(fshot_data, fallback, key):
                try:
                    if fshot_data['othbatballs'][overnum - 1] == 0 \
                    or fshot_data['batballs'][overnum - 1] == 0 \
                    or fshot_data['othbatshots'][overnum - 1] == 0 \
                    or fshot_data['batshots'][overnum - 1] == 0:
                        return fallback[batter][key]['1-20']
                    return (fshot_data['batshots'][overnum - 1] /
                            fshot_data['batballs'][overnum - 1]) / \
                           (fshot_data['othbatshots'][overnum - 1] /
                            fshot_data['othbatballs'][overnum - 1])
                except:
                    return fallback[batter][key]['1-20']

            spin_int = get_metric(s_intent[batter], intent_dict, 'spin')
            pace_int = get_metric(p_intent[batter], intent_dict, 'pace')
            spin_fs  = get_fshot(s_fshot[batter], fshot_dict, 'spin')
            pace_fs  = get_fshot(p_fshot[batter], fshot_dict, 'pace')

            if bfaced <= negdur[batter]:
                spin_int = pace_int = 0
            if spin_int < 0.95:
                spin_int = 0
            if pace_int < 0.95:
                pace_int = 0

            phase_w = phase_experience[batter][phase] / 100
            intent += (
                pace_int * phase_w * pace_prob / np.sqrt(pace_fs) +
                spin_int * phase_w * spin_prob / np.sqrt(spin_fs)
            )

        acc[s_ball] = intent / (120 - s_ball)

    over_avgs    = [np.mean(acc[i:i+6]) for i in range(0, 120, 6)]
    top_indices  = np.argsort(over_avgs)[-n:][::-1]
    return [(i+1, over_avgs[i]) for i in top_indices]


# --- Main Dashboard ---


# --- Load core stats ---
FILES = {
    "Intent & Reliability":      "data/t20_intent_reliability_stats.bin",
    "Intent Impact Metrics":     "data/batter_intent_stats.bin",
    "360° Shot Metrics":         "data/t20_batter_stats_360.bin",
    "Entropy Focus":             "data/t20_entropy_focus.bin",
    "spin durations":            "data/negative_dur_spin.bin",
    "pace durations":            "data/negative_dur_pace.bin",
    'metric_percentiles':        "data/metric_percentiles.bin"

    


}

@st.cache_data
def load_core_stats(files_map):
    stats = {}
    # Load each .bin into stats[label]
    for label, path in files_map.items():
        p = Path(path)
        if not p.exists():
            st.error(f"❌ File not found: {path}")
            stats[label] = {}
        else:
            with open(p, "rb") as f:
                stats[label] = pickle.load(f)

    # Build list of key‐sets for all sections *except* metric_percentiles
    relevant_sets = [
        set(d.keys())
        for label, d in stats.items()
        if label != "metric_percentiles" and isinstance(d, dict)
    ]

    # Compute intersection only over those relevant sets
    common = sorted(set.intersection(*relevant_sets)) if relevant_sets else []

    return stats, common


stats_dicts, common_batters = load_core_stats(FILES)

st.sidebar.title("Select Batter")
if not common_batters:
    st.sidebar.warning("No common batters found.")
    st.stop()
batter = st.sidebar.selectbox("Batter", common_batters)

# rename maps omitted for brevity (use same as before)
rename_maps = {
    "Intent & Reliability": {
        "pace_intent":      "Pace Intent",
        "pace_reliability": "Pace Reliability",
        "pace_int_rel":     "Pace Int-Rel",
        "spin_intent":      "Spin Intent",
        "spin_reliability": "Spin Reliability",
        "spin_int_rel":     "Spin Int-Rel",
        "avg_intent":       "Avg Intent",
        "avg_reliability":  "Avg Reliability",
        "avg_int_rel":      "Avg Int-Rel",
    },
    "Intent Impact Metrics": {
        "final_intent_impact":      "Final Intent Impact",
        "impact_acceleration":      "Impact Acceleration",
        "negative_impact_duration": "Neg. Impact Duration",
        "impact_improvement":       "Impact Improvement",
        "improvement_rate":         "Improvement Rate",
        "improvement_pct":          "Improvement %",
    },
    "360° Shot Metrics": {
        "score360":               "360 Score",
        "ground_coverage_pct":    "Ground Coverage",
        "audacity":               "Difficult Shot Execution",
        "aggressive_bp_pct":      "Boundary % on Difficult Shots",
        "overall_bp_pct":         "Overall Boundary %",
        "diff_shot_effect":       "Difficult Shot Effect",
    },
    "Entropy Focus": {
        "primary_focus":     f"Primary Focus against {batter}",
        "least_affected_by": "Least Affected By",
    }
}

def make_df(section_label, entry, percentiles, batter):
    # build base df
    if section_label=='Entropy Focus':
        colval = 'Value'
    else:
        colval = "Value (Percentile)" 
    df = pd.DataFrame.from_dict(entry, orient="index", columns=[colval])
    df.index.name = "Metric"
    df = df.rename(index=rename_maps.get(section_label, {}))

    # for each row, if numeric, append percentile in brackets
    def fmt(val, raw_metric):
        if isinstance(val, numbers.Number):
            p = percentiles.get(section_label, {}) \
                           .get(batter, {}) \
                           .get(raw_metric, None)
            if p is not None:
                return f"{val:.2f} ({p:.0f}%)"
            else:
                return f"{val:.2f}"
        return val

    # apply formatting: need raw_metric names in same order as df.index
    # so map df.index back to raw_metric via invert rename_map
    inv_map = {v:k for k,v in rename_maps.get(section_label, {}).items()}
    raw_metrics = [inv_map.get(m, m) for m in df.index]
    df[colval] = [
        fmt(df.loc[df.index[i], colval], raw_metrics[i])
        for i in range(len(df))
    ]

    return df


tabs = st.tabs([
    "Overview",
    "Intent & Reliability",
    "Intent Impact Metrics",
    "360° Shot Metrics",
    "Strategy Values"
])

with tabs[0]:
    st.subheader(f"Overview: {batter}")

    # --- compute three average percentiles ---
    pct = stats_dicts["metric_percentiles"]
    ir_vals     = list(pct["Intent & Reliability"].get(batter, {}).values())
    imp_vals    = list(pct["Intent Impact Metrics"].get(batter, {}).values())
    shot_vals   = list(pct["360° Shot Metrics"].get(batter, {}).values())
    ir_avg  = np.mean(ir_vals)  if ir_vals  else 0
    imp_avg = np.mean(imp_vals) if imp_vals else 0
    sh_avg  = np.mean(shot_vals) if shot_vals else 0

    # --- gauge helper ---
    # --- gauge helper with transparent background ---
    import matplotlib.cm as cm

    def gauge_chart(value, title):
        # create figure & axes with no background
        fig, ax = plt.subplots(
            figsize=(2.2, 2.2),
            subplot_kw={'aspect':'equal'},
            facecolor='none'               # figure background transparent
        )
        ax.set_facecolor('none')          # axes background transparent

        cmap = cm.get_cmap('YlOrRd')
        color = cmap(value / 100)

        # donut: [value, remainder]
        ax.pie(
            [value, 100 - value],
            colors=[color, 'lightgray'],
            startangle=90,
            counterclock=False,
            wedgeprops={'width':0.3, 'edgecolor':'white'}
        )

        ax.annotate(
            f"{value:.0f}%", xy=(0,0),
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            color='white'
        )
        ax.set_title(title, fontsize=10, pad=12, color='white')
        ax.axis('off')

        # ensure the figure patch is transparent
        fig.patch.set_alpha(0)
        return fig


    # --- render three gauges ---
    c1, c2, c3 = st.columns(3)
    c1.pyplot(gauge_chart(ir_avg,  "Intent & Reliability Profile"))
    c2.pyplot(gauge_chart(imp_avg, "Impact Profile"))
    c3.pyplot(gauge_chart(sh_avg,  "360° Profile"))

    st.markdown("Use the tabs above to dive into each analysis in detail.")

with tabs[1]:
    st.subheader("Intent & Reliability")
    df_ir = make_df(
    "Intent & Reliability",
    stats_dicts["Intent & Reliability"][batter],
    stats_dicts["metric_percentiles"],
    batter
    )
    st.dataframe(df_ir, use_container_width=True)

    

with tabs[2]:
    st.subheader("Intent Impact Metrics")
    df_ir = make_df(
    "Intent Impact Metrics",
    stats_dicts["Intent Impact Metrics"][batter],
    stats_dicts["metric_percentiles"],
    batter
    )  
    st.dataframe(df_ir, use_container_width=True)

    

    st.markdown("### Intent Impact Progression")
    fig = plot_intent_impact(batter)
    if fig:
        st.pyplot(fig)
    else:
        st.write("_No plot available._")



with tabs[3]:
    st.subheader("360° Shot Metrics")
    df_ir = make_df(
    "360° Shot Metrics",
    stats_dicts["360° Shot Metrics"][batter],
    stats_dicts["metric_percentiles"],
    batter
    )
    st.dataframe(df_ir, use_container_width=True)


    # ======== Here is your unmodified plot call ========
    # Set these to whatever you were using alongside your function
    match_type    = 'data'
    minmag        = 0
    max_magnitude = 35
    percentile    = 90

    # Generate and render the plot exactly as in your code
    plot_int_wagons2(batter, minmag, max_magnitude, percentile)
    st.pyplot(plt)

with tabs[4]:
    st.subheader("Key Performance Factors")
    st.dataframe(make_df("Entropy Focus", stats_dicts["Entropy Focus"][batter],stats_dicts["metric_percentiles"],
    batter), use_container_width=True)

    # ── New: Time to Settle metrics ──
    spin_settle = stats_dicts["spin durations"].get(batter, None)
    pace_settle = stats_dicts["pace durations"].get(batter, None)
    col1, col2 = st.columns(2)
    col1.metric(
        "Time to Settle vs Spin",
        f"{int(spin_settle)} balls" if spin_settle is not None else "N/A"
    )
    col2.metric(
        "Time to Settle vs Pace",
        f"{int(pace_settle)} balls" if pace_settle is not None else "N/A"
    )

    # ── Entry Points UI ──
    st.markdown("### Entry Point Calculator")
    ground_list = ["Neutral Venue"] + [g for g in gchar.keys() if g != "Neutral Venue"]
    ground      = st.selectbox("Select Ground", ground_list, key="entropy_ground")
    num_spinners = st.slider("Opposition Spinners", 0, 6, 2, key="entropy_spin")
    num_pacers   = st.slider("Opposition Pacers",   0, 6, 4, key="entropy_pace")

    if st.button("Calculate Entry Point", key="entropy_btn"):
        if num_spinners == 0 and num_pacers == 0:
            st.error("❗ Please select at least one spinner or pacer.")
        else:
            entries = get_top_3_overs(batter, ground, num_spinners, num_pacers, 3)
            st.markdown("---")
            st.markdown("#### Top 3 Optimal Entry Overs")
            medals = ["🥇", "🥈", "🥉"]
            for idx, (over_num, score) in enumerate(entries):
                st.markdown(f"""
                    <div style="
                        background-color:#f0f2f6;
                        padding:12px 20px;
                        margin-bottom:10px;
                        border-left:6px solid #1e90ff;
                        border-radius:8px;
                    ">
                        <h5 style="margin:0; color:#222;">{medals[idx]} <b>Over {over_num}</b></h5>
                        <p style="margin:0; color:#444;">
                            Acceleration Score: <code style="font-size:16px;">{score:.4f}</code>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

