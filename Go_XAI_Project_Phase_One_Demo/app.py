import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import time
import glob
import os
from sgfmill import sgf

# -------------------------------------------------------------------------
#  PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Go XAI Project",
    layout="wide",
    page_icon="assets/favicon.ico"
)

# -----  INJECT GOTHIC–ARCANE–TECH THEME (FONTS +   CSS--------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
:root {
  --bg: #040404;
  --bg2: #0A0F0E;
  --panel: #1C1F1E;
  --text: #F9F9F9;
  --accent: #0EA66B;
  --accent-glow: #94FFD9;
  --silver: #BDC3C7;
  --danger: #7A1A1A;
}

/* GLOBAL BACKGROUND */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
}

/* HEADERS */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--accent-glow) !important;
    letter-spacing: 0.5px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--silver);
}

/* CARDS */
.card {
    background: var(--panel);
    border: 1px solid var(--silver);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 20px rgba(0, 255, 175, 0.12);
}

/* EXPANDERS */
.streamlit-expanderHeader {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--accent-glow) !important;
}

.streamlit-expanderContent {
    background: var(--bg2) !important;
}

/* METRIC BOX */
[data-testid="stMetric"] {
    background-color: var(--panel);
    padding: 10px;
    border-radius: 10px;
    border: 1px solid var(--silver);
}

/* BUTTONS */
div.stButton > button:first-child {
    background-color: var(--accent);
    color: black;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    border: none;
    font-weight: 600;
    transition: 0.2s ease-in-out;
}

div.stButton > button:hover {
    background-color: var(--accent-glow);
    color: #000000;
    transform: scale(1.03);
}

/* INPUT FIELDS */
input, textarea, select {
    background-color: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--silver) !important;
}

/* SCROLLBAR */
::-webkit-scrollbar-thumb {
    background: var(--accent);
}

/* DIVIDERS */
hr {
    border-color: var(--silver);
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg2) !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--silver) !important;
    font-family: 'Cormorant Garamond', serif !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-glow) !important;
    border-bottom-color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# HELPER: RESET STATE
# -------------------------------------------------------------------------
def reset_state():
    st.session_state.analysis_done = False
    st.session_state.current_explanation = ""

# -------------------------------------------------------------------------
#  SCENARIO DATABASE
# -------------------------------------------------------------------------
scenario_data = {
    "Scenario A": {
        "name": "Opening Direction",
        "stones": [(3, 16, 'W'), (16, 3, 'B'), (15, 16, 'B'), (3, 3, 'W')],
        "ai_move": (16, 14),
        "who_am_i": "You are playing **White**.",
        "situation": "Fuseki Phase with two corners owned.",
        "problem": "Black threatens a Shimari enclosure at the top-right.",
        "rate": "12.5%",
        "explanations": {
            "Novice (18k-10k)": "Play at Q15 to stop Black from taking the corner.",
            "Intermediate (9k-1d)": "Approach at Q15 to prevent the P16/Q15 enclosure.",
            "Expert (2d+)": "Q15 reduces influence and avoids a 12.5% efficiency loss."
        }
    },

    "Scenario B": {
        "name": "Life & Death – Vital Point",
        "stones": [(0,0,'B'),(1,0,'B'),(2,0,'B'),(3,0,'B'), (0,1,'B'), (3,1,'B'),
                   (0,2,'W'), (1,2,'W'), (2,2,'W'), (3,2,'W')],
        "ai_move": (1, 1),
        "who_am_i": "You are playing **Black**.",
        "situation": "Your corner group is surrounded with no eyes.",
        "problem": "White will kill the group by occupying the vital eye-making point.",
        "rate": "21.8%",
        "explanations": {
            "Novice (18k-10k)": "Play at B2 to form two eyes.",
            "Intermediate (9k-1d)": "B2 is the vital point for dividing eye space.",
            "Expert (2d+)": "Rectangular-4 nakade: occupy the symmetry point."
        }
    },

    "Scenario C": {
        "name": "Shape Vital Point",
        "stones": [(10,10,'B'), (9,10,'B'), (10,9,'B'), (8,11,'W'), (11,8,'W')],
        "ai_move": (9, 9),
        "who_am_i": "You are playing **Black**.",
        "situation": "Mid-fight with thin shape structure.",
        "problem": "White is threatening to cut your formation.",
        "rate": "13.3%",
        "explanations": {
            "Novice (18k-10k)": "Connect to avoid being cut.",
            "Intermediate (9k-1d)": "J10 is the strongest shape connection.",
            "Expert (2d+)": "Tiger’s Mouth is optimal against peeps."
        }
    }
}

# -------------------------------------------------------------------------
# --- ROBUST SGF DATA LOADER ---
# -------------------------------------------------------------------------
@st.cache_data
def load_sgf_data(folder_path, recursive=False, limit=5000):
    records = []

    if recursive:
        search_path = f"{folder_path}/**/*.sgf"
    else:
        search_path = f"{folder_path}/*.sgf"

    files = glob.glob(search_path, recursive=recursive)[:limit]
    if not files:
        return pd.DataFrame()

    progress_bar = st.progress(0)

    for i, filepath in enumerate(files):
        try:
            with open(filepath, "rb") as f:
                raw = f.read()
                sgf_text = raw.decode('utf-8', errors='replace')
                game = sgf.Sgf_game.from_string(sgf_text)
                root = game.get_root()

                result = root.get("RE") if root.has_property("RE") else "Unknown"
                b_rank = root.get("BR") if root.has_property("BR") else "Unknown"
                w_rank = root.get("WR") if root.has_property("WR") else "Unknown"
                b_name = root.get("PB") if root.has_property("PB") else "Unknown"
                w_name = root.get("PW") if root.has_property("PW") else "Unknown"

                winner_color = "Draw"
                if str(result).startswith("B"): winner_color = "Black"
                if str(result).startswith("W"): winner_color = "White"

                move_count = len(game.get_main_sequence()) - 1
                folder_rank = os.path.basename(os.path.dirname(filepath))

                records.append({
                    "Filename": os.path.basename(filepath),
                    "Winner Color": winner_color,
                    "Black Player": b_name,
                    "White Player": w_name,
                    "Black Rank": b_rank,
                    "White Rank": w_rank,
                    "Moves": move_count,
                    "Folder Category": folder_rank 
                })

        except Exception:
            pass

        if i % 20 == 0:
            progress_bar.progress((i+1) / len(files))

    progress_bar.empty()
    return pd.DataFrame(records)

# -------------------------------------------------------------------------
# --- DRAW GO BOARD ---
# -------------------------------------------------------------------------
def draw_board_with_overlay(stones, highlight=None, heatmap_active=False, heatmap_type="Influence"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('#DCB35C')

    for i in range(19):
        ax.plot([i, i], [0, 18], 'k-', lw=0.5, alpha=0.6)
        ax.plot([0, 18], [i, i], 'k-', lw=0.5, alpha=0.6)

    stars = [(3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)]
    for sx, sy in stars:
        ax.plot(sx, sy, 'ko', markersize=3)

    if heatmap_active:
        x, y = np.meshgrid(np.linspace(-1, 19, 100), np.linspace(-1, 19, 100))
        z = np.zeros_like(x)

        if heatmap_type == "Influence":
            for sx, sy, color in stones:
                if color == 'B':
                    z += np.exp(-((x - sx)**2 + (y - sy)**2)/3.0)
            cmap = 'Greens'
        else:
            for sx, sy, color in stones:
                if color == 'W':
                    z += np.exp(-((x - sx)**2 + (y - sy)**2)/3.0)
            cmap = 'Reds'

        ax.imshow(z, extent=[-1,19,-1,19], origin='lower', cmap=cmap, alpha=0.4)

    for x_, y_, color in stones:
        c = 'black' if color=='B' else 'white'
        edge = 'white' if color=='B' else 'black'
        circle = plt.Circle((x_,y_), 0.45, color=c, ec=edge, zorder=3)
        ax.add_artist(circle)

    if highlight:
        hx, hy = highlight
        rect = plt.Rectangle((hx-0.5, hy-0.5), 1,1,
                             linewidth=2.5, edgecolor='#FF0000', facecolor='none', zorder=4)
        ax.add_patch(rect)

    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 19)
    ax.axis('off')
    return fig

# -------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------------
st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.title("Beyond the Move")

page = st.sidebar.radio("Navigation", ["1. Home", "2. Interactive Demo", "3. Dataset Explorer", "4. Project Roadmap & Researcher"])

st.sidebar.divider()
st.sidebar.subheader("Explanation Level")
game_rank = st.sidebar.select_slider(
    "Go Game Rank:",
    options=["Novice (18k-10k)", "Intermediate (9k-1d)", "Expert (2d+)"]
)
st.sidebar.info(f"Explanation Mode: **{game_rank.split()[0]}**")

# -------------------------------------------------------------------------
# PAGE 1 — HOME
# -------------------------------------------------------------------------
if page == "1. Home":
    
    st.title("Beyond the Move")
    st.subheader("An Explainable AI (XAI) Framework for Cognitive Skill Acquisition in the Game of Go")
    colx1, colx2 = st.columns([1,1])
    colx1.caption("By Ipek N. Sipahi ")  
    colx2.caption("Advised by  Dr. Gamze Türkmen ")
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Project Overview")  
        st.markdown("""
                    While AI has mastered the game of Go, it often acts as an oracle—telling us what to play, but not *why*. 
                    This project explores the integration of Explainable AI (XAI) techniques into move recommendation systems.

                    ### Research Question
                    > **How can Explainable AI methods be used to make AI move recommendations in strategy games more interpretable and human-like?**

                    **Research Goal:** To create a "Glass Box" AI that visualizes its strategic reasoning, helping human players improve their intuition rather than just memorizing computer moves.
                """)
        
    with col2:
        col_x1, col_x2 = st.columns([1,2])
        col_x1.image("assets/ai_of_go.png",use_container_width=True)
        col_x1.caption("Figure 1: Credits to Pasific Standart - The Artificial Intelligence That Solved Go")
        col_x2.subheader("AIfication in Go")
        col_x2.markdown("""
                        **"AIfication"** describes the evolution of Game AI.  
                        Initially, engines relied on brute-force calculation.  
                        Next, Deep Learning agents like AlphaGo achieved superhuman *intuition*  
                        but remained opaque "Black Boxes" ([Park, 2022](https://www.mdpi.com/2409-9287/7/3/55)).

                        **The current frontier shifts from winning → to Explainability.**  
                        This project addresses this *third wave*, using XAI to transform AI from an inscrutable **Oracle** into a collaborative **Tutor** that facilitates human cognitive skill acquisition.
                    """)
        st.markdown("""
                    ### Why Go?
                    The game of Go was selected as the primary domain for this research because it represents the ultimate benchmark for intuitive AI. Unlike Chess, which relies heavily on tactical calculation, Go requires visual pattern recognition and spatial intuition—cognitive processes that align closely with how Convolutional Neural Networks (CNNs) operate. Furthermore, while systems like AlphaGo have solved the problem of performance ($10^{170}$ possible states), they have created a massive 'Interpretability Gap.' By tackling Go, we address the most prominent example of a 'Black Box' super-intelligence, allowing us to test whether XAI can translate mathematical probability into human-understandable concepts like 'Shape' and 'Influence'.""")
# -------------------------------------------------------------------------
# PAGE 2 — INTERACTIVE DEMO
# -------------------------------------------------------------------------
elif page == "2. Interactive Demo":
    st.header("XAI-Powered Go Move Analysis Demo")

    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    tab1, tab2 = st.tabs(["Scenario Library", "Upload SGF"])

    stones, ai_move, explanation = [], None, ""
    who_am_i = situation = problem = ""
    rate = ""

    # TAB 1 — Scenario Library
    with tab1:
        st.markdown("### Select a Learning Scenario")

        scenario = st.selectbox("Choose Scenario:", list(scenario_data.keys()), on_change=reset_state)
        data = scenario_data[scenario]

        stones     = data["stones"]
        ai_move    = data["ai_move"]
        who_am_i   = data["who_am_i"]
        situation  = data["situation"]
        problem    = data["problem"]
        rate       = data["rate"]
        explanation = data["explanations"][game_rank]

        with st.expander("ℹ️ Context", expanded=True):
            st.markdown(f"**Player:** {who_am_i}")
            st.markdown(f"**Situation:** {situation}")
            st.markdown(f"**Problem:** {problem}")

    # TAB 2 — SGF Upload
    with tab2:
        st.markdown("### Analyze Your Own Game")

        uploaded_file = st.file_uploader("Upload SGF", type=['sgf'], on_change=reset_state)

        if uploaded_file:
            st.success(f"Loaded: {uploaded_file.name}")
            
            # Dummy example – replace with your future model inference
            stones = [(4,4,'B'), (16,4,'W'), (4,16,'W'), (16,16,'B'), (3,14,'B')]
            ai_move = (5,2)
            rate = "16.8%"
            explanation = "Tiger’s Mouth is effective here to secure shape."

    # MAIN UI — Board + Analysis
    c1, c2 = st.columns([1.5,1])

    with c1:
        st.subheader("Board Visualization")

        col_x1, col_x2 = st.columns(2)
        show_influence = col_x1.checkbox("Influence Map")
        show_weakness = col_x2.checkbox("Weakness Map")

        heatmap_mode = "Influence" if show_influence else "Weakness" if show_weakness else None
        is_active = bool(heatmap_mode)

        highlight_move = ai_move if st.session_state.analysis_done else None

        if stones:
            fig = draw_board_with_overlay(
                stones,
                highlight=highlight_move,
                heatmap_active=is_active,
                heatmap_type=heatmap_mode
            )
            st.pyplot(fig)

    with c2:
        st.subheader("AI Suggestion")

        if st.button("Suggest Move", type="primary"):
            with st.spinner("Thinking..."):
                time.sleep(0.8)
            st.session_state.analysis_done = True
            st.rerun()

        if st.session_state.analysis_done:
            st.success(f"**Recommended Move:** {ai_move}")
            st.metric("Win Rate Impact", rate)
            st.markdown("### Deep Analysis")
            st.info(explanation)

# -------------------------------------------------------------------------
# PAGE 3 — PROFESSIONAL DATASET EXPLORER
# -------------------------------------------------------------------------
elif page == "3. Dataset Explorer":
    st.header(" Partial Dataset Analytics")

    root_folder = "data/Pro"
    
    if not os.path.exists(root_folder):
        st.error(f" Folder '{root_folder}' not found.")
    else:
        subfolders = sorted([f.name for f in os.scandir(root_folder) if f.is_dir()])

        mode = st.radio("Select Scope:", ["Specific Rank Folder", " All Pro Rank Data"], horizontal=True)
        df = pd.DataFrame()

        if mode == "Specific Rank Folder":
            if subfolders:
                selected_rank = st.selectbox("Select Rank:", subfolders)
                target_path = os.path.join(root_folder, selected_rank)
                df = load_sgf_data(target_path, limit=5000)
        else:
            df = load_sgf_data(root_folder, recursive=True, limit=5000)

        if not df.empty:
            st.divider()

            m1, m2, m3, m4 = st.columns(4)
            total_games = len(df)
            black_wins = len(df[df["Winner Color"] == "Black"])
            white_wins = len(df[df["Winner Color"] == "White"])
            avg_moves = int(df["Moves"].mean())

            m1.metric("Total Games", f"{total_games:,}")
            m2.metric("Black Wins", f"{black_wins} ({black_wins/total_games:.1%})")
            m3.metric("White Wins", f"{white_wins} ({white_wins/total_games:.1f}%)")
            m4.metric("Avg Moves", f"{avg_moves}")

            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader(" Win Rate by Rank Level")
                if "Folder Category" in df.columns:
                    win_dist = df.groupby("Folder Category")["Winner Color"].value_counts().unstack()
                    st.bar_chart(win_dist)

            with c2:
                st.subheader(" Game Length Distribution")
                st.bar_chart(df["Moves"].value_counts().sort_index(), color="#408A67")

            st.subheader(" Player Registry")
            st.dataframe(
                df[["Black Player", "Black Rank", "White Player", "White Rank",
                    "Winner Color", "Moves", "Folder Category"]],
                use_container_width=True
            )

# -------------------------------------------------------------------------
# PAGE 4 — About Me & Future Plans
# -------------------------------------------------------------------------
# --- PAGE 4: ROADMAP & PROFILE ---
elif page == "4. Project Roadmap & Researcher":
    st.title("Project Roadmap & Researcher Profile")

    col1, col2 = st.columns([2,1])
    with col1:
# --- SECTION 1: PROJECT DEVELOPMENT PLAN ---
        st.header("1. Project Development Roadmap")
        st.markdown("""
        This research is structured into two development phases, designed to transition from a technical baseline to a human-centric evaluation of AI interpretability.
        """)

        # --- PHASE I (Current) ---
        st.subheader("Phase I: Baseline Architecture & Interaction Primitives (Deadline: Jan 15, 2026)")
        st.markdown("""
        **Focus:** Establishing a supervised learning baseline and validating the visual explanation pipeline.

        * **Milestone 1 (Dec 1 - Dec 15): Tensor Representation & Feature Engineering.**
        Construction of a high-dimensional input tensor ($19 \\times 19 \\times 17$) encoding expert domain knowledge (liberties, ladder status, and history) to align model input with human perception.
        * **Milestone 2 (Dec 15 - Jan 5): ResNet Implementation.**
        Training a 5-Block Residual Neural Network (ResNet) to mitigate vanishing gradients. Success Metric: Achieving >80% Top-5 Accuracy on the FoxGo validation set.
        * **Milestone 3 (Jan 5 - Jan 15): Axiomatic Attribution.**
        Integration of **Integrated Gradients (Captum)** to generate attribution maps that satisfy the axioms of Sensitivity and Implementation Invariance.
        """)
    
        st.progress(0.60)
        st.caption("Current Status: Data Ingestion & UI Prototyping Complete. Model Training in Progress.")

        # --- PHASE II (The "Ambitious" Part - Professors will love this) ---
        st.subheader("Phase II: Topological Learning & Cognitive Evaluation (Feb - June 2026)")
        st.markdown("""
        **Focus:** Extending the architecture to Graph Neural Networks (GNNs) and measuring pedagogical efficacy.

        * **Research Question:** Can topological graph representations (GCN/GAT) capture "Aji" (latent potential) better than grid-based CNNs?
        * **Methodology:** Representing the Go board as a sparse graph of connected stone clusters rather than a dense image grid.
        * **Validation:** A pilot "Cognitive Load" study comparing novice players' ability to solve Tsumego (puzzles) when assisted by Heatmaps vs. standard AI win-rates.
        """)

    with col2:

    # --- SECTION 2: RESEARCHER PROFILE ---
        st.image("assets/researcher_avatar.png", use_container_width=True)
    
        st.subheader("İpek Naz Sipahi / Aidoneus")
        st.markdown("**B.E. Candidate, Computer Engineering**")
        st.markdown("*Manisa Celal Bayar University, Turkey*")
        
        st.markdown("""
        **Research Focus:** Explainable AI (XAI), Cognitive Science, and Game Theory.
        
        I am an aspiring researcher dedicated to solving the "Black Box" problem in high-stakes decision systems. My work focuses on bridging the gap between computational accuracy and human interpretability, specifically through the lens of **Kansei Engineering** and **Human-in-the-Loop** systems.
        
        This project serves as my undergraduate project and a preliminary proposal for my Master's research. I am actively seeking opportunities to continue this line of inquiry in a laboratory environment that values both algorithmic rigor and cognitive alignment.
        """)

        # Add a "Skills" section to show technical competence
        st.markdown("**Technical Stack:**")
        st.code("Python, PyTorch, Streamlit, Pandas, LaTeX, OpenCV, Git")

        st.subheader("Connect")
        st.markdown("""
        * [**GitHub Portfolio**](https://github.com/4idoneus)
        * [**LinkedIn Profile**](https://www.linkedin.com/in/ipeknazsipahi/)
        * [**ResearchGate**](https://www.researchgate.net/profile/Ipek-Naz-Sipahi?ev=hdr_xprf)
        * [**Personal Website**](https://4idoneus.github.io)
        """)
        
        st.link_button("Contact via Email", "mailto:220315037@ogr.cbu.edu.tr")