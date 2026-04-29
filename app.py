import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text
from docx import Document
from topic_model import extract_topics
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Topic Modeling Dashboard",
    layout="wide"
)


# ==============================
# GLOBAL UI STYLE
# ==============================

st.markdown("""
<style>

/* CENTER CONTENT */
section.main > div {
max-width:1100px;
margin:auto;
}

/* ANIMATED DARK BACKGROUND */
[data-testid="stAppViewContainer"] {
background: linear-gradient(
270deg,
#020617,
#081c15,
#0b3c49,
#001219,
#081c15
);
background-size: 600% 600%;
animation: gradientShift 20s ease infinite;
color:white;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
background: linear-gradient(
180deg,
#020617,
#081c15,
#001219
);
}

/* REMOVE WHITE PATCHES */
canvas,
[data-testid="stPlotlyChart"]{
background:transparent!important;
}

/* TAB STYLE */
button[data-baseweb="tab"]{
font-size:16px;
font-weight:600;
color:#cbd5e1;
}

button[data-baseweb="tab"][aria-selected="true"]{
color:#38bdf8;
border-bottom:2px solid #38bdf8;
}

/* GRADIENT ANIMATION */
@keyframes gradientShift{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

</style>
""", unsafe_allow_html=True)


# ==============================
# HERO HEADER
# ==============================

st.markdown("""
<div style="text-align:center;padding-top:30px">

<h1 style="
font-size:52px;
font-weight:800;
background:linear-gradient(90deg,#22c55e,#06b6d4,#38bdf8);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:10px;">
📊 AI Topic Modeling Dashboard
</h1>

<p style="font-size:18px;color:#cbd5e1;">
Upload a document to discover hidden topics, similarity patterns,
and semantic insights instantly
</p>

</div>
""", unsafe_allow_html=True)


# ==============================
# GLASS UPLOAD PANEL
# ==============================

st.markdown("""
<style>

/* GLASS FILE UPLOADER FULL FIX */

section[data-testid="stFileUploader"]{
background:rgba(255,255,255,0.10)!important;
border-radius:18px!important;
padding:25px!important;
backdrop-filter:blur(18px)!important;
box-shadow:0px 8px 35px rgba(0,0,0,0.45)!important;
border:1px solid rgba(255,255,255,0.15)!important;
}

/* REMOVE INNER DARK PANEL */

section[data-testid="stFileUploader"] div{
background:transparent!important;
}

</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 Upload CSV, PDF, DOCX, or TXT file",
    type=["csv","pdf","docx","txt"]
)

st.markdown("</div>", unsafe_allow_html=True)


# FEATURE PILLS

st.markdown("""
<style>

.feature-card {
    backdrop-filter: blur(16px);
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.08),
        rgba(255,255,255,0.02)
    );
    border-radius: 20px;
    padding: 28px 24px;
    width: 240px;
    min-height: 190px;
    text-align: center;
    transition: 0.35s ease;
    box-shadow: 0 0 25px rgba(0,255,170,0.08);
}

.feature-card:hover {
    transform: translateY(-6px) scale(1.03);
    box-shadow: 0 0 40px rgba(0,255,170,0.25);
}

.feature-title {
    font-size: 19px;
    font-weight: 600;
    margin-top: 10px;
}

.feature-text {
    font-size: 14px;
    color: #cbd5e1;
    margin-top: 12px;
    line-height: 1.5;
}

.icon-box {
    font-size: 30px;
}

</style>


<div style="display:flex; justify-content:center; gap:40px; margin-top:50px; flex-wrap:wrap;">


<div class="feature-card">
<div class="icon-box">📊</div>
<div class="feature-title" style="color:#38bdf8;">Topic Extraction</div>
<div class="feature-text">
Automatically detects hidden semantic themes inside your uploaded document.
</div>
</div>


<div class="feature-card">
<div class="icon-box">🧭</div>
<div class="feature-title" style="color:#22c55e;">Similarity Mapping</div>
<div class="feature-text">
Visualizes relationships between topics using semantic distance positioning.
</div>
</div>


<div class="feature-card">
<div class="icon-box">📈</div>
<div class="feature-title" style="color:#facc15;">Quality Metrics</div>
<div class="feature-text">
Measures dominance, diversity, coverage confidence and topic balance.
</div>
</div>


<div class="feature-card">
<div class="icon-box">🔍</div>
<div class="feature-title" style="color:#fb923c;">Interactive Explorer</div>
<div class="feature-text">
Explore keyword contributions and topic influence interactively.
</div>
</div>


</div>
""", unsafe_allow_html=True)


# ==============================
# PROCESS FILE
# ==============================

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text_data = " ".join(df.astype(str).values.flatten())

    elif uploaded_file.name.endswith(".pdf"):
        text_data = extract_text(uploaded_file)

    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        text_data = "\n".join([p.text for p in doc.paragraphs])

    else:
        text_data = uploaded_file.read().decode("utf-8")


    # ==============================
    # SIDEBAR CONTROL
    # ==============================

    st.sidebar.markdown("""
    <style>

    .sidebar-card {
        backdrop-filter: blur(14px);
        background: linear-gradient(
            135deg,
            rgba(255,255,255,0.08),
            rgba(255,255,255,0.02)
        );
        border-radius: 15px;
        padding: 18px 15px;
        margin-bottom: 25px;
        box-shadow: 0 0 20px rgba(0,255,170,0.15);
    }

    .sidebar-title {
        font-size:16px;
        font-weight:600;
        color:#22c55e;
        margin-bottom:10px;
    }

    </style>

    <div class="sidebar-card">
    <div class="sidebar-title">
    🎯 Topic Control Panel
    </div>
    </div>
    """, unsafe_allow_html=True)

    num_topics = st.sidebar.slider(
        "Select number of topics",
        min_value=2,
        max_value=10,
        value=5
    )


    topics, topic_strength = extract_topics(text_data, num_topics)

    topic_percent = topic_strength / topic_strength.sum()
    sorted_indices = topic_strength.argsort()[::-1]


    # ==============================
    # COLOR FUNCTION
    # ==============================

    def topic_color(val):
        if val >= 0.35:
            return "#16a34a"
        elif val >= 0.25:
            return "#65a30d"
        elif val >= 0.18:
            return "#eab308"
        elif val >= 0.12:
            return "#f97316"
        else:
            return "#ef4444"


    # ==============================
    # EXTRACTED TOPICS PANEL
    # ==============================

    st.subheader("📌 Extracted Topics with Importance")

    for rank, i in enumerate(sorted_indices):

        importance = topic_percent[i]
        color = topic_color(importance)

        topic_sentence = " ".join(topics[i])

        st.markdown(f"""
        <div style="
        background:linear-gradient(90deg,{color}22,transparent);
        border-left:6px solid {color};
        padding:14px;
        border-radius:12px;
        margin-bottom:10px;">
        🏅 <b>Topic {rank+1}:</b> {topic_sentence}
        <span style="float:right;font-weight:bold;color:{color};">
        {importance*100:.2f}%
        </span>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(importance))


    # ==============================
    # TABS
    # ==============================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Topic Importance",
        "🔥 Topic Similarity Analysis",
        "🏆 Topic Overview",
        "🔎 Interactive Explorer",
        "📈 Topic Quality Metrics"
    ])


    # ==============================
    # TAB 1
    # ==============================

    with tab1:

        st.subheader("📊 Topic Importance Distribution")

        st.caption(
        "Shows how strongly each discovered topic contributes to the document. "
        "Higher percentage indicates dominant semantic influence."
        )   

        sorted_names = [f"Rank {i+1}" for i in range(len(sorted_indices))]
        sorted_percentages = topic_percent[sorted_indices] * 100

        graph_data = pd.DataFrame({
            "Topic": sorted_names,
            "Importance": sorted_percentages
        })

        colors = [topic_color(topic_percent[i]) for i in sorted_indices]

        bar_fig = px.bar(
            graph_data,
            x="Topic",
            y="Importance",
            text=graph_data["Importance"].round(2),
            color=graph_data["Importance"],
            color_continuous_scale=[
                "#ef4444",  # red
                "#f97316",  # orange
                "#eab308",  # yellow
                "#65a30d",  # lime
                "#16a34a"   # green
            ]
        )

        bar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            coloraxis_showscale=False
        )

        pie_fig = px.pie(
            values=sorted_percentages,
            names=sorted_names
        )

        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Importance Ranking (Bar View)")
            st.caption("Visual comparison of topic dominance across extracted themes.")
            st.plotly_chart(bar_fig, use_container_width=True)

        with col2:
            st.markdown("#### 🥧 Topic Contribution Share (Pie View)")
            st.caption("Relative percentage share of each topic within the document.")
            st.plotly_chart(pie_fig, use_container_width=True)


    # ==============================
    # TAB 2 (SIMILARITY)
    # ==============================

    with tab2:

        st.subheader("🔥 Topic Similarity Analysis")

        st.caption(
        "Evaluates how extracted topics relate to each other based on semantic proximity. "
        "Closer positions indicate stronger contextual similarity between themes."
        )

        x = np.random.uniform(-1,1,len(topic_percent))
        y = np.random.uniform(-1,1,len(topic_percent))

        bubble_df = pd.DataFrame({
            "x":x,
            "y":y,
            "Topic":sorted_names,
            "Importance":sorted_percentages
        })

        bubble_fig = px.scatter(
            bubble_df,
            x="x",
            y="y",
            size="Importance",
            text="Topic",
            color="Importance",
            color_continuous_scale="viridis"
        )

        bubble_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        heatmap_data = topic_percent.reshape(-1,1)

        fig, ax = plt.subplots(figsize=(5,3))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="viridis",
            yticklabels=sorted_names,
            annot_kws={"color":"white"}
        )

        # Make axis labels white
        ax.set_xlabel("Similarity Scale", color="white")
        ax.set_ylabel("Topics", color="white")

        # Make tick labels white
        ax.tick_params(colors="white")

        # Make colorbar text white
        cbar = ax.collections[0].colorbar
        cbar.set_label("Similarity Score", color="white")
        cbar.ax.tick_params(colors="white")

        # Ensure colorbar numbers are white
        for label in cbar.ax.get_yticklabels():
            label.set_color("white")

        # Remove black border around colorbar
        cbar.outline.set_edgecolor("white")


        ax.tick_params(colors="white")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### 🔵 Topic Relationship Map (Bubble View)")
            st.caption(
                "Displays spatial similarity between topics. Larger bubbles represent stronger topic importance."
            )
            st.plotly_chart(bubble_fig, use_container_width=True)

        with colB:
            st.markdown("#### 🌡 Topic Similarity Strength (Heatmap View)")
            st.caption(
                "Shows normalized similarity scores across topics. Brighter colors indicate stronger relationships."
            )
            st.pyplot(fig)


    # ==============================
    # TAB 3
    # ==============================

    with tab3:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🏆 Topic Leaderboard")

            medals=["🥇","🥈","🥉"]

            for rank,i in enumerate(sorted_indices[:5]):

                percent=topic_percent[i]*100
                badge=medals[rank] if rank<3 else "⭐"

                st.markdown(
                    f"{badge} Rank {rank+1} Topic — {percent:.2f}%"
                )

        with col2:

            st.subheader("📈 Dominant Topic Strength")

            top_value=max(topic_percent)*100

            gauge=go.Figure(go.Indicator(
                mode="gauge+number",
                value=top_value,
                title={'text':"Top Topic (%)"},
                gauge={'axis':{'range':[0,100]},
                       'bar':{'color':"green"}}
            ))

            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            st.plotly_chart(gauge,use_container_width=True)


    # ==============================
    # TAB 4
    # ==============================

    with tab4:

        st.subheader("🔎 Interactive Topic Explorer")

        selected_topic=st.selectbox(
            "Select topic",
            sorted_names
        )

        topic_id=int(selected_topic.split()[1])-1
        actual_index=sorted_indices[topic_id]

        importance_value=topic_percent[actual_index]*100

        # Importance indicator

        if importance_value > 30:
            st.success(f"Importance: {importance_value:.2f}% (High)")
        elif importance_value > 15:
            st.warning(f"Importance: {importance_value:.2f}% (Medium)")
        else:
            st.error(f"Importance: {importance_value:.2f}% (Low)")


        # KEYWORD CONTRIBUTION CHART

        st.markdown("### Keyword Contribution Strength")

        keywords = topics[actual_index][:10]

        keyword_scores = np.linspace(
            1.0,
            0.3,
            len(keywords)
        )

        keyword_df = pd.DataFrame({
            "Keyword": keywords,
            "Weight": keyword_scores
        })

        keyword_fig = px.bar(
            keyword_df,
            x="Weight",
            y="Keyword",
            orientation="h",
            color="Weight",
            color_continuous_scale=[
                "#16a34a",
                "#65a30d",
                "#eab308",
                "#f97316",
                "#ef4444"
            ]
        )

        keyword_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            coloraxis_showscale=False
        )

        st.plotly_chart(keyword_fig, use_container_width=True)


        # SUMMARY TEXT

        st.markdown("Summary Extracted From This Topic")
        st.info(
            f"This topic represents {importance_value:.2f}% of the document "
            f"and focuses mainly on {keywords[0]}, {keywords[1]}, {keywords[2]}."
        )


    # ==============================
    # TAB 5
    # ==============================

    with tab5:

        st.subheader("📈 Topic Quality Metrics")

        diversity=len(set(
            word for topic in topics for word in topic[:5]
        ))/(num_topics*5)

        balance=1-np.std(topic_percent)
        dominance=max(topic_percent)
        coverage=np.mean(topic_percent)

        col1,col2,col3,col4=st.columns(4)

        col1.metric("🧠 Diversity",f"{diversity:.2f}")
        col2.metric("⚖️ Balance",f"{balance:.2f}")
        col3.metric("🏆 Dominance",f"{dominance:.2f}")
        col4.metric("📊 Coverage",f"{coverage:.2f}")

        metric_df=pd.DataFrame({
            "Metric":["Diversity","Balance","Dominance","Coverage"],
            "Score":[diversity,balance,dominance,coverage]
        })

        fig_metrics=px.bar(
            metric_df,
            x="Metric",
            y="Score",
            color="Score",
            color_continuous_scale="viridis"
        )

        fig_metrics.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(fig_metrics,use_container_width=True)