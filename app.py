import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# -----------------------------
# CSS (Cards + Tabs)
# -----------------------------
st.markdown("""
<style>

.metric-card {
    background-color:#f8f9fa;
    padding:15px;
    border-radius:10px;
    margin-bottom:10px;
    border:1px solid #e5e7eb;
}

.metric-title {font-size:12px;color:#6b7280;}
.metric-value {font-size:24px;font-weight:bold;color:#111827;}

.green {color:#16a34a;}
.red {color:#dc2626;}
.blue {color:#2563eb;}

/* TAB CONTAINER */
.stTabs [data-baseweb="tab-list"]{
    gap:6px;
}

/* NORMAL TAB */
.stTabs [data-baseweb="tab"]{
    background:#f5f5f5;
    border-radius:8px;
    padding:6px 12px;
    font-size:14px;
    font-weight:500;
    color:#333;
    border:1px solid #ddd;
}

/* HOVER */
.stTabs [data-baseweb="tab"]:hover{
    background:#e9ecef;
}

/* ACTIVE TAB */
.stTabs [aria-selected="true"]{
    background:#1f2937;
    color:white;
    border:1px solid #1f2937;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATA PATH
# -----------------------------

DATA_PATH = "data"

def load(file):
    return pd.read_csv(os.path.join(DATA_PATH,file))

# Load files
markowitz_eval = load("markowitz_evaluation (1).csv")
markowitz_weights = load("markowitz_optimal_weights (1).csv")

lr_eval = load("lr_evaluation.csv")
lr_weights = load("lr_daily_weights.csv")

dsr_eval = load("dsr_evaluation.csv")
dsr_weights = load("dsr_daily_weights.csv")

mdd_eval = load("mdd_evaluation.csv")
mdd_weights = load("mdd_daily_weights.csv")

cnn_eval = load("cnn_fusion_evaluation.csv")
cnn_weights = load("cnn_daily_weights.csv")

# -----------------------------
# METRIC CARD
# -----------------------------

def card(title,value,color=""):
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">{title}</div>
    <div class="metric-value {color}">{value}</div>
    </div>
    """,unsafe_allow_html=True)

# -----------------------------
# SHOW METRICS
# -----------------------------

def show_metrics(df):

    m=dict(zip(df["Metric"],df["Value"]))

    initial=1000000
    final=m.get("Final Portfolio Value",0)

    gain=final-initial
    ret=m.get("Cumulative Return",0)*100
    sharpe=m.get("Sharpe Ratio",0)
    mdd=m.get("Max Drawdown",0)*100

    c1,c2,c3=st.columns(3)
    with c1: card("Initial Value",f"Rs{initial:,.0f}")
    with c2: card("Final Value",f"Rs{final:,.0f}","green")
    with c3: card("Gain/Loss",f"Rs{gain:,.0f} ({ret:.2f}%)","green")

    c4,c5=st.columns(2)
    with c4: card("Sharpe Ratio",f"{sharpe:.4f}")
    with c5: card("Return",f"{ret:.2f}%","green")

    c6,c7=st.columns(2)
    
    with c6: card("Max Drawdown",f"-{mdd:.2f}%","red")

   

# -----------------------------
# WEIGHTS CHART
# -----------------------------

def plot_weights(df):

    if "Weight (%)" in df.columns:

        df=df[df["Weight (%)"]>0]
        fig=px.bar(df,x="Stock",y="Weight (%)",color="Stock")

    else:

        latest=df.iloc[-1].drop(labels=["Date"],errors="ignore")

        df=pd.DataFrame({"Stock":latest.index,"Weight":latest.values})
        df["Weight"]=pd.to_numeric(df["Weight"],errors="coerce")
        df=df.dropna()
        df=df[df["Weight"]>0].sort_values("Weight",ascending=False)

        fig=px.bar(df,x="Stock",y="Weight",color="Stock")

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# SENTIMENT PANEL
# -----------------------------

# -----------------------------
# SENTIMENT PANEL
# -----------------------------

def sentiment_panel():

    st.subheader("Sentiment Analysis")

    try:

        sentiment_df = load("sentiment_results.csv")

        # Select stock
        company = st.selectbox(
            "Select Stock",
            sentiment_df["Company"].unique()
        )

        data = sentiment_df[sentiment_df["Company"] == company].iloc[0]

        label = data["Sentiment_Label"]
        score = data["Sentiment_Score"]
        confidence = data["Confidence"]
        articles = data["Article_Count"]
        explanation = data["Explanation"]

        # Sentiment color
        if label == "POSITIVE":
            color = "green"
        elif label == "NEGATIVE":
            color = "red"
        else:
            color = "blue"

        # Display cards
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Company</div>
        <div class="metric-value">{company}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Sentiment</div>
        <div class="metric-value {color}">{label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Confidence</div>
        <div class="metric-value">{confidence}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Articles</div>
        <div class="metric-value">{articles}</div>
        </div>
        """, unsafe_allow_html=True)

        # Explanation
        st.subheader("Explanation")
        st.info(explanation)

    except:
        st.warning("Sentiment results file not found.")
# -----------------------------
# TITLE
# -----------------------------

st.title("Portfolio Optimizer")
st.write("Choose Your Plan")
st.divider()

# -----------------------------
# LAYOUT (Main + Right Panel)
# -----------------------------

main, right = st.columns([3,1])

with main:

    tab1,tab2,tab3=st.tabs(["Markowitz","Reinforcement Learning","CNN"])

    with tab1:
        st.subheader("Markowitz Algorithm")
        show_metrics(markowitz_eval)
        plot_weights(markowitz_weights)

    with tab2:

        r1,r2,r3=st.tabs(["MDD","DSR","LR"])

        with r1:
            show_metrics(mdd_eval)
            plot_weights(mdd_weights)

        with r2:
            show_metrics(dsr_eval)
            plot_weights(dsr_weights)

        with r3:
            show_metrics(lr_eval)
            plot_weights(lr_weights)

    with tab3:
        show_metrics(cnn_eval)
        plot_weights(cnn_weights)

# -----------------------------
# RIGHT SIDE SENTIMENT
# -----------------------------

with right:
    sentiment_panel()