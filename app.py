"""Olympo Prediction - Olympic medal prediction dashboard (Streamlit)."""
import datetime as dt
import io
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Olympo Prediction", page_icon="🏅", layout="wide")

DATA_FILE = "olympics_medal_table_2024.csv"
MODEL_FILE = "olympic_best_model.pkl"
BASE_YEAR = 2024
FUTURE_YEARS = [2028, 2032, 2036, 2040, 2044, 2048]
MEDAL_COLORS = {
    "Gold": "#6233EF",
    "Silver": "#2558C6",
    "Bronze": "#A31779",
    "Total": "#55D50B"
}

# ---------- Styling ----------
st.markdown(
    """
    <style>
    :root {
        --accent: #8B5CF6;
        --accent-blue: #38BDF8;
        --accent-teal: #2DD4BF;
        --accent-pink: #F472B6;
        --background: #080D19;
        --surface: #111827;
        --border: #26344D;
        --muted: #9CAAC2;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(99, 102, 241, 0.12), transparent 35%),
            radial-gradient(ellipse at 90% 10%, rgba(45, 212, 191, 0.08), transparent 30%),
            linear-gradient(135deg, #080D19 0%, #0B1220 55%, #0D1424 100%);
        color: #E5EDFF;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 25, 0.85);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1600px;
    }

    /* Main title */
    .olympo-title {
        font-size: 2.8rem;
        font-weight: 850;
        letter-spacing: -1.2px;
        background: linear-gradient(100deg, #A78BFA 0%, #60A5FA 48%, #2DD4BF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.25;
    }

    .olympo-sub {
        color: #9CAAC2;
        margin-top: 0.5rem;
        font-size: 1rem;
        letter-spacing: 0.2px;
    }

    .olympo-brand {
        font-size: 1.55rem;
        font-weight: 850;
        letter-spacing: -0.6px;
        line-height: 1.25;
        background: linear-gradient(110deg, #A78BFA, #60A5FA, #2DD4BF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .olympo-brand-sub {
        color: #94A3B8;
        font-size: 0.76rem;
        line-height: 1.7;
        margin-top: 0.5rem;
        letter-spacing: 0.3px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1020 0%, #0C1425 100%);
        border-right: 1px solid #202D45;
        min-width: 280px;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #C5D0E5;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.5rem;
        flex-direction: column;
        align-items: stretch;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: #10192A;
        border: 1px solid #24334B;
        padding: 0.72rem 0.85rem;
        border-radius: 12px;
        width: 100%;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(110deg, #28204B, #172C48);
        border-color: #8B5CF6;
        color: #DDD6FE;
        font-weight: 650;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.13);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        border-color: #38BDF8;
        background-color: #17243A;
    }

    /* Main navigation */
    [data-testid="stMain"] div[role="radiogroup"] {
        gap: 0.55rem;
        flex-wrap: wrap;
    }

    [data-testid="stMain"] div[role="radiogroup"] label {
        background: #111A2B;
        border: 1px solid #2A3952;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        transition: all 0.2s ease;
    }

    [data-testid="stMain"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(110deg, #302354, #183451);
        border-color: #8B5CF6;
        color: #DDD6FE;
        font-weight: 650;
    }

    [data-testid="stMain"] div[role="radiogroup"] label:hover {
        border-color: #2DD4BF;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #151F33 0%, #101827 100%);
        border: 1px solid #2A3952;
        border-radius: 16px;
        padding: 18px 18px 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
        transition: all 0.22s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #6475A3;
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(56, 189, 248, 0.09);
    }

    [data-testid="stMetricLabel"] {
        color: #A5B4CF !important;
    }

    [data-testid="stMetricValue"] {
        color: #F1F5FF !important;
        font-weight: 800;
    }

    [data-testid="stMetricDelta"] {
        color: #2DD4BF !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(110deg, #7C3AED, #4F7DF3);
        color: #FFFFFF;
        border: 1px solid #8B7CF6;
        border-radius: 11px;
        padding: 0.58rem 1.2rem;
        font-weight: 700;
        transition: all 0.2s ease;
        box-shadow: 0 5px 18px rgba(99, 102, 241, 0.13);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(110deg, #8B5CF6, #0EA5E9);
        border-color: #67E8F9;
        color: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: 0 7px 22px rgba(56, 189, 248, 0.2);
    }

    /* Text inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background-color: #111B2D !important;
        color: #E5EDFF !important;
        border: 1px solid #33435F !important;
        border-radius: 10px !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 1px #8B5CF6 !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #71819C !important;
    }

    /* Dropdowns */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: #2558C6 !important;
        border-color: #33435F !important;
        border-radius: 10px !important;
    }

    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: #2558C6 !important;
        border: 1px solid #8B5CF6 !important;
        color: #EDE9FE !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #141F33 !important;
        border: 1px solid #33435F !important;
    }

    [role="option"] {
        background-color: #141F33 !important;
        color: #E5EDFF !important;
    }

    [role="option"]:hover {
        background-color: #253653 !important;
    }

    /* Sliders */
    [data-testid="stSlider"] [role="slider"] {
        background: #A78BFA !important;
        border-color: #A78BFA !important;
    }

    [data-testid="stSlider"] [data-testid="stTickBar"] {
        background: #34415D;
    }

    /* Tabs */
    [data-testid="stTabs"] [role="tab"] {
        color: #A5B4CF;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #A78BFA !important;
        border-bottom-color: #8B5CF6 !important;
    }

    [data-testid="stTabs"] [role="tab"]:hover {
        color: #67E8F9 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: linear-gradient(145deg, #111B2D, #0F1726);
        border: 1px solid #293850;
        border-radius: 13px;
    }

    [data-testid="stExpander"] summary:hover {
        color: #A78BFA;
    }

    /* Data tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #293850;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] th {
        background-color: #1B2940 !important;
        color: #C4B5FD !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #34435D;
        background: #111C2D;
        color: #E5EDFF;
    }

    /* Links */
    a {
        color: #67E8F9 !important;
    }

    a:hover {
        color: #A78BFA !important;
    }

    /* Headings and text */
    h1, h2, h3 {
        color: #E5EDFF;
        letter-spacing: -0.4px;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #C1CCE0;
    }

    /* Dividers */
    hr {
        border-color: #26344D !important;
    }

    /* Radio buttons and checkboxes */
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        color: #D6E0F2;
    }

    [data-testid="stRadio"] input:checked + div,
    [data-testid="stCheckbox"] input:checked + div {
        color: #A78BFA;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #080D19;
    }

    ::-webkit-scrollbar-thumb {
        background: #34435F;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #8B5CF6;
    }

    /* Success and status messages */
    [data-testid="stAlert"][kind="success"] {
        border-color: #2DD4BF;
    }

    /* Responsive title */
    @media (max-width: 768px) {
        .olympo-title {
            font-size: 2.1rem;
        }

        .block-container {
            padding-top: 1rem;
        }
    }
    
/* FORCE PURPLE COUNTRY SELECTION COLORS */
.stMultiSelect [data-baseweb="tag"],
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: #302354 !important;
    background-color: #302354 !important;
    border: 1px solid #A78BFA !important;
    color: #F5F3FF !important;
}

.stMultiSelect [data-baseweb="tag"] *,
[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
    color: #F5F3FF !important;
    fill: #F5F3FF !important;
}

/* FORCE PURPLE RANGE SLIDERS */
.stSlider [data-baseweb="slider"] [role="slider"],
[data-testid="stSlider"] [role="slider"] {
    background: #A78BFA !important;
    background-color: #A78BFA !important;
    border: 2px solid #C4B5FD !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
}

/* Selected track */
.stSlider [data-baseweb="slider"] > div > div,
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background-color: #8B5CF6 !important;
}

/* Slider track background */
.stSlider [data-baseweb="slider"] [data-testid="stTickBar"],
[data-testid="stSlider"] [data-testid="stTickBar"] {
    background: #334155 !important;
}

/* Remove yellow browser accent */
.stApp input,
.stApp input[type="range"] {
    accent-color: #8B5CF6 !important;
}

/* Purple focus states */
.stApp [data-baseweb="select"]:focus-within {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 1px #8B5CF6 !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #24183D !important;
    border: 1px solid #8B5CF6 !important;
}

[role="option"] {
    background-color: #24183D !important;
    color: #FFFFFF !important;
}

[role="option"]:hover {
    background-color: #49316B !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data & model ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["NOC"] = df["NOC"].astype(str).apply(lambda s: re.sub(r"[^\w\s\.\-'()]", "", s).strip())
    df = df.rename(columns={"NOC": "Country"})
    for c in ["Gold", "Silver", "Bronze"]:
        df[c] = df[c].fillna(0).astype(int)
    df["Total"] = df[["Gold", "Silver", "Bronze"]].sum(axis=1).astype(int)
    df["Rank"] = df["Rank"].fillna(0).astype(int)
    return df


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_FILE)
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("model_name", "Model"), bundle.get("features", ["Gold", "Silver", "Bronze"]), bundle.get("target", "Total")
    return bundle, type(bundle).__name__, ["Gold", "Silver", "Bronze"], "Total"


def predict_total(model, features, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(frame[features]), 0, None)


def project_medals(df: pd.DataFrame, year: int, growth: dict) -> pd.DataFrame:
    """Project medals for a future Games using compound growth per Games (every 4 years)."""
    n = (year - BASE_YEAR) / 4
    out = df[["Country"]].copy()
    for m in ["Gold", "Silver", "Bronze"]:
        out[m] = np.round(df[m] * (1 + growth[m] / 100) ** n).astype(int).clip(lower=0)
    out["Year"] = year
    return out


df = load_data()
model, model_name, features, target = load_model()
df["Predicted Total"] = np.round(predict_total(model, features, df), 2)

# ---------- Sidebar: brand, navigation, forecast assumptions ----------
SECTIONS = ["Dataset Overview", "Best Model", "Total Medals Predictor", "Medal Forecast by Year", "Report"]
NAV_ICONS = {
    "Dataset Overview": "📊  Dataset Overview",
    "Best Model": "🧠  Best Model",
    "Total Medals Predictor": "🎯  Total Medals Predictor",
    "Medal Forecast by Year": "🔮  Medal Forecast by Year",
    "Report": "📄  Report",
}
SECTION_BLURB = {
    "Dataset Overview": "Browse, filter and chart the Paris 2024 medal table",
    "Best Model": "How the winning model scores, and where it misses",
    "Total Medals Predictor": "Predict total medals from gold, silver and bronze inputs",
    "Medal Forecast by Year": "Which countries can win gold, silver or bronze in future Games",
    "Report": "Generate and download a medal forecast report",
}

with st.sidebar:
    st.markdown('<p class="olympo-brand">🏅 Olympo<br>Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="olympo-brand-sub">Olympic medal analytics &amp; forecasts<br>'
                'built on the Paris 2024 medal table</p>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", SECTIONS, format_func=lambda s: NAV_ICONS[s], label_visibility="collapsed")
    st.divider()
    st.header("⚙️ Forecast assumptions")
    st.caption("Change in medals per Olympic Games (every 4 years).")
    growth = {
        "Gold": st.slider("Gold growth % per Games", -30, 30, 3),
        "Silver": st.slider("Silver growth % per Games", -30, 30, 2),
        "Bronze": st.slider("Bronze growth % per Games", -30, 30, 2),
    }

# ---------- Main header ----------
st.markdown('<p class="olympo-title">🏅 Olympo Prediction</p>', unsafe_allow_html=True)
st.markdown(f'<p class="olympo-sub">{SECTION_BLURB[page]}</p>', unsafe_allow_html=True)
st.divider()


def country_filter(key: str, default_all=True):
    countries = sorted(df["Country"].unique())
    return st.multiselect("Countries", countries, default=countries if default_all else [], key=key)


# ---------- 1. Dataset Overview ----------
if page == "Dataset Overview":
    st.header("📊 Dataset Overview")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel = country_filter("ov_c")
    with c2:
        min_total = st.slider("Minimum total medals", 0, int(df["Total"].max()), 0)
    with c3:
        sort_by = st.selectbox("Sort by", ["Rank", "Gold", "Silver", "Bronze", "Total"])

    rank_range = st.slider("Rank range", int(df["Rank"].min()), int(df["Rank"].max()), (int(df["Rank"].min()), int(df["Rank"].max())))
    f = df[df["Country"].isin(sel) & (df["Total"] >= min_total) & df["Rank"].between(*rank_range)]
    f = f.sort_values(sort_by, ascending=(sort_by == "Rank"))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Countries", len(f))
    m2.metric("🥇 Gold", int(f["Gold"].sum()))
    m3.metric("🥈 Silver", int(f["Silver"].sum()))
    m4.metric("🥉 Bronze", int(f["Bronze"].sum()))
    m5.metric("Total", int(f["Total"].sum()))

    st.dataframe(f.drop(columns=["Predicted Total"]), width="stretch", hide_index=True)

    if len(f) > 5:
     top_n = st.slider(
        "Countries shown in chart",
        min_value=5,
        max_value=len(f),
        value=min(15, len(f)))
    else:
     top_n = len(f)
    st.info(f"Showing all {top_n} available countries.")
    top = f.sort_values("Total", ascending=False).head(top_n)
    fig = px.bar(top, x="Country", y=["Gold", "Silver", "Bronze"], barmode="stack",
                 color_discrete_map=MEDAL_COLORS, template="plotly_dark", title="Medal distribution")
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    a.plotly_chart(px.pie(values=[f.Gold.sum(), f.Silver.sum(), f.Bronze.sum()], names=["Gold", "Silver", "Bronze"],
                          color=["Gold", "Silver", "Bronze"], color_discrete_map=MEDAL_COLORS, hole=.5,
                          template="plotly_dark", title="Share of medal types"), width="stretch")
    b.plotly_chart(px.histogram(f, x="Total", nbins=20, template="plotly_dark", title="Total medals distribution",
                                color_discrete_sequence=[MEDAL_COLORS["Total"]]), width="stretch")

    with st.expander("Summary statistics"):
        st.dataframe(f[["Gold", "Silver", "Bronze", "Total"]].describe().round(2), width="stretch")

# ---------- 2. Best Model ----------
elif page == "Best Model":
    st.header("🧠 Best Model")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    c1, c2 = st.columns([2, 1])
    with c1:
        sel = country_filter("bm_c")
    with c2:
        metric_view = st.selectbox("Residual view", ["Absolute error", "Signed error"])
    f = df[df["Country"].isin(sel)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Model", model_name)
    k2.metric("Target", target)
    k3.metric("Features", ", ".join(features))
    k4.metric("Intercept", f"{float(getattr(model, 'intercept_', 0)):.4f}")

    if len(f) > 1:
        y, yp = f["Total"], f["Predicted Total"]
        s1, s2, s3 = st.columns(3)
        s1.metric("R² score", f"{r2_score(y, yp):.4f}")
        s2.metric("MAE", f"{mean_absolute_error(y, yp):.4f}")
        s3.metric("RMSE", f"{np.sqrt(mean_squared_error(y, yp)):.4f}")

    coef = pd.DataFrame({"Feature": features, "Coefficient": np.ravel(getattr(model, "coef_", np.zeros(len(features))))})
    a, b = st.columns(2)
    a.plotly_chart(px.bar(coef, x="Feature", y="Coefficient", color="Feature", color_discrete_map=MEDAL_COLORS,
                          template="plotly_dark", title="Model coefficients"), width="stretch")
    fig = px.scatter(f, x="Total", y="Predicted Total", hover_name="Country", template="plotly_dark",
                     title="Actual vs predicted total", color_discrete_sequence=[MEDAL_COLORS["Gold"]])
    mx = float(max(f["Total"].max() if len(f) else 1, 1))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect fit", line=dict(dash="dash", color="#4FD1C5")))
    b.plotly_chart(fig, width="stretch")

    res = f.assign(Error=(f["Predicted Total"] - f["Total"]))
    if metric_view == "Absolute error":
        res["Error"] = res["Error"].abs()
    st.dataframe(res[["Country", "Gold", "Silver", "Bronze", "Total", "Predicted Total", "Error"]], width="stretch", hide_index=True)

# ---------- 3. Total Medals Predictor ----------
elif page == "Total Medals Predictor":
    st.header("🎯 Total Medals Predictor")
    mode = st.radio("Prediction mode", ["Custom input", "Pick a country"], horizontal=True)

    if mode == "Pick a country":
        country = st.selectbox("Country", df["Country"])
        row = df[df["Country"] == country].iloc[0]
        g0, s0, b0 = int(row.Gold), int(row.Silver), int(row.Bronze)
    else:
        g0, s0, b0 = 10, 10, 10

    c1, c2, c3 = st.columns(3)
    g = c1.number_input("🥇 Gold", 0, 200, g0)
    s = c2.number_input("🥈 Silver", 0, 200, s0)
    b = c3.number_input("🥉 Bronze", 0, 200, b0)
    pred = float(predict_total(model, features, pd.DataFrame([[g, s, b]], columns=features))[0])
    st.success(f"### Predicted total medals: **{pred:.0f}**")

    st.subheader("Predicted totals across countries")
    f1, f2 = st.columns([2, 1])
    with f1:
        sel = country_filter("tp_c")
    with f2:
        year = st.selectbox("Games", [BASE_YEAR] + FUTURE_YEARS)
    proj = df[["Country", "Gold", "Silver", "Bronze"]] if year == BASE_YEAR else project_medals(df, year, growth)
    proj = proj[proj["Country"].isin(sel)].copy()
    proj["Predicted Total"] = np.round(predict_total(model, features, proj)).astype(int)
    if len(proj) > 5:
     top_n = st.slider(
        "Top N countries",
        min_value=5,
        max_value=len(proj),
        value=min(20, len(proj)))
    else:
     top_n = len(proj)
    st.info(f"Showing all {top_n} available countries.")
    top = proj.sort_values("Predicted Total", ascending=False).head(top_n)

    st.plotly_chart(px.bar(top, x="Predicted Total", y="Country", orientation="h", template="plotly_dark",
                           color="Predicted Total", color_continuous_scale="YlOrBr",
                           title=f"Predicted total medals – {year}").update_yaxes(autorange="reversed"),
                    width="stretch")
    a, b2 = st.columns(2)
    trend = pd.concat([project_medals(df, y, growth) for y in FUTURE_YEARS] +
                      [df[["Country", "Gold", "Silver", "Bronze"]].assign(Year=BASE_YEAR)])
    trend = trend[trend["Country"].isin(top["Country"].head(6))]
    trend["Predicted Total"] = predict_total(model, features, trend)
    a.plotly_chart(px.line(trend.sort_values("Year"), x="Year", y="Predicted Total", color="Country", markers=True,
                           template="plotly_dark", title="Trend for top 6 countries"), width="stretch")
    b2.plotly_chart(px.treemap(top, path=["Country"], values="Predicted Total", template="plotly_dark",
                               color="Predicted Total", color_continuous_scale="YlOrBr", title="Share of predicted medals"),
                    width="stretch")
    st.dataframe(top, width="stretch", hide_index=True)

# ---------- 4. Medal Forecast by Year ----------
elif page == "Medal Forecast by Year":
    st.header("🔮 Medal Forecast by Year")
    year = st.select_slider("📅 Year (main filter)", options=FUTURE_YEARS, value=FUTURE_YEARS[0])
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel = country_filter("fy_c")
    with c2:
        medal = st.selectbox("Medal type", ["Gold", "Silver", "Bronze"])
    with c3:
        min_count = st.number_input("Minimum medals of that type", 1, 50, 1)

    proj = project_medals(df, year, growth)
    proj = proj[proj["Country"].isin(sel)]
    proj["Total"] = np.round(predict_total(model, features, proj)).astype(int)

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Countries winning 🥇 Gold in {year}", int((proj["Gold"] >= 1).sum()))
    k2.metric(f"Countries winning 🥈 Silver in {year}", int((proj["Silver"] >= 1).sum()))
    k3.metric(f"Countries winning 🥉 Bronze in {year}", int((proj["Bronze"] >= 1).sum()))

    counts = pd.DataFrame([{"Year": y, "Medal": m, "Countries": int((p[m] >= 1).sum())}
                           for y in FUTURE_YEARS for p in [project_medals(df[df["Country"].isin(sel)], y, growth)]
                           for m in ["Gold", "Silver", "Bronze"]])
    a, b = st.columns(2)
    a.plotly_chart(px.line(counts, x="Year", y="Countries", color="Medal", markers=True, color_discrete_map=MEDAL_COLORS,
                           template="plotly_dark", title="Number of medal-winning countries over time"),
                   width="stretch")
    winners = proj[proj[medal] >= min_count].sort_values(medal, ascending=False)
    b.plotly_chart(px.bar(winners.head(20), x="Country", y=medal, template="plotly_dark",
                          color_discrete_sequence=[MEDAL_COLORS[medal]], title=f"Top {medal} winners – {year}"),
                   width="stretch")

    st.subheader(f"{len(winners)} countries forecast to win ≥ {min_count} {medal} in {year}")
    st.dataframe(winners, width="stretch", hide_index=True)

# ---------- 5. Report ----------
elif page == "Report":
    st.header("📄 Report")
    c1, c2, c3 = st.columns(3)
    year = c1.selectbox("Forecast year", FUTURE_YEARS)
    top_n = c2.slider("Countries in report", 5, len(df), 15)
    fmt = c3.selectbox("Download format", ["HTML", "CSV", "Excel"])

    proj = project_medals(df, year, growth)
    proj["Predicted Total"] = np.round(predict_total(model, features, proj)).astype(int)
    proj = proj.sort_values("Predicted Total", ascending=False).head(top_n)
    base = df.sort_values("Total", ascending=False).head(top_n)[["Country", "Gold", "Silver", "Bronze", "Total"]]

    if st.button("⚡ Generate report", type="primary"):
        st.session_state["report"] = dict(year=year, proj=proj, base=base, fmt=fmt,
                                          made=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))

    rep = st.session_state.get("report")
    if rep:
        st.subheader(f"Olympo Prediction Report – {rep['year']}")
        st.caption(f"Generated {rep['made']} • Model: {model_name} • Growth/Games: "
                   f"Gold {growth['Gold']}%, Silver {growth['Silver']}%, Bronze {growth['Bronze']}%")
        m1, m2, m3 = st.columns(3)
        m1.metric("Leader", rep["proj"].iloc[0]["Country"])
        m2.metric("Leader predicted total", int(rep["proj"].iloc[0]["Predicted Total"]))
        m3.metric("Countries with Gold", int((rep["proj"]["Gold"] >= 1).sum()))
        st.markdown("**2024 actual (top countries)**")
        st.dataframe(rep["base"], width="stretch", hide_index=True)
        st.markdown(f"**{rep['year']} forecast**")
        st.dataframe(rep["proj"], width="stretch", hide_index=True)
        chart = px.bar(rep["proj"], x="Country", y=["Gold", "Silver", "Bronze"], color_discrete_map=MEDAL_COLORS,
                       template="plotly_dark", title=f"Forecast medals – {rep['year']}")
        st.plotly_chart(chart, width="stretch")

        name = f"olympo_report_{rep['year']}"
        if fmt == "CSV":
            data, mime, fname = rep["proj"].to_csv(index=False).encode(), "text/csv", f"{name}.csv"
        elif fmt == "Excel":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                rep["base"].to_excel(w, sheet_name="2024 Actual", index=False)
                rep["proj"].to_excel(w, sheet_name=f"{rep['year']} Forecast", index=False)
            data, mime, fname = buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", f"{name}.xlsx"
        else:
            html = f"""<html><head><meta charset='utf-8'><title>Olympo Prediction Report</title>
            <style>body{{background:#0E1117;color:#E6EDF3;font-family:Arial;padding:24px}}
            table{{border-collapse:collapse}}td,th{{border:1px solid #30363D;padding:6px 10px}}
            th{{background:#161B22;color:#F5C542}}h1{{color:#F5C542}}</style></head><body>
            <h1>Olympo Prediction Report – {rep['year']}</h1>
            <p>Generated {rep['made']} · Model: {model_name} · Features: {', '.join(features)}</p>
            <p>Growth per Games: Gold {growth['Gold']}%, Silver {growth['Silver']}%, Bronze {growth['Bronze']}%</p>
            <h2>2024 actual</h2>{rep['base'].to_html(index=False)}
            <h2>{rep['year']} forecast</h2>{rep['proj'].to_html(index=False)}
            {chart.to_html(full_html=False, include_plotlyjs='cdn')}</body></html>"""
            data, mime, fname = html.encode(), "text/html", f"{name}.html"
        st.download_button("⬇️ Download report", data, file_name=fname, mime=mime, type="primary")
    else:
        st.info("Choose options and click **Generate report**.")

st.sidebar.divider()
st.sidebar.caption("Olympo Prediction • Data: Paris 2024 medal table")
