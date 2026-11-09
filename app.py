"""
Olympo Prediction - Olympic medal prediction dashboard (Streamlit)
Run:  streamlit run app.py
Put Medals.csv and medal_probability_model_1.pkl next to this file
(or upload them from the sidebar).
"""
import io
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ theme
BLUE, YELLOW, BLACK, GREEN, RED = "#0085C7", "#F4C300", "#E8E8E8", "#009F3D", "#DF0024"
RINGS = [BLUE, YELLOW, BLACK, GREEN, RED]
MEDAL_COLORS = {"Gold": "#FFC300", "Silver": "#C0C0C0", "Bronze": "#CD7F32"}
FEATURES = ["Age", "Gold", "Silver", "Bronze", "Total", "Season",
            "Prev_Appearances", "Prev_Medals", "Gold_Share"]

st.set_page_config(page_title="Olympo Prediction", page_icon="🏅", layout="wide")

st.markdown(f"""
<style>
.stApp {{ background: radial-gradient(circle at 20% 0%, #13203a 0%, #0a0e17 55%); color:#e8e8e8; }}
section[data-testid="stSidebar"] {{ background:#070a11; border-right:1px solid #1d2638; }}
h1,h2,h3 {{ font-family: 'Georgia', serif; letter-spacing:.5px; }}
.rings-bar {{ height:5px; border-radius:4px;
  background: linear-gradient(90deg,{BLUE} 0 20%,{YELLOW} 20% 40%,{BLACK} 40% 60%,{GREEN} 60% 80%,{RED} 80% 100%); margin:4px 0 18px; }}
.kpi {{ background:#111827; border:1px solid #1f2a3d; border-radius:14px; padding:16px 18px; }}
.kpi .v {{ font-size:28px; font-weight:700; }}
.kpi .l {{ font-size:12px; text-transform:uppercase; color:#8b97ad; letter-spacing:1px; }}
div.stDownloadButton > button {{ background:{BLUE}; color:white; border:0; border-radius:10px; font-weight:600; }}
</style>""", unsafe_allow_html=True)

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            colorway=RINGS)


def header(title, sub=""):
    st.markdown(f"## {title}")
    st.markdown('<div class="rings-bar"></div>', unsafe_allow_html=True)
    if sub:
        st.caption(sub)


def kpis(items):
    cols = st.columns(len(items))
    for c, (label, value), color in zip(cols, items, RINGS):
        c.markdown(f'<div class="kpi" style="border-top:3px solid {color}">'
                   f'<div class="l">{label}</div><div class="v">{value}</div></div>',
                   unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load_data(raw_bytes):
    df_raw = pd.read_csv(io.BytesIO(raw_bytes), encoding="cp1252")
    df = df_raw.copy()
    df.columns = ["Athlete", "Age", "Year", "Closing_Date", "Gold", "Silver", "Bronze", "Total"]
    df = df.dropna(subset=["Athlete", "Age"]).drop_duplicates()
    df = df.groupby(["Athlete", "Year"], as_index=False).agg(
        Age=("Age", "max"), Gold=("Gold", "sum"), Silver=("Silver", "sum"),
        Bronze=("Bronze", "sum"), Total=("Total", "sum"))
    df["Season"] = np.where(df["Year"] % 4 == 0, 1, 0)
    df = df.sort_values(["Athlete", "Season", "Year"])
    df["Prev_Appearances"] = df.groupby(["Athlete", "Season"]).cumcount()
    df["Prev_Medals"] = df.groupby(["Athlete", "Season"])["Total"].cumsum() - df["Total"]
    df["Gold_Share"] = (df["Gold"] / df["Total"]).fillna(0)
    nxt = df.set_index(["Athlete", "Year"])[["Total", "Gold", "Silver", "Bronze"]]
    key = list(zip(df["Athlete"], df["Year"] + 4))
    df["Label"] = [int(k in nxt.index) for k in key]
    for col in ["Total", "Gold", "Silver", "Bronze"]:
        df["Next_" + col] = [nxt.loc[k, col] if k in nxt.index else 0 for k in key]
    df["SeasonName"] = np.where(df["Season"] == 1, "Summer", "Winter")
    return df_raw, df.reset_index(drop=True)


@st.cache_resource
def load_bundle(pkl_bytes):
    return pickle.loads(pkl_bytes)


def bundle_prob(bundle, X):
    m = bundle["model"]
    X = X[bundle["features"]]
    if bundle.get("is_classifier", hasattr(m, "predict_proba")):
        return m.predict_proba(X)[:, 1]
    return np.clip(m.predict(X), 0, 1)


def read_source(label, default_name, types):
    up = st.sidebar.file_uploader(label, type=types)
    if up is not None:
        return up.getvalue()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in default_name:
        p = os.path.join(here, name)
        if os.path.exists(p):
            return open(p, "rb").read()
    return None


# ------------------------------------------------------------------ sidebar
# ==================== ADVANCED SIDEBAR ====================

st.markdown("""
<style>
/* Sidebar background: preserve existing theme colors */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255, 255, 255, 0.12);
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    padding-top: 8px;
    padding-bottom: 8px;
}

/* Olympic rings divider */
.rings-bar {
    margin: 8px 0 24px 0;
}

/* Navigation heading */
section[data-testid="stSidebar"] .nav-heading {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.7;
    margin: 8px 0 14px 4px;
}

/* Space between navigation items */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* Navigation cards */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex;
    align-items: center;
    padding: 12px 14px;
    margin: 0;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 12px;
    transition: all 0.2s ease;
}

/* Hover effect */
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    border-color: rgba(255, 255, 255, 0.35);
    transform: translateX(3px);
}

/* Radio button spacing */
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 0.91rem;
    font-weight: 500;
    margin: 0;
}

/* Hide radio circles for a cleaner navigation look */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

/* Sidebar separator */
section[data-testid="stSidebar"] hr {
    margin: 24px 0 16px 0;
    opacity: 0.25;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🏅 Olympo Prediction")
    st.markdown(
        '<div class="rings-bar"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nav-heading">Main Navigation</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigate",
        [
            "Dataset Overview",
            "Best Model",
            "Total Medals Predictor",
            "Medal % Predictor",
            "Generate Report"
        ],
        format_func=lambda x: {
            "Dataset Overview": "📊  Dataset Overview",
            "Best Model": "🤖  Best Model",
            "Total Medals Predictor": "🏆  Total Medals Predictor",
            "Medal % Predictor": "📈  Medal % Predictor",
            "Generate Report": "📄  Generate Report"
        }[x],
        label_visibility="collapsed",
        key="main_navigation"
    )

    st.markdown("---")

# ==================== KEEP YOUR EXISTING DATA LOADING ====================

csv_bytes = read_source("Medals.csv", ["Medals.csv"], ["csv"])
pkl_bytes = read_source(
    "Model (.pkl)",
    ["medal_probability_model_1.pkl", "medal_probability_model.pkl"],
    ["pkl"]
)

if csv_bytes is None:
    st.title("🏅 Olympo Prediction")
    st.markdown(
        '<div class="rings-bar"></div>',
        unsafe_allow_html=True
    )
    st.info(
        "Upload **Medals.csv** in the sidebar "
        "(or place it next to app.py) to start."
    )
    st.stop()

df_raw, df = load_data(csv_bytes)
bundle = load_bundle(pkl_bytes) if pkl_bytes else None

data = df[df["Year"] <= 2008].copy()
future = df[df["Year"] >= 2010].copy()


# ------------------------------------------------------------------ models
def get_prob(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.predict(X), 0, 1)


@st.cache_resource
def train_all(_data, key):
    X, y, g = _data[FEATURES], _data["Label"], _data["Athlete"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, g))
    models = {
        "Simple Linear Regression": (LinearRegression(), ["Total"]),
        "Multiple Linear Regression": (LinearRegression(), FEATURES),
        "Ridge Regression": (make_pipeline(StandardScaler(), Ridge(alpha=10.0)), FEATURES),
        "Lasso Regression": (make_pipeline(StandardScaler(), Lasso(alpha=0.005)), FEATURES),
        "Decision Tree": (DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=42), FEATURES),
        "Random Forest": (RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=30,
                                                random_state=42, n_jobs=-1), FEATURES),
        "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), FEATURES),
    }
    rows, fitted = [], {}
    for name, (m, cols) in models.items():
        m.fit(X.iloc[tr][cols], y.iloc[tr])
        ptr, pte = get_prob(m, X.iloc[tr][cols]), get_prob(m, X.iloc[te][cols])
        rows.append({"Model": name,
                     "Train AUC": roc_auc_score(y.iloc[tr], ptr), "Test AUC": roc_auc_score(y.iloc[te], pte),
                     "Train Acc": accuracy_score(y.iloc[tr], ptr >= .5),
                     "Test Acc": accuracy_score(y.iloc[te], pte >= .5)})
        fitted[name] = (m, cols)
    res = pd.DataFrame(rows).sort_values("Test AUC", ascending=False).reset_index(drop=True)
    res["Overfit Gap"] = res["Train AUC"] - res["Test AUC"]
    return res, fitted


@st.cache_resource
def train_count_model(data, seed=42):
    df = data.copy()

    df["Season"] = df["SeasonName"].map(
        {"Summer": 1, "Winter": 0}
    )

    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Total", "Year"])
    df = df.sort_values("Year")

    if "Athlete" in df.columns:
        df["Prev_Appearances"] = (
            df.groupby("Athlete").cumcount()
        )

        df["Prev_Medals"] = (
            df.groupby("Athlete")["Total"].transform(
                lambda s: s.shift(1).fillna(0).cumsum()
            )
        )

    df["Gold_Share"] = np.where(
        df["Total"] > 0,
        df["Gold"] / df["Total"].replace(0, np.nan),
        0
    )

    df["Gold_Share"] = df["Gold_Share"].fillna(0)

    X = df[FEATURES]
    y = df["Total"].clip(lower=0)

    years = sorted(df["Year"].unique())

    if len(years) >= 3:
        split_year = years[max(1, int(len(years) * 0.8))]
        train_mask = df["Year"] < split_year
        test_mask = df["Year"] >= split_year

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            train_mask = df["Year"] < years[-1]
            test_mask = df["Year"] == years[-1]
    else:
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=seed
        )
        train_mask = np.isin(indices, train_idx)
        test_mask = np.isin(indices, test_idx)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=seed
        )
    }

    results = {}
    best_name = None
    best_mae = float("inf")

    for name, estimator in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator)
        ])

        pipe.fit(X.loc[train_mask], y.loc[train_mask])

        test_pred = np.maximum(
            0, pipe.predict(X.loc[test_mask])
        )

        mae = mean_absolute_error(y.loc[test_mask], test_pred)

        results[name] = {
            "model": pipe,
            "MAE": mae
        }

        if mae < best_mae:
            best_mae = mae
            best_name = name

    best_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", models[best_name])
    ])

    best_model.fit(X, y)

    test_pred = np.maximum(
        0, results[best_name]["model"].predict(X.loc[test_mask])
    )

    test_y = y.loc[test_mask]

    metrics = {
        "Best Model": best_name,
        "Validation MAE": mean_absolute_error(test_y, test_pred),
        "Validation RMSE": np.sqrt(
            mean_squared_error(test_y, test_pred)
        ),
        "Validation R2": (
            r2_score(test_y, test_pred)
            if len(test_y) > 1 else np.nan
        ),
        "Model Comparison": pd.DataFrame([
            {"Model": n, "Validation MAE": r["MAE"]}
            for n, r in results.items()
        ]).sort_values("Validation MAE")
    }

    return best_model, metrics


def medal_again_prob(X):
    if bundle is not None:
        return bundle_prob(bundle, X)
    _, fitted = train_all(data, len(data))
    m, cols = fitted["Logistic Regression"]
    return get_prob(m, X[cols])


def medal_type_probability(rows, medal, target_year):
    """P(win `medal` at target_year) = P(medal again, chained per Games cycle) x P(medal type | medalled)."""
    rows = rows.copy()
    gl = data[data["Label"] == 1]
    global_share = (gl["Next_" + medal] > 0).mean()
    own_share = (rows[medal] > 0).astype(float)
    k = 2.0  # smoothing toward the global rate
    type_share = (own_share * (rows["Prev_Appearances"] + 1) + global_share * k) / (rows["Prev_Appearances"] + 1 + k)

    cycles = np.maximum(1, ((target_year - rows["Year"]) // 4).astype(int))
    probs = np.ones(len(rows))
    cur = rows[FEATURES].copy()
    for c in range(cycles.max()):
        p = medal_again_prob(cur)
        active = (cycles > c).values
        probs = np.where(active, probs * p, probs)
        cur = cur.assign(Age=cur["Age"] + 4, Prev_Appearances=cur["Prev_Appearances"] + 1,
                         Prev_Medals=cur["Prev_Medals"] + cur["Total"])
    return np.clip(probs * type_share.values, 0, 1) * 100


# ================================================================== pages
if page == "Dataset Overview":
    header("Dataset Overview", "Medal winners 2000 – 2012 · one row per athlete per Games")
    c1, c2, c3, c4 = st.columns(4)
    years = c1.multiselect("Year", sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
    season = c2.selectbox("Season", ["All", "Summer", "Winter"])
    a_min, a_max = int(df["Age"].min()), int(df["Age"].max())
    ages = c3.slider("Age range", a_min, a_max, (a_min, a_max))
    name = c4.text_input("Search athlete")
    f = df[df["Year"].isin(years) & df["Age"].between(*ages)]
    if season != "All":
        f = f[f["SeasonName"] == season]
    if name:
        f = f[f["Athlete"].str.contains(name, case=False, na=False)]

    kpis([("Rows", f"{len(f):,}"), ("Athletes", f"{f['Athlete'].nunique():,}"),
          ("Gold", f"{int(f['Gold'].sum()):,}"), ("Silver", f"{int(f['Silver'].sum()):,}"),
          ("Bronze", f"{int(f['Bronze'].sum()):,}")])
    st.write("")
    g1, g2 = st.columns(2)
    by_year = f.groupby("Year")[["Gold", "Silver", "Bronze"]].sum().reset_index().melt("Year")
    g1.plotly_chart(px.bar(by_year, x="Year", y="value", color="variable", color_discrete_map=MEDAL_COLORS,
                           title="Medals by year").update_layout(**PLOT), use_container_width=True)
    g2.plotly_chart(px.histogram(f, x="Age", nbins=30, color="SeasonName", title="Age of medal winners",
                                 color_discrete_sequence=[BLUE, RED]).update_layout(**PLOT), use_container_width=True)
    g3, g4 = st.columns(2)
    top = f.groupby("Athlete")["Total"].sum().nlargest(15).reset_index()
    g3.plotly_chart(px.bar(top, x="Total", y="Athlete", orientation="h", title="Top 15 athletes",
                           color_discrete_sequence=[YELLOW]).update_layout(**PLOT, yaxis=dict(autorange="reversed")),
                    use_container_width=True)
    corr = f[FEATURES + ["Label"]].corr().round(2)
    g4.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale=[RED, "#111827", BLUE],
                              title="Correlation").update_layout(**PLOT), use_container_width=True)
    st.subheader("Data")
    st.dataframe(f.drop(columns=["Season"]), use_container_width=True, height=380)
    with st.expander("Raw data summary"):
        st.write("Shape:", df_raw.shape)
        st.dataframe(df_raw.describe().round(2), use_container_width=True)


elif page == "Best Model":
    header(
        "Best Model",
        "Explore the saved .pkl model, important features, and dataset patterns"
    )

    if bundle is None:
        st.error("No saved .pkl model found. Please load the model file.")
        st.stop()

    saved_model = bundle["model"]
    saved_name = bundle.get("model_name", "Saved Model")

    st.success(f"Loaded saved model: {saved_name}")

    feature_names = bundle.get("feature_names")

    if feature_names is None:
        feature_names = getattr(saved_model, "feature_names_in_", None)

    if feature_names is None:
        feature_names = [
            c for c in data.select_dtypes(include=np.number).columns
            if c not in ["Rank"]
        ]

    feature_names = [
        c for c in feature_names if c in data.columns
    ]

    if not feature_names:
        st.warning("No matching model features were found in the dataset.")
        st.stop()

    st.subheader("Feature Filters")

    f1, f2 = st.columns(2)

    with f1:
        selected_features = st.multiselect(
            "Select features to analyse",
            feature_names,
            default=feature_names,
            key="best_model_features"
        )

    with f2:
        graph_type = st.selectbox(
            "Feature graph",
            [
                "Feature Importance",
                "Feature Distribution",
                "Correlation Heatmap"
            ],
            key="best_model_graph_type"
        )

    if not selected_features:
        st.warning("Select at least one feature.")
        st.stop()

    st.divider()

    st.subheader("Saved Model Overview")

    k1, k2, k3 = st.columns(3)

    k1.metric("Saved Model", saved_name)
    k2.metric("Available Features", len(feature_names))
    k3.metric("Selected Features", len(selected_features))

    st.divider()

    numeric_features = [
        c for c in selected_features
        if pd.api.types.is_numeric_dtype(data[c])
    ]

    if graph_type == "Feature Importance":
        st.subheader("Feature Importance")

        estimator = (
            saved_model.steps[-1][1]
            if hasattr(saved_model, "steps")
            else saved_model
        )

        importance = getattr(
            estimator, "feature_importances_", None
        )

        importance_type = "Feature Importance"

        if importance is None and hasattr(estimator, "coef_"):
            importance = np.abs(np.ravel(estimator.coef_))
            importance_type = "Absolute Model Coefficient"

        if importance is None:
            st.info(
                "This model does not expose feature importance or coefficients. "
                "Try Feature Distribution or Correlation Heatmap."
            )
        elif len(importance) != len(feature_names):
            st.warning(
                "The model's feature importance values do not match "
                "the available feature names."
            )
        else:
            importance_df = pd.DataFrame({
                "Feature": list(feature_names),
                "Importance": importance
            })

            importance_df = importance_df[
                importance_df["Feature"].isin(selected_features)
            ].sort_values("Importance", ascending=True)

            if not importance_df.empty:
                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    title=importance_type
                )
                fig.update_layout(**PLOT)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    importance_df.sort_values(
                        "Importance", ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    elif graph_type == "Feature Distribution":
        st.subheader("Feature Distribution")

        if not numeric_features:
            st.warning("Select numeric features to display distributions.")
        else:
            distribution_feature = st.selectbox(
                "Choose a feature",
                numeric_features,
                key="best_model_distribution_feature"
            )

            fig = px.histogram(
                data,
                x=distribution_feature,
                nbins=25,
                marginal="box",
                title=f"Distribution of {distribution_feature}"
            )
            fig.update_layout(**PLOT)
            st.plotly_chart(fig, use_container_width=True)

            if len(numeric_features) >= 2:
                x_feature, y_feature = st.columns(2)

                with x_feature:
                    x_axis = st.selectbox(
                        "X-axis",
                        numeric_features,
                        index=0,
                        key="best_model_x_axis"
                    )

                with y_feature:
                    y_axis = st.selectbox(
                        "Y-axis",
                        numeric_features,
                        index=min(1, len(numeric_features) - 1),
                        key="best_model_y_axis"
                    )

                fig = px.scatter(
                    data,
                    x=x_axis,
                    y=y_axis,
                    title=f"{x_axis} vs {y_axis}",
                )
                fig.update_layout(**PLOT)
                st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Correlation Heatmap":
        st.subheader("Feature Correlation Heatmap")

        if len(numeric_features) < 2:
            st.warning(
                "Select at least two numeric features for a correlation heatmap."
            )
        else:
            correlation = data[numeric_features].corr()

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Between Selected Features"
            )
            fig.update_layout(**PLOT)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Filtered Dataset")

    if numeric_features:
        st.dataframe(
            data[numeric_features],
            use_container_width=True,
            hide_index=True
        )


elif page == "Total Medals Predictor":
    header("Total Medals Predictor", "Expected medals for each athlete at their next Games")
    model, met = train_count_model(data, len(data))
    pred = future.copy()
    pred["Predicted_Total"] = model.predict(pred[FEATURES]).round(2)
    pred["Next_Games"] = pred["Year"] + 4
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Season", ["All", "Summer", "Winter"])
    top_n = c2.slider("Show top N", 5, 50, 20)
    name = c3.text_input("Search athlete")
    p = pred if season == "All" else pred[pred["SeasonName"] == season]
    if name:
        p = p[p["Athlete"].str.contains(name, case=False, na=False)]
    kpis([("Athletes", f"{len(p):,}"), ("Expected medals", f"{p['Predicted_Total'].sum():,.0f}")])
    st.write("")
    topp = p.nlargest(top_n, "Predicted_Total")
    g1, g2 = st.columns(2)
    g1.plotly_chart(px.bar(topp, x="Predicted_Total", y="Athlete", orientation="h", color="SeasonName",
                           color_discrete_map={"Summer": BLUE, "Winter": GREEN}, title="Top predicted athletes")
                    .update_layout(**PLOT, yaxis=dict(autorange="reversed")), use_container_width=True)
    g2.plotly_chart(px.scatter(p, x="Total", y="Predicted_Total", color="Age", size="Prev_Medals",
                               hover_name="Athlete", color_continuous_scale=[BLUE, YELLOW, RED],
                               title="Last Games medals vs predicted").update_layout(**PLOT), use_container_width=True)
    st.dataframe(p.sort_values("Predicted_Total", ascending=False)[
        ["Athlete", "Year", "Next_Games", "SeasonName", "Age", "Gold", "Silver", "Bronze", "Total",
         "Prev_Medals", "Predicted_Total"]], use_container_width=True, height=380)

    st.subheader("Predict a custom athlete")
    a, b, c, d, e = st.columns(5)
    age = a.number_input("Age", 14, 60, 24)
    gold = b.number_input("Gold", 0, 10, 1)
    silver = c.number_input("Silver", 0, 10, 0)
    bronze = d.number_input("Bronze", 0, 10, 0)
    seas = e.selectbox("Season ", ["Summer", "Winter"])
    pa, pm = st.columns(2)
    prev_app = pa.number_input("Previous medal Games", 0, 10, 0)
    prev_med = pm.number_input("Previous medals", 0, 40, 0)
    tot = gold + silver + bronze
    row = pd.DataFrame([{"Age": age, "Gold": gold, "Silver": silver, "Bronze": bronze, "Total": tot,
                         "Season": int(seas == "Summer"), "Prev_Appearances": prev_app, "Prev_Medals": prev_med,
                         "Gold_Share": gold / tot if tot else 0}])
    st.success(f"Expected medals at next Games: **{model.predict(row[FEATURES])[0]:.2f}**  ·  "
               f"Chance of any medal: **{medal_again_prob(row)[0] * 100:.1f}%**")

elif page == "Medal % Predictor":
    header("Medal % Predictor", "Chance (%) that an athlete wins Gold, Silver or Bronze at an upcoming Games")
    c1, c2, c3, c4 = st.columns(4)
    medal = c1.selectbox("Medal", ["Gold", "Silver", "Bronze"])
    season = c2.selectbox("Season", ["Summer", "Winter"])
    years = [2028, 2032, 2036, 2040] if season == "Summer" else [2026, 2030, 2034, 2038]
    target = c3.selectbox("Upcoming year", years)
    top_n = c4.slider("Top N", 5, 50, 15)
    pool = future[future["SeasonName"] == season]
    athlete = st.selectbox("Athlete (optional)", ["All athletes"] + sorted(pool["Athlete"].unique()))
    pool = pool.copy()
    pool[f"{medal}_%"] = medal_type_probability(pool, medal, target).round(2)
    if athlete != "All athletes":
        one = pool[pool["Athlete"] == athlete]
        st.markdown(f"### {athlete}")
        rows = []
        for md in ["Gold", "Silver", "Bronze"]:
            for y in years:
                rows.append({"Medal": md, "Year": y, "Chance %": medal_type_probability(one, md, y)[0]})
        rr = pd.DataFrame(rows)
        kpis([(f"{md} in {target}", f"{rr[(rr.Medal == md) & (rr.Year == target)]['Chance %'].iloc[0]:.1f}%")
              for md in ["Gold", "Silver", "Bronze"]])
        st.plotly_chart(px.line(rr, x="Year", y="Chance %", color="Medal", markers=True,
                                color_discrete_map=MEDAL_COLORS, title="Chance over upcoming Games")
                        .update_layout(**PLOT), use_container_width=True)
    else:
        topp = pool.nlargest(top_n, f"{medal}_%")
        kpis([("Medal", medal), ("Games", f"{season} {target}"),
              ("Avg chance", f"{pool[f'{medal}_%'].mean():.2f}%"), ("Top chance", f"{topp[f'{medal}_%'].max():.1f}%")])
        st.write("")
        g1, g2 = st.columns(2)
        g1.plotly_chart(px.bar(topp, x=f"{medal}_%", y="Athlete", orientation="h",
                               color_discrete_sequence=[MEDAL_COLORS[medal]], title=f"Top {top_n} · {medal} chance")
                        .update_layout(**PLOT, yaxis=dict(autorange="reversed")), use_container_width=True)
        g2.plotly_chart(px.histogram(pool, x=f"{medal}_%", nbins=30, color_discrete_sequence=[BLUE],
                                     title="Distribution of chances").update_layout(**PLOT), use_container_width=True)
        st.dataframe(pool.sort_values(f"{medal}_%", ascending=False)[
            ["Athlete", "Year", "Age", "Gold", "Silver", "Bronze", "Prev_Medals", f"{medal}_%"]],
            use_container_width=True, height=380)
    st.caption("Method: chance of medalling again (saved .pkl model, chained per Games) × the athlete's "
               "smoothed share of that medal colour.")

else:  # Generate Report
    header("Generate Report", "Build a summary report and download it")
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Season", ["Summer", "Winter"])
    years = [2028, 2032, 2036] if season == "Summer" else [2026, 2030, 2034]
    target = c2.selectbox("Prediction year", years)
    top_n = c3.slider("Top athletes", 5, 30, 10)
    if st.button("Generate report", type="primary"):
        res, _ = train_all(data, len(data))
        cm, met = train_count_model(data, len(data))
        pool = future[future["SeasonName"] == season].copy()
        pool["Predicted_Total"] = cm.predict(pool[FEATURES]).round(2)
        for md in ["Gold", "Silver", "Bronze"]:
            pool[f"{md}_%"] = medal_type_probability(pool, md, target).round(2)
        top = pool.nlargest(top_n, "Predicted_Total")[
            ["Athlete", "Age", "Total", "Predicted_Total", "Gold_%", "Silver_%", "Bronze_%"]]
        best = res.iloc[0]

        def tbl(d):
            return d.to_html(index=False, float_format=lambda v: f"{v:.3f}", border=0)

        html = f"""<html><head><meta charset="utf-8"><title>Olympo Prediction Report</title>
<style>body{{background:#0a0e17;color:#e8e8e8;font-family:Georgia,serif;padding:32px;max-width:1000px;margin:auto}}
.bar{{height:6px;background:linear-gradient(90deg,{BLUE} 0 20%,{YELLOW} 20% 40%,{BLACK} 40% 60%,{GREEN} 60% 80%,{RED} 80%)}}
table{{border-collapse:collapse;width:100%;font-family:Arial;font-size:13px}}th{{background:#13203a;text-align:left}}
th,td{{padding:6px 8px;border-bottom:1px solid #1f2a3d}}h2{{color:{YELLOW}}}</style></head><body>
<h1>🏅 Olympo Prediction Report</h1><div class="bar"></div>
<p>Generated {datetime.now():%d %b %Y %H:%M} · {season} Games {target}</p>
<h2>1. Dataset overview</h2><p>{len(df):,} athlete-Games rows · {df['Athlete'].nunique():,} athletes ·
years {df['Year'].min()}–{df['Year'].max()} · Gold {int(df['Gold'].sum())}, Silver {int(df['Silver'].sum())},
Bronze {int(df['Bronze'].sum())}. Share who medalled again: {data['Label'].mean():.1%}.</p>
<h2>2. Best model</h2><p><b>{best['Model']}</b> — Test AUC {best['Test AUC']:.3f}, Test accuracy {best['Test Acc']:.1%}.
Saved model: {bundle['model_name'] if bundle else 'n/a'}.</p>{tbl(res)}
<h2>3. Total medals predictor</h2><p>{met['Best Model']} · MAE {met['Validation MAE']:.3f} · R² {met['Validation R2']:.3f}</p>
<h2>4. Top {top_n} athletes for {season} {target}</h2>{tbl(top)}
</body></html>"""
        st.session_state["report_html"] = html
        st.session_state["report_csv"] = top.to_csv(index=False)
    if "report_html" in st.session_state:
        st.components.v1.html(st.session_state["report_html"], height=700, scrolling=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download report (HTML)", st.session_state["report_html"],
                           file_name="olympo_report.html", mime="text/html")
        d2.download_button("⬇ Download predictions (CSV)", st.session_state["report_csv"],
                           file_name="olympo_predictions.csv", mime="text/csv")
