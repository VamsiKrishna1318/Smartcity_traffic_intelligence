
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.20);
    padding: 12px;
    border-radius: 10px;
}
.small-note {font-size: 0.88rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/smart_city_traffic_cleaned.csv", parse_dates=["Date"])
    df["Congestion_Level"] = pd.Categorical(
        df["Congestion_Level"],
        categories=["Low", "Medium", "High"],
        ordered=True
    )
    return df

@st.cache_resource
def train_model(data):
    features = ["Hour", "Vehicle_Count", "Avg_Speed", "Location", "Is_Weekend"]
    target = "Congestion_Level"
    X = data[features].copy()
    y = data[target].astype(str).copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Location"]),
            ("num", "passthrough",
             ["Hour", "Vehicle_Count", "Avg_Speed", "Is_Weekend"])
        ]
    )
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return pipe, y_test, pred

df = load_data()

st.title("🚦 Smart City Traffic Intelligence")
st.caption(
    "Data analytics, congestion intelligence and interactive exploration — "
    "built with Python, Pandas, Plotly and Streamlit."
)

# ---------- Navigation ----------
page = st.sidebar.radio(
    "Navigate",
    ["Executive Dashboard", "Deep Dive", "Anomalies", "Congestion Prediction", "Data Explorer"]
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

locations = sorted(df["Location"].unique())
selected_locations = st.sidebar.multiselect(
    "Junction",
    locations,
    default=locations
)

date_min, date_max = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

selected_levels = st.sidebar.multiselect(
    "Congestion",
    ["Low", "Medium", "High"],
    default=["Low", "Medium", "High"]
)

hours = st.sidebar.slider("Hour range", 0, 23, (0, 23))

filtered = df[
    df["Location"].isin(selected_locations) &
    df["Congestion_Level"].isin(selected_levels) &
    df["Hour"].between(hours[0], hours[1])
].copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[
        (filtered["Date"].dt.date >= date_range[0]) &
        (filtered["Date"].dt.date <= date_range[1])
    ]

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ---------- Shared calculations ----------
high_rate = filtered["Congestion_Level"].eq("High").mean() * 100
avg_traffic = filtered["Vehicle_Count"].mean()
avg_speed = filtered["Avg_Speed"].mean()

# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================
if page == "Executive Dashboard":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(filtered):,}")
    c2.metric("Avg Traffic", f"{avg_traffic:.1f}")
    c3.metric("Avg Speed", f"{avg_speed:.1f}")
    c4.metric("High Congestion", f"{high_rate:.1f}%")
    c5.metric("Junctions", filtered["Location"].nunique())

    st.divider()

    left, right = st.columns(2)

    with left:
        hourly = filtered.groupby("Hour", as_index=False)["Vehicle_Count"].mean()
        fig = px.line(
            hourly, x="Hour", y="Vehicle_Count", markers=True,
            title="Average Traffic Volume by Hour",
            labels={"Vehicle_Count": "Average Vehicles"}
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        junction = filtered.groupby("Location", as_index=False).agg(
            Avg_Vehicles=("Vehicle_Count", "mean"),
            Avg_Speed=("Avg_Speed", "mean")
        )
        fig = px.bar(
            junction, x="Location", y="Avg_Vehicles",
            text_auto=".1f", title="Average Traffic by Junction"
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        con = filtered["Congestion_Level"].value_counts().reindex(
            ["Low", "Medium", "High"]
        ).fillna(0).reset_index()
        con.columns = ["Congestion_Level", "Count"]
        fig = px.bar(
            con, x="Congestion_Level", y="Count",
            text_auto=True, title="Congestion Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        day = filtered.assign(
            Day_Type=filtered["Is_Weekend"].map({False: "Weekday", True: "Weekend"})
        ).groupby("Day_Type", as_index=False)["Vehicle_Count"].mean()
        fig = px.bar(
            day, x="Day_Type", y="Vehicle_Count",
            text_auto=".1f", title="Weekday vs Weekend Traffic"
        )
        st.plotly_chart(fig, use_container_width=True)

    heat = filtered.pivot_table(
        index="Location", columns="Hour",
        values="Vehicle_Count", aggfunc="mean"
    )
    fig = px.imshow(
        heat, aspect="auto",
        labels={"x": "Hour", "y": "Junction", "color": "Avg Vehicles"},
        title="Traffic Intensity by Junction and Hour"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key findings")
    hourly_all = filtered.groupby("Hour")["Vehicle_Count"].mean()
    top_hour = int(hourly_all.idxmax())
    top_hour_value = hourly_all.max()
    top_junction = filtered.groupby("Location")["Vehicle_Count"].mean().idxmax()

    st.markdown(f"""
    - **Busiest observed hour:** {top_hour:02d}:00, averaging **{top_hour_value:.1f} vehicles**.
    - **Highest-volume junction:** **{top_junction}** within the selected data.
    - **High-congestion share:** **{high_rate:.1f}%** of selected observations.
    - Traffic volume and speed should be interpreted separately because correlation alone does not establish causation.
    """)

# ============================================================
# DEEP DIVE
# ============================================================
elif page == "Deep Dive":
    st.header("🔎 Deep Dive Analysis")

    tab1, tab2, tab3 = st.tabs(
        ["Time Patterns", "Junction Patterns", "Relationships"]
    )

    with tab1:
        hourly = filtered.groupby("Hour", as_index=False).agg(
            Avg_Vehicles=("Vehicle_Count", "mean"),
            Avg_Speed=("Avg_Speed", "mean"),
            High_Congestion_Rate=(
                "Congestion_Level",
                lambda x: (x == "High").mean() * 100
            )
        )

        fig = px.line(
            hourly, x="Hour", y="Avg_Vehicles",
            markers=True, title="Average Traffic by Hour"
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            hourly, x="Hour", y="High_Congestion_Rate",
            markers=True, title="High Congestion Rate by Hour"
        )
        fig.update_layout(xaxis=dict(dtick=1))
        fig.update_yaxes(title="High Congestion Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        junction = filtered.groupby("Location", as_index=False).agg(
            Avg_Traffic=("Vehicle_Count", "mean"),
            Median_Traffic=("Vehicle_Count", "median"),
            Traffic_Std=("Vehicle_Count", "std"),
            Avg_Speed=("Avg_Speed", "mean"),
            High_Congestion_Rate=(
                "Congestion_Level",
                lambda x: (x == "High").mean() * 100
            )
        )

        st.dataframe(junction.round(2), use_container_width=True, hide_index=True)

        fig = px.bar(
            junction, x="Location", y="High_Congestion_Rate",
            text_auto=".1f",
            title="High Congestion Rate by Junction"
        )
        fig.update_yaxes(title="High Congestion Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr = filtered[["Vehicle_Count", "Avg_Speed"]].corr().iloc[0, 1]
        st.metric("Vehicle Count ↔ Average Speed correlation", f"{corr:.3f}")

        fig = px.scatter(
            filtered,
            x="Vehicle_Count",
            y="Avg_Speed",
            color="Congestion_Level",
            hover_data=["Date", "Hour", "Location"],
            title="Vehicle Count vs Average Speed"
        )
        st.plotly_chart(fig, use_container_width=True)

        corr_matrix = filtered[["Vehicle_Count", "Avg_Speed", "Hour"]].corr()
        fig = px.imshow(
            corr_matrix, text_auto=".2f",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ANOMALIES
# ============================================================
elif page == "Anomalies":
    st.header("⚠️ Traffic Anomaly Analysis")
    st.write(
        "Potential anomalies are identified using the IQR method. "
        "An anomaly is a statistical observation, not automatically an error."
    )

    q1, q3 = filtered["Vehicle_Count"].quantile([.25, .75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    anomaly = filtered[
        (filtered["Vehicle_Count"] < low) |
        (filtered["Vehicle_Count"] > high)
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Q1", f"{q1:.1f}")
    c2.metric("Q3", f"{q3:.1f}")
    c3.metric("IQR", f"{iqr:.1f}")

    st.metric("Potential traffic-count anomalies", f"{len(anomaly):,}")

    if anomaly.empty:
        st.success(
            "No vehicle-count observations fall outside the 1.5×IQR rule "
            "for the selected data."
        )
    else:
        st.dataframe(
            anomaly.sort_values("Vehicle_Count", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Traffic distribution")
    fig = px.box(
        filtered, y="Vehicle_Count", x="Location",
        points="outliers", title="Vehicle Count Distribution by Junction"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CONGESTION PREDICTION
# ============================================================
elif page == "Congestion Prediction":
    st.header("🤖 Congestion Prediction — Machine Learning Extension")
    st.write(
        "A Random Forest classifier is used as an optional predictive layer. "
        "The target is the dataset's existing Congestion_Level label."
    )

    model, y_test, pred = train_model(df)
    accuracy = accuracy_score(y_test, pred)

    st.metric("Hold-out test accuracy", f"{accuracy * 100:.2f}%")
    st.caption(
        "This metric is a test-set measurement on this dataset; it should not be "
        "treated as evidence of real-world deployment performance."
    )

    cm = confusion_matrix(y_test, pred, labels=["Low", "Medium", "High"])
    fig = px.imshow(
        cm,
        x=["Low", "Medium", "High"],
        y=["Low", "Medium", "High"],
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Try a prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        p_location = st.selectbox("Junction", sorted(df["Location"].unique()))
        p_hour = st.slider("Hour", 0, 23, 18)
    with c2:
        p_vehicles = st.number_input(
            "Vehicle count", min_value=0, max_value=1000, value=180
        )
        p_speed = st.number_input(
            "Average speed", min_value=0, max_value=150, value=35
        )
    with c3:
        p_weekend = st.selectbox("Weekend?", ["No", "Yes"])

    row = pd.DataFrame([{
        "Hour": p_hour,
        "Vehicle_Count": p_vehicles,
        "Avg_Speed": p_speed,
        "Location": p_location,
        "Is_Weekend": p_weekend == "Yes"
    }])

    result = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["classifier"].classes_

    st.success(f"Predicted congestion level: **{result}**")

    prob_df = pd.DataFrame({
        "Congestion_Level": classes,
        "Probability": probabilities * 100
    })
    fig = px.bar(
        prob_df, x="Congestion_Level", y="Probability",
        text_auto=".1f", title="Prediction probabilities"
    )
    fig.update_yaxes(title="Probability (%)", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DATA EXPLORER
# ============================================================
else:
    st.header("📋 Data Explorer")
    st.write(f"Showing **{len(filtered):,}** records after applying the filters.")

    st.dataframe(
        filtered.sort_values(["Date", "Hour"]),
        use_container_width=True,
        hide_index=True
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv,
        file_name="filtered_traffic_data.csv",
        mime="text/csv"
    )

    st.subheader("Descriptive statistics")
    numeric = filtered[["Vehicle_Count", "Avg_Speed"]].describe().T
    st.dataframe(numeric.round(2), use_container_width=True)
