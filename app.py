import os
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_model import (
    COST_COLS,
    FEATURE_COLS,
    TARGET_COL,
    clean_enrich_data,
    get_feature_options,
    load_raw_data,
    load_trained_pipeline,
    make_input_dataframe,
    recompute_metrics_with_loaded_pipeline,
    validate_user_inputs,
)


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Study Abroad Cost Explorer & Model",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 Study Abroad Cost Explorer & ML Model")
st.caption("Interactive exploration + prediction using the trained pipeline from train_model.ipynb")


# -------------------------
# Caching helpers
# -------------------------
@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    df = load_raw_data("International_Education_Costs.csv")
    df = clean_enrich_data(df)
    return df


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return load_trained_pipeline("model.pkl")


# -------------------------
# Sidebar Navigation
# -------------------------
section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Exploration",
        "Model Demonstration",
        "About",
    ],
)


def render_overview(df: pd.DataFrame):
    st.subheader("Overview")
    st.write(
        """
        Use this app to explore the international education cost dataset and
        try the trained machine learning model (saved as `model.pkl`).
        The model pipeline mirrors the preprocessing in `train_model.ipynb`:
        One-Hot Encoding for categorical features and Standard Scaling for numerical ones.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Countries", df["Country"].nunique())
    with c2:
        st.metric("Universities", df.get("University", pd.Series(dtype=str)).nunique())
    with c3:
        st.metric("Programs", df["Program"].nunique())
    with c4:
        st.metric("Rows", len(df))

    st.markdown("---")
    st.subheader("Data Quality Snapshot")
    # Missing values per column
    missing = df.isna().sum().sort_values(ascending=False)
    zeros = (df == 0).sum().sort_values(ascending=False)
    dq = pd.DataFrame({"missing": missing, "zeros": zeros})
    st.dataframe(dq, use_container_width=True)


def render_data_exploration(df: pd.DataFrame):
    st.subheader("Data Exploration")

    # Filters
    opts = get_feature_options(df)
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        countries = c1.multiselect("Country", opts["Country"], default=[])
        levels = c2.multiselect("Level", opts["Level"], default=[])
        programs = c3.multiselect("Program", opts["Program"], default=[])
        duration_range = c4.slider(
            "Duration (years)",
            min_value=float(max(0.5, df["Duration_Years"].min() if "Duration_Years" in df else 0.5)),
            max_value=float(max(1.0, df["Duration_Years"].max() if "Duration_Years" in df else 6.0)),
            value=(float(df["Duration_Years"].min()), float(df["Duration_Years"].max()))
            if "Duration_Years" in df
            else (0.5, 6.0),
            step=0.5,
        )

    filtered = df.copy()
    if countries:
        filtered = filtered[filtered["Country"].isin(countries)]
    if levels:
        filtered = filtered[filtered["Level"].isin(levels)]
    if programs:
        filtered = filtered[filtered["Program"].isin(programs)]
    if "Duration_Years" in filtered:
        filtered = filtered[
            (filtered["Duration_Years"] >= duration_range[0])
            & (filtered["Duration_Years"] <= duration_range[1])
        ]

    st.caption(f"Showing {len(filtered)} of {len(df)} rows after filters")

    # Summary stats
    st.markdown("### Basic Statistics")
    st.dataframe(filtered.describe(include="all").transpose(), use_container_width=True)

    # Visualizations
    st.markdown("### Visualizations")
    v1c1, v1c2 = st.columns(2)
    with v1c1:
        if "Tuition_USD" in filtered:
            fig = px.histogram(filtered, x="Tuition_USD", nbins=30, title="Tuition (USD) Distribution")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    with v1c2:
        if "Living_Cost_Index" in filtered:
            fig = px.histogram(
                filtered, x="Living_Cost_Index", nbins=30, title="Living Cost Index Distribution"
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    v2c1, v2c2 = st.columns(2)
    with v2c1:
        if {"Country", "Tuition_USD"}.issubset(filtered.columns):
            avg = (
                filtered.groupby("Country")["Tuition_USD"].mean().reset_index().sort_values("Tuition_USD", ascending=False)
            )
            fig = px.bar(avg.head(25), x="Country", y="Tuition_USD", title="Avg Tuition by Country (Top 25)")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    with v2c2:
        if {"Level", TARGET_COL}.issubset(filtered.columns):
            avg = filtered.groupby("Level")[TARGET_COL].mean().reset_index()
            fig = px.bar(avg, x="Level", y=TARGET_COL, title="Avg Estimated Annual Cost by Level")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    v3c1, v3c2 = st.columns(2)
    with v3c1:
        if {"Living_Cost_Index", "Rent_USD"}.issubset(filtered.columns):
            fig = px.scatter(
                filtered,
                x="Living_Cost_Index",
                y="Rent_USD",
                color="Country" if "Country" in filtered else None,
                title="Living Cost vs Rent",
                trendline="ols",
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    with v3c2:
        if {"Country", TARGET_COL}.issubset(filtered.columns):
            avg = filtered.groupby("Country")[TARGET_COL].mean().reset_index()
            fig = px.choropleth(
                avg,
                locations="Country",
                locationmode="country names",
                color=TARGET_COL,
                color_continuous_scale="Viridis",
                title="Global Map: Avg Estimated Annual Cost",
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def render_model_demo(df: pd.DataFrame):
    st.subheader("Model Demonstration")
    st.write(
        """
        The loaded pipeline includes preprocessing (One-Hot Encoding + Scaling) and a trained regressor.
        Provide inputs below to predict the Estimated Annual Cost.
        """
    )
    
    # Add helpful explanation
    with st.expander("ℹ️ What do these inputs mean?", expanded=False):
        st.markdown("""
        ### 📚 Input Guide
        
        **Country, Level & Program:** Select your destination and field of study.
        
        **Duration:** How many years your program takes (e.g., Bachelor = 3-4 years, Master = 1-2 years).
        
        **Living Cost (Affordability):** 
        - Think of this as a "cost of living score" for the city
        - **40-70:** Budget-friendly cities (Prague, Lisbon, Budapest)
        - **100:** Average cost cities (Boston, Toronto, Amsterdam)
        - **150-200:** Expensive cities (London, Zurich, Singapore)
        
        **Exchange Rate:**
        - How much 1 US Dollar is worth in the local currency
        - **If rate is 1.0:** Country uses USD (USA)
        - **If rate is less than 1:** Local currency is stronger (e.g., 0.79 for British Pound)
        - **If rate is more than 1:** Local currency is weaker (e.g., 83 for Indian Rupee)
        - Click the currency buttons for quick selection!
        """)

    try:
        pipe = get_pipeline()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Please ensure model.pkl exists and was trained with a compatible scikit-learn version.")
        return

    # Show metrics recomputed on a fresh split for transparency
    try:
        with st.spinner("Computing model performance on held-out split..."):
            metrics = recompute_metrics_with_loaded_pipeline(pipe, df)
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE ($)", f"{metrics.mae:,.0f}")
        m2.metric("R²", f"{metrics.r2:.3f}")
        m3.metric("Test Size", f"{metrics.test_size}")
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")

    st.markdown("---")
    st.markdown("### 📝 Input Parameters")
    
    with st.expander("💡 See Examples for Different Countries", expanded=False):
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.write("**🇺🇸 USA Example**")
            st.write("""
            - **Country:** USA
            - **Level:** Master
            - **Program:** Computer Science
            - **Duration:** 2 years
            - **Living Cost:** 100 (Moderate)
            - **Exchange Rate:** 1.0 (USD)
            """)
        with ex2:
            st.write("**🇬🇧 UK Example**")
            st.write("""
            - **Country:** United Kingdom
            - **Level:** Master
            - **Program:** Business
            - **Duration:** 1.5 years
            - **Living Cost:** 140 (Expensive)
            - **Exchange Rate:** 0.79 (GBP)
            """)
        with ex3:
            st.write("**🇩🇪 Germany Example**")
            st.write("""
            - **Country:** Germany
            - **Level:** Bachelor
            - **Program:** Engineering
            - **Duration:** 3 years
            - **Living Cost:** 75 (Affordable)
            - **Exchange Rate:** 0.92 (EUR)
            """)

    opts = get_feature_options(df)
    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", options=opts["Country"], help="Select the destination country") if opts["Country"] else st.text_input("Country")
    with c2:
        level = st.selectbox("Level", options=opts["Level"], help="Select degree level") if opts["Level"] else st.text_input("Level")
    with c3:
        program = st.selectbox("Program", options=opts["Program"], help="Select your program/major") if opts["Program"] else st.text_input("Program")

    c4, c5, c6 = st.columns(3)
    with c4:
        duration_years = st.number_input(
            "Duration (Years)", 
            min_value=0.5, 
            max_value=10.0, 
            value=2.0, 
            step=0.5,
            help="Program duration in years"
        )
    with c5:
        st.write("**Living Cost (Affordability)**")
        st.caption("How expensive is the city? Lower = Cheaper, Higher = More Expensive")
        
        # Quick selection buttons
        col_low, col_med, col_high = st.columns(3)
        with col_low:
            if st.button("💰 Low Cost"):
                st.session_state.living_cost = 60.0
        with col_med:
            if st.button("💵 Medium Cost"):
                st.session_state.living_cost = 100.0
        with col_high:
            if st.button("💸 High Cost"):
                st.session_state.living_cost = 150.0
        
        living_cost_index = st.slider(
            "Adjust Cost Level",
            min_value=40.0,
            max_value=200.0,
            value=st.session_state.get('living_cost', 100.0),
            step=5.0,
            help="40-70: Affordable cities (e.g., Prague, Lisbon)\n100: Average cost (e.g., Boston, Toronto)\n150-200: Expensive cities (e.g., London, Zurich)"
        )
        
        # Visual indicator
        if living_cost_index < 70:
            st.success("✅ Budget-Friendly")
        elif living_cost_index < 120:
            st.info("ℹ️ Moderate Cost")
        else:
            st.warning("⚠️ Expensive City")
    with c6:
        st.write("**Currency Exchange Rate**")
        st.caption("How much is 1 USD worth in local currency?")
        
        # Currency presets
        currency_col1, currency_col2, currency_col3 = st.columns(3)
        with currency_col1:
            if st.button("🇺🇸 USD"):
                st.session_state.exchange = 1.0
        with currency_col2:
            if st.button("🇪🇺 EUR"):
                st.session_state.exchange = 0.92
        with currency_col3:
            if st.button("🇬🇧 GBP"):
                st.session_state.exchange = 0.79
        
        currency_col4, currency_col5, currency_col6 = st.columns(3)
        with currency_col4:
            if st.button("🇨🇦 CAD"):
                st.session_state.exchange = 1.36
        with currency_col5:
            if st.button("🇦🇺 AUD"):
                st.session_state.exchange = 1.53
        with currency_col6:
            if st.button("🇮🇳 INR"):
                st.session_state.exchange = 83.0
        
        exchange_rate = st.number_input(
            "Enter Exchange Rate (1 USD = ? Local Currency)",
            min_value=0.1,
            max_value=200.0,
            value=st.session_state.get('exchange', 1.0),
            step=0.1,
            format="%.2f",
            help="Examples:\n• USA: 1.00 (USD)\n• Europe: ~0.92 (EUR)\n• UK: ~0.79 (GBP)\n• Canada: ~1.36 (CAD)\n• Australia: ~1.53 (AUD)\n• India: ~83.00 (INR)"
        )
        
        # Show conversion example
        if exchange_rate != 1.0:
            converted = 1000 * exchange_rate
            st.caption(f"📊 Example: $1,000 USD ≈ {converted:,.0f} local currency")

    error = validate_user_inputs(
        country=country,
        level=level,
        program=program,
        duration_years=duration_years,
        living_cost_index=living_cost_index,
        exchange_rate=exchange_rate,
    )

    st.markdown("###")
    predict_btn = st.button("🔮 Predict Estimated Annual Cost", type="primary", use_container_width=True)
    
    if predict_btn:
        if error:
            st.error(f"❌ Validation Error: {error}")
        else:
            try:
                with st.spinner("🤖 Making prediction..."):
                    X = make_input_dataframe(
                        country=country,
                        level=level,
                        program=program,
                        duration_years=duration_years,
                        living_cost_index=living_cost_index,
                        exchange_rate=exchange_rate,
                    )
                    y_pred = float(pipe.predict(X)[0])
                
                st.success(f"### 💰 Predicted Annual Cost: ${y_pred:,.2f}")
                
                # Show breakdown
                with st.expander("📊 View Cost Breakdown Estimate", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Prediction", f"${y_pred:,.2f}")
                        st.caption("This is the AI model's prediction based on historical data")
                    with col2:
                        monthly_est = y_pred / 12
                        st.metric("Estimated Monthly", f"${monthly_est:,.2f}")
                        st.caption("Approximate monthly cost")
                
                st.info(
                    "ℹ️ **Note:** This prediction is based on the 6 input features. "
                    "The model automatically handles preprocessing (encoding & scaling). "
                    "Actual costs may vary based on lifestyle, scholarships, and other factors."
                )
            except ValueError as ve:
                st.error(f"❌ Value Error: {ve}")
                st.info("Please check that all input values are valid numbers.")
            except Exception as ex:
                st.error(f"❌ Prediction Error: {str(ex)}")
                st.info("This might be due to an unseen category value. Try selecting different options.")
                with st.expander("🔍 Technical Details"):
                    st.exception(ex)


def render_about():
    st.subheader("About")
    st.write(
        """
        - Dataset: `International_Education_Costs.csv` with columns for tuition, living costs, rent, visa, insurance, and more.
        - Target: `Estimated_Annual_Cost` computed as tuition + living (index × $12k) + rent×12 + visa + insurance.
        - Features used by the model: Country, Level, Program, Duration_Years, Living_Cost_Index, Exchange_Rate.
        - Training reference: `train_model.ipynb` builds a preprocessing pipeline and evaluates multiple regressors.
        - Model file: `model.pkl` stores the best performing pipeline.

        Performance is recomputed in-app using the same split seed for transparency. Caching is enabled for both data and model.
        """
    )


# -------------------------
# Main
# -------------------------
try:
    data_df = get_dataset()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

if section == "Overview":
    render_overview(data_df)
elif section == "Data Exploration":
    render_data_exploration(data_df)
elif section == "Model Demonstration":
    render_model_demo(data_df)
else:
    render_about()
