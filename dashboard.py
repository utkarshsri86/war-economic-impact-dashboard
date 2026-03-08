import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ── Page Config ──────────────────────────────────
st.set_page_config(page_title="War Economic Impact", layout="wide", page_icon="⚔️")
st.title("⚔️ War Economic Impact Dashboard")
st.markdown("Explore economic damage from conflicts & predict war costs using ML")

# ── Load Data & Model ────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('war_economic_impact_dataset.csv')

df = load_data()
model   = joblib.load('model.pkl')
scaler  = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

# ── Sidebar Filters ──────────────────────────────
st.sidebar.header("🔍 Filters")
regions        = st.sidebar.multiselect("Region",        df['Region'].unique(),        default=df['Region'].unique())
conflict_types = st.sidebar.multiselect("Conflict Type", df['Conflict_Type'].unique(), default=df['Conflict_Type'].unique())
status         = st.sidebar.multiselect("Status",        df['Status'].unique(),        default=df['Status'].unique())

filtered = df[
    df['Region'].isin(regions) &
    df['Conflict_Type'].isin(conflict_types) &
    df['Status'].isin(status)
]

# ── KPI Cards ────────────────────────────────────
st.markdown("### 📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Conflicts",    f"{len(filtered):,}")
col2.metric("Avg GDP Change",     f"{filtered['GDP_Change_%'].mean():.1f}%")
col3.metric("Avg Inflation",      f"{filtered['Inflation_Rate_%'].mean():.1f}%")
col4.metric("Avg Poverty Spike",  f"{(filtered['During_War_Poverty_Rate_%'] - filtered['Pre_War_Poverty_Rate_%']).mean():.1f}%")

st.divider()

# ── Charts ───────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig = px.box(filtered, x='Region', y='GDP_Change_%', color='Region',
                 title='GDP Change by Region')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(filtered, x='War_Duration' if 'War_Duration' in filtered else filtered.eval('End_Year - Start_Year'),
                     y='Cost_of_War_USD', color='Conflict_Type',
                     title='War Duration vs Cost', log_y=True,
                     labels={'x': 'Duration (years)', 'Cost_of_War_USD': 'Cost (USD, log)'})
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig = px.bar(filtered.groupby('Region')['Food_Insecurity_Rate_%'].mean().reset_index(),
                 x='Region', y='Food_Insecurity_Rate_%', color='Region',
                 title='Avg Food Insecurity by Region')
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.histogram(filtered, x='Inflation_Rate_%', color='Conflict_Type',
                       title='Inflation Rate Distribution', nbins=40)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── ML Prediction Section ─────────────────────────
st.markdown("### 🤖 Predict Cost of War")
st.markdown("Adjust the sliders to estimate the cost of a conflict:")

p1, p2, p3 = st.columns(3)
war_duration         = p1.slider("War Duration (years)",         1, 20, 3)
gdp_change           = p1.slider("GDP Change %",                -80, 20, -15)
inflation            = p2.slider("Inflation Rate %",             0, 200, 30)
currency_deval       = p2.slider("Currency Devaluation %",       0, 300, 50)
food_insecurity      = p3.slider("Food Insecurity Rate %",       0, 60, 20)
unemployment_spike   = p3.slider("Unemployment Spike (pp)",      0, 40, 10)

region_opt        = st.selectbox("Region",        df['Region'].unique())
conflict_type_opt = st.selectbox("Conflict Type", df['Conflict_Type'].unique())

if st.button("🔮 Predict Cost of War", type="primary"):
    from sklearn.preprocessing import LabelEncoder
    import warnings; warnings.filterwarnings('ignore')
    
    # Build input matching training features
    input_dict = {
        'War_Duration': war_duration,
        'Poverty_Increase': 10,
        'Unemployment_Increase': unemployment_spike,
        'Informal_Economy_Increase': 5,
        'GDP_Change_%': gdp_change,
        'Inflation_Rate_%': inflation,
        'Currency_Devaluation_%': currency_deval,
        'Food_Insecurity_Rate_%': food_insecurity,
        'Youth_Unemployment_Change_%': unemployment_spike * 1.5,
        'Currency_Black_Market_Rate_Gap_%': 20,
        'Conflict_Type_enc': list(df['Conflict_Type'].unique()).index(conflict_type_opt),
        'Region_enc': list(df['Region'].unique()).index(region_opt),
        'Status_enc': 0,
        'Black_Market_Activity_Level_enc': 1,
        'Most_Affected_Sector_enc': 0,
        'War_Profiteering_Documented_enc': 0,
    }
    
    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)
    prediction = np.expm1(model.predict(input_scaled)[0])
    
    st.success(f"💰 Estimated Cost of War: **${prediction:,.0f}**")
    st.info("This estimate is based on patterns learned from 100,000 historical conflict records.")