import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="War Economic Impact Dashboard",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #e63946;
        margin: 5px 0;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #e63946; }
    .metric-label { font-size: 13px; color: #aaaaaa; margin-top: 4px; }
    .section-header {
        background: linear-gradient(90deg, #e63946, #457b9d);
        padding: 10px 20px;
        border-radius: 8px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin: 20px 0 10px 0;
    }
    .stSelectbox label { color: #ffffff !important; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #e63946; }
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA & MODEL ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('war_economic_impact_dataset.csv')

@st.cache_resource
def load_model():
    model    = joblib.load('model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

df = load_data()
model, scaler, features = load_model()

# Feature engineer for display
df['War_Duration'] = df['End_Year'] - df['Start_Year']
df['Poverty_Increase'] = df['During_War_Poverty_Rate_%'] - df['Pre_War_Poverty_Rate_%']
df['Informal_Economy_Increase'] = df['Informal_Economy_Size_During_War_%'] - df['Informal_Economy_Size_Pre_War_%']


# ── SIDEBAR ──────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; font-size: 48px; padding: 10px 0'>🕊️</div>
<h2 style='text-align:center; color:#e63946;'>War Economic<br>Dashboard</h2>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.header("🔍 Filter Data")

regions = st.sidebar.multiselect(
    "Region",
    options=sorted(df['Region'].unique()),
    default=sorted(df['Region'].unique())
)

conflict_types = st.sidebar.multiselect(
    "Conflict Type",
    options=sorted(df['Conflict_Type'].unique()),
    default=sorted(df['Conflict_Type'].unique())
)

status_filter = st.sidebar.multiselect(
    "Status",
    options=sorted(df['Status'].unique()),
    default=sorted(df['Status'].unique())
)

year_range = st.sidebar.slider(
    "Start Year Range",
    min_value=int(df['Start_Year'].min()),
    max_value=int(df['Start_Year'].max()),
    value=(int(df['Start_Year'].min()), int(df['Start_Year'].max()))
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Dataset Info**")
st.sidebar.info(f"Total Records: {len(df):,}\nColumns: {df.shape[1]}")

# ── APPLY FILTERS ────────────────────────────────────────────────────
filtered = df[
    df['Region'].isin(regions) &
    df['Conflict_Type'].isin(conflict_types) &
    df['Status'].isin(status_filter) &
    df['Start_Year'].between(year_range[0], year_range[1])
]

# ── HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
            padding: 30px; border-radius: 12px; margin-bottom: 20px;
            border: 1px solid #e63946;'>
    <h1 style='color: white; margin:0; font-size: 36px;'>
        ⚔️ War Economic Impact Dashboard
    </h1>
    <p style='color: #aaaaaa; margin: 8px 0 0 0; font-size: 16px;'>
        Explore economic damage from 100,000 conflict records & predict unemployment using ML
    </p>
</div>
""", unsafe_allow_html=True)

# ── TAB NAVIGATION ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "💰 Economic Impact",
    "👥 Poverty & Unemployment",
    "🤖 ML Predictor"
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📌 Key Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Conflicts",    f"{len(filtered):,}")
    c2.metric("Avg GDP Change",     f"{filtered['GDP_Change_%'].mean():.1f}%",
              delta=f"{filtered['GDP_Change_%'].mean():.1f}%")
    c3.metric("Avg Inflation",      f"{filtered['Inflation_Rate_%'].mean():.1f}%")
    c4.metric("Avg Poverty Spike",  f"{filtered['Poverty_Increase'].mean():.1f}%")
    c5.metric("Ongoing Conflicts",  f"{len(filtered[filtered['Status']=='Ongoing']):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Conflict type donut
        ct_counts = filtered['Conflict_Type'].value_counts().reset_index()
        ct_counts.columns = ['Conflict_Type', 'Count']
        fig = px.pie(ct_counts, values='Count', names='Conflict_Type',
                     title='Conflict Type Distribution',
                     hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Region bar
        region_counts = filtered['Region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']
        fig = px.bar(region_counts, x='Region', y='Count', color='Region',
                     title='Conflicts by Region',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Timeline
    st.markdown("### 📅 Conflicts Over Time")
    timeline = filtered.groupby('Start_Year').size().reset_index(name='Count')
    fig = px.area(timeline, x='Start_Year', y='Count',
                  title='Number of Conflicts Started Per Year',
                  color_discrete_sequence=['#e63946'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='white')
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — ECONOMIC IMPACT
# ════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        # GDP by region box plot
        fig = px.box(filtered, x='Region', y='GDP_Change_%', color='Region',
                     title='GDP Change % by Region',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.5)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Inflation by conflict type
        avg_inf = filtered.groupby('Conflict_Type')['Inflation_Rate_%'].mean().reset_index()
        fig = px.bar(avg_inf, x='Conflict_Type', y='Inflation_Rate_%',
                     color='Conflict_Type',
                     title='Avg Inflation Rate by Conflict Type',
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Currency devaluation scatter
        fig = px.scatter(filtered, x='Inflation_Rate_%', y='Currency_Devaluation_%',
                         color='Region', size='War_Duration',
                         title='Inflation vs Currency Devaluation',
                         hover_data=['Conflict_Name', 'Primary_Country'],
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Black market activity
        bm = filtered['Black_Market_Activity_Level'].value_counts().reset_index()
        bm.columns = ['Level', 'Count']
        fig = px.pie(bm, values='Count', names='Level',
                     title='Black Market Activity Level',
                     hole=0.4,
                     color_discrete_sequence=['#e63946','#f4a261','#2a9d8f','#457b9d'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    # Informal economy comparison
    st.markdown("### 🏪 Informal Economy: Pre-War vs During War")
    inf_data = filtered.groupby('Region').agg(
        Pre_War=('Informal_Economy_Size_Pre_War_%', 'mean'),
        During_War=('Informal_Economy_Size_During_War_%', 'mean')
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Pre-War',    x=inf_data['Region'], y=inf_data['Pre_War'],
                         marker_color='#457b9d'))
    fig.add_trace(go.Bar(name='During War', x=inf_data['Region'], y=inf_data['During_War'],
                         marker_color='#e63946'))
    fig.update_layout(barmode='group', title='Informal Economy Size by Region (%)',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='white')
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — POVERTY & UNEMPLOYMENT
# ════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        # Poverty comparison grouped bar
        pov_data = filtered.groupby('Region').agg(
            Pre_War=('Pre_War_Poverty_Rate_%', 'mean'),
            During_War=('During_War_Poverty_Rate_%', 'mean')
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Pre-War',    x=pov_data['Region'], y=pov_data['Pre_War'],
                             marker_color='#457b9d'))
        fig.add_trace(go.Bar(name='During War', x=pov_data['Region'], y=pov_data['During_War'],
                             marker_color='#e63946'))
        fig.update_layout(barmode='group', title='Poverty Rate: Pre vs During War (%)',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Unemployment comparison
        unemp_data = filtered.groupby('Region').agg(
            Pre_War=('Pre_War_Unemployment_%', 'mean'),
            During_War=('During_War_Unemployment_%', 'mean')
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Pre-War',    x=unemp_data['Region'], y=unemp_data['Pre_War'],
                             marker_color='#2a9d8f'))
        fig.add_trace(go.Bar(name='During War', x=unemp_data['Region'], y=unemp_data['During_War'],
                             marker_color='#f4a261'))
        fig.update_layout(barmode='group', title='Unemployment Rate: Pre vs During War (%)',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Food insecurity by region
        fi = filtered.groupby('Region')['Food_Insecurity_Rate_%'].mean().reset_index()
        fig = px.bar(fi, x='Region', y='Food_Insecurity_Rate_%', color='Region',
                     title='Avg Food Insecurity Rate by Region (%)',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Extreme poverty distribution
        fig = px.histogram(filtered, x='Extreme_Poverty_Rate_%', color='Conflict_Type',
                           title='Extreme Poverty Rate Distribution',
                           nbins=40, barmode='overlay', opacity=0.7,
                           color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    # Youth unemployment heatmap
    st.markdown("### 👶 Youth Unemployment Change by Region & Conflict Type")
    heat_data = filtered.groupby(['Region', 'Conflict_Type'])['Youth_Unemployment_Change_%'].mean().unstack()
    fig = px.imshow(heat_data,
                    color_continuous_scale='Reds',
                    title='Avg Youth Unemployment Change % (Heatmap)',
                    aspect='auto')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 4 — ML PREDICTOR
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 Predict During-War Unemployment Rate")
    st.markdown("Adjust the inputs below to get an ML prediction using **Gradient Boosting (R²=0.94)**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Economic Indicators**")
        gdp_change        = st.slider("GDP Change %",              -85,  -5,  -30)
        inflation         = st.slider("Inflation Rate %",            0, 295,   60)
        currency_deval    = st.slider("Currency Devaluation %",      0, 300,   80)
        food_insecurity   = st.slider("Food Insecurity Rate %",      3,  86,   20)

    with col2:
        st.markdown("**🔧 Conflict Details**")
        war_duration      = st.slider("War Duration (years)",        1,  30,    5)
        poverty_increase  = st.slider("Poverty Increase %",          0,  84,   12)
        informal_econ     = st.slider("Informal Economy Increase %", 2,  60,   25)
        youth_unemp       = st.slider("Youth Unemployment Change %", 0, 108,   18)

    with col3:
        st.markdown("**🌍 Conflict Profile**")
        bm_gap            = st.slider("Black Market Rate Gap %",     0, 500,  100)
        region_sel        = st.selectbox("🌍 Region", sorted(df["Region"].unique()))
        conflict_sel      = st.selectbox("⚔️ Conflict Type", sorted(df["Conflict_Type"].unique()))
        status_sel        = st.selectbox("📌 Conflict Status", sorted(df["Status"].unique()))
        bm_level_sel      = st.selectbox("🕵️ Black Market Level", sorted(df["Black_Market_Activity_Level"].unique()))
        sector_sel        = st.selectbox("🏭 Most Affected Sector", sorted(df["Most_Affected_Sector"].unique()))
        profiteering_sel  = st.selectbox("💰 War Profiteering Documented? (Yes / No)", sorted(df["War_Profiteering_Documented"].unique()))

    st.markdown("---")

    if st.button("🔮 Predict Unemployment Rate", type="primary", use_container_width=True):

        # Encode categoricals same way as training
        region_enc     = sorted(df['Region'].unique()).index(region_sel)
        conflict_enc   = sorted(df['Conflict_Type'].unique()).index(conflict_sel)
        status_enc     = sorted(df['Status'].unique()).index(status_sel)
        bm_enc         = sorted(df['Black_Market_Activity_Level'].unique()).index(bm_level_sel)
        sector_enc     = sorted(df['Most_Affected_Sector'].unique()).index(sector_sel)
        profit_enc     = sorted(df['War_Profiteering_Documented'].unique()).index(profiteering_sel)

        input_dict = {
            'War_Duration'                    : war_duration,
            'Poverty_Increase'                : poverty_increase,
            'Informal_Economy_Increase'       : informal_econ,
            'GDP_Change_%'                    : gdp_change,
            'Inflation_Rate_%'                : inflation,
            'Currency_Devaluation_%'          : currency_deval,
            'Food_Insecurity_Rate_%'          : food_insecurity,
            'Youth_Unemployment_Change_%'     : youth_unemp,
            'Currency_Black_Market_Rate_Gap_%': bm_gap,
            'Conflict_Type_enc'               : conflict_enc,
            'Region_enc'                      : region_enc,
            'Status_enc'                      : status_enc,
            'Black_Market_Activity_Level_enc' : bm_enc,
            'Most_Affected_Sector_enc'        : sector_enc,
            'War_Profiteering_Documented_enc' : profit_enc,
        }

        input_df     = pd.DataFrame([input_dict])[features]
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]

        # Display result
        r1, r2, r3 = st.columns(3)
        r1.metric("🎯 Predicted Unemployment", f"{prediction:.2f}%")
        r2.metric("📊 Pre-War Avg (Dataset)",  f"{df['Pre_War_Unemployment_%'].mean():.2f}%")
        r3.metric("📈 Predicted Spike",        f"+{prediction - df['Pre_War_Unemployment_%'].mean():.2f}%")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prediction,
            delta = {'reference': df['Pre_War_Unemployment_%'].mean(), 'increasing': {'color': 'red'}},
            title = {'text': "Predicted During-War Unemployment %", 'font': {'color': 'white'}},
            gauge = {
                'axis'  : {'range': [0, 90], 'tickcolor': 'white'},
                'bar'   : {'color': '#e63946'},
                'steps' : [
                    {'range': [0,  20], 'color': '#2a9d8f'},
                    {'range': [20, 50], 'color': '#f4a261'},
                    {'range': [50, 90], 'color': '#e63946'},
                ],
                'threshold': {
                    'line' : {'color': 'white', 'width': 4},
                    'thickness': 0.75,
                    'value': df['During_War_Unemployment_%'].mean()
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        # Severity message
        if prediction < 20:
            st.success(f"🟢 LOW IMPACT — Predicted unemployment {prediction:.1f}% is below average. Economy relatively stable.")
        elif prediction < 40:
            st.warning(f"🟡 MODERATE IMPACT — Predicted unemployment {prediction:.1f}%. Significant disruption expected.")
        else:
            st.error(f"🔴 SEVERE IMPACT — Predicted unemployment {prediction:.1f}%. Extreme economic crisis expected.")

        st.info("ℹ️ Model: Gradient Boosting Regressor  |  R²=0.94  |  MAE=2.72%  |  Trained on 80,000 records")
