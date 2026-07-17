import streamlit as st
import numpy as np
import pandas as pd
import joblib

lasso        = joblib.load("model_lasso.pkl")
ridge        = joblib.load("model_ridge.pkl")
gbr          = joblib.load("model_gbr.pkl")
xgb_model    = joblib.load("model_xgb.pkl")
scaler       = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=1600");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.55);
        z-index: 0;
    }
    .block-container { position: relative; z-index: 1; }
    .main-title {
        text-align: center; font-size: 3.5rem; font-weight: 900;
        color: white; text-shadow: 0 0 30px rgba(233,69,96,0.8); padding: 1rem 0;
    }
    .subtitle { text-align: center; color: #a0aec0; font-size: 1.1rem; margin-bottom: 2rem; }
    .section-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px; padding: 1.5rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #e94560;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(233,69,96,0.4); padding-bottom: 0.5rem;
    }
    label, p, span { color: white !important; }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #c0392b) !important;
        color: white !important; font-size: 1.4rem !important;
        font-weight: 800 !important; border: none !important;
        border-radius: 14px !important; padding: 1rem 2rem !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(233,69,96,0.5) !important;
    }
    .stButton > button:hover { transform: translateY(-3px) !important; }
    .result-box {
        background: linear-gradient(135deg, rgba(233,69,96,0.3), rgba(15,52,96,0.5));
        border: 2px solid #e94560; border-radius: 20px;
        padding: 2.5rem; text-align: center;
        box-shadow: 0 0 40px rgba(233,69,96,0.3); margin-top: 1.5rem;
    }
    .result-price { font-size: 3.5rem; font-weight: 900; color: #e94560 !important; }
    .model-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px; padding: 1rem; text-align: center;
    }
    .model-name { color: #a0aec0 !important; font-size: 0.85rem; }
    .model-val  { color: white !important; font-size: 1.3rem; font-weight: 700; }
    hr { border-color: rgba(255,255,255,0.15) !important; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter house details below to get an AI-powered price estimate</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card"><div class="section-header">📐 Size & Space</div>', unsafe_allow_html=True)
    GrLivArea    = st.slider("Above Ground Living Area (sqft)", 500, 5000, 1500)
    TotalBsmtSF  = st.slider("Total Basement Area (sqft)", 0, 3000, 800)
    FirstFlrSF   = st.slider("1st Floor Area (sqft)", 300, 4000, 1000)
    SecondFlrSF  = st.slider("2nd Floor Area (sqft)", 0, 2000, 0)
    GarageArea   = st.slider("Garage Area (sqft)", 0, 1500, 400)
    TotalPorchSF = st.slider("Total Porch Area (sqft)", 0, 600, 50)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card"><div class="section-header">⭐ Quality & Condition</div>', unsafe_allow_html=True)
    OverallQual  = st.selectbox("Overall Quality (1-10)", list(range(1, 11)), index=4)
    OverallCond  = st.selectbox("Overall Condition (1-10)", list(range(1, 11)), index=4)
    ExterQual    = st.selectbox("Exterior Quality", [0,1,2,3,4,5],
                                format_func=lambda x: ['None','Po','Fa','TA','Gd','Ex'][x], index=3)
    KitchenQual  = st.selectbox("Kitchen Quality", [0,1,2,3,4,5],
                                format_func=lambda x: ['None','Po','Fa','TA','Gd','Ex'][x], index=3)
    BsmtQual     = st.selectbox("Basement Quality", [0,1,2,3,4,5],
                                format_func=lambda x: ['None','Po','Fa','TA','Gd','Ex'][x], index=3)
    GarageFinish = st.selectbox("Garage Finish", [0,1,2,3],
                                format_func=lambda x: ['None','Unfinished','Rough Fin','Finished'][x], index=2)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card"><div class="section-header">🏗️ Structure & Age</div>', unsafe_allow_html=True)
    YearBuilt    = st.slider("Year Built", 1900, 2010, 1990)
    YearRemodAdd = st.slider("Year Remodeled", 1900, 2010, 1995)
    YrSold       = st.selectbox("Year Sold", [2006,2007,2008,2009,2010], index=4)
    GarageCars   = st.selectbox("Garage Car Capacity", [0,1,2,3,4], index=2)
    TotalBaths   = st.selectbox("Total Bathrooms", [1.0,1.5,2.0,2.5,3.0,3.5], index=2)
    Fireplaces   = st.selectbox("Fireplaces", [0,1,2,3], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

def build_input():
    row = pd.DataFrame(0, index=[0], columns=feature_cols)
    row['GrLivArea']    = GrLivArea
    row['TotalBsmtSF']  = TotalBsmtSF
    row['1stFlrSF']     = FirstFlrSF
    row['2ndFlrSF']     = SecondFlrSF
    row['GarageArea']   = GarageArea
    row['OverallQual']  = OverallQual
    row['OverallCond']  = OverallCond
    row['ExterQual']    = ExterQual
    row['KitchenQual']  = KitchenQual
    row['BsmtQual']     = BsmtQual
    row['GarageFinish'] = GarageFinish
    row['GarageCars']   = GarageCars
    row['TotalBaths']   = TotalBaths
    row['Fireplaces']   = Fireplaces
    row['TotalPorchSF'] = TotalPorchSF
    row['TotalSF']      = TotalBsmtSF + FirstFlrSF + SecondFlrSF
    row['HouseAge']     = YrSold - YearBuilt
    row['RemodAge']     = YrSold - YearRemodAdd
    row['IsRemodeled']  = int(YearBuilt != YearRemodAdd)
    row['IsNew']        = int(YrSold == YearBuilt)
    row['HasGarage']    = int(GarageArea > 0)
    row['HasFireplace'] = int(Fireplaces > 0)
    row['HasBasement']  = int(TotalBsmtSF > 0)
    return row

st.markdown("---")
if st.button("🔮 Predict Sale Price"):
    input_df = build_input()
    scaled   = scaler.transform(input_df)
    p_lasso  = np.expm1(lasso.predict(scaled)[0])
    p_ridge  = np.expm1(ridge.predict(scaled)[0])
    p_gbr    = np.expm1(gbr.predict(scaled)[0])
    p_xgb    = np.expm1(xgb_model.predict(scaled)[0])
    final = (0.40*p_lasso + 0.30*p_gbr + 0.30*p_xgb)

    st.markdown(f"""
    <div class="result-box">
        <div style="color:#a0aec0; margin-bottom:0.5rem;">🏡 Estimated Sale Price</div>
        <div class="result-price">${final:,.0f}</div>
        <div style="color:#a0aec0; margin-top:0.5rem;">Blended ML model prediction</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, name, val in zip([c1,c2,c3,c4],
                               ["Lasso","Ridge","GBR","XGBoost"],
                               [p_lasso, p_ridge, p_gbr, p_xgb]):
        col.markdown(f"""
        <div class="model-card">
            <div class="model-name">{name}</div>
            <div class="model-val">${val:,.0f}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center;color:#4a5568;">Trained on Ames Housing Dataset | Kaggle Competition</p>', unsafe_allow_html=True)