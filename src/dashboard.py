import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import requests

st.set_page_config(page_title='UBI PoC Dashboard', layout='wide')
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
model = joblib.load(os.path.join(MODEL_DIR, 'risk_model.joblib'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
meta = json.load(open(os.path.join(MODEL_DIR, 'metadata.json')))
FEATURES = meta['features']

df = pd.read_csv('data/trips.csv')
X = scaler.transform(df[FEATURES].values)
p = model.predict_proba(X)[:,1]
df['risk_probability'] = p
df['safe_score'] = np.round((1.0 - p) * 100).astype(int)

st.title('Usage-Based Insurance - Telematics PoC')

st.metric('Validation AUC', f"{meta['val_auc']:.4f}")

st.subheader('Trip Explorer')
st.dataframe(df.head(500))

st.subheader('Premium Estimator')
cols = st.columns(3)
row = df.sample(1, random_state=1).iloc[0]
with st.form('quote'):
    vals = {}
    for k in FEATURES:
        vals[k] = st.number_input(k, value=float(row[k]))
    submitted = st.form_submit_button('Score & Quote')

if submitted:
    payload = {k: float(vals[k]) for k in FEATURES}
    try:
        r = requests.post('http://localhost:8000/quote', json=payload, timeout=2)
        st.write(r.json())
    except Exception:
        X1 = scaler.transform(np.array([[payload[k] for k in FEATURES]]))
        p1 = float(model.predict_proba(X1)[0, 1])
        safe = int(round((1.0 - p1) * 100))
        mult = 0.75 + 0.5*(1.0 - (1.0 - p1))
        prem = 1000.0 * np.clip(mult, 0.65, 1.35)
        st.write({'risk_probability': p1, 'safe_score': safe, 'annual_premium': float(np.round(prem, 2))})

