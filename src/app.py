from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os
import numpy as np
from .pricing import quote

app = FastAPI()

MODEL_DIR = os.getenv('MODEL_DIR', 'models')
model = joblib.load(os.path.join(MODEL_DIR, 'risk_model.joblib'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))

FEATURES = ['miles','night_miles','mean_speed','max_speed_over_limit', 'accel_events_per_100mi','brake_events_per_100mi', 'phone_use_ratio','rain_flag','incident_density']



class TripFeatures(BaseModel):
    miles: float
    night_miles: float
    mean_speed: float
    max_speed_over_limit: float
    accel_events_per_100mi: float
    brake_events_per_100mi: float
    phone_use_ratio: float
    rain_flag: int
    incident_density: float

@app.get('/health')
def health():
    return {"status": "ok"}

@app.post('/score')
def score(f: TripFeatures):
    x = np.array([[getattr(f, k) for k in FEATURES]])
    xs = scaler.transform(x)
    p = float(model.predict_proba(xs)[0, 1])
    score100 = int(round((1.0 - p) * 100))
    return {'risk_probability': p, 'safe_score': score100}

@app.post('/quote')
def quote_from_features(f: TripFeatures):
    x = np.array([[getattr(f, k) for k in FEATURES]])
    xs = scaler.transform(x)
    p = float(model.predict_proba(xs)[0, 1])
    res = quote(1.0 - p)
    res.update({'risk_probability': p, 'safe_score': int(round((1.0 - p) * 100))})
    return res

