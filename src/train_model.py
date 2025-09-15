import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--out', default='models')
args = parser.parse_args()

cols = ['miles', 'night_miles','mean_speed','max_speed_over_limit', 'accel_events_per_100mi','brake_events_per_100mi', 'phone_use_ratio','rain_flag','incident_density']

df = pd.read_csv(args.data)
X = df[cols].values
y = df['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

base = GradientBoostingClassifier(random_state=7)
cal = CalibratedClassifierCV(base, method='sigmoid', cv=3)
cal.fit(X_train_s, y_train)

val_proba = cal.predict_proba(X_val_s)[:,1]
auc = roc_auc_score(y_val, val_proba)
print(f'Validation AUC: {auc}')

os.makedirs(args.out, exist_ok=True)
joblib.dump(cal, os.path.join(args.out, 'risk_model.joblib'))
joblib.dump(scaler, os.path.join(args.out, 'scaler.joblib'))

metadata = {'features': cols, 'val_auc': float(auc)}

with open(os.path.join(args.out, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved model to {args.out}")

