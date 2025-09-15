import argparse, json
import pandas as pd
import numpy as np
import joblib, os
from sklearn.metrics import roc_auc_score, brier_score_loss


parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--model_dir', default='models')
args = parser.parse_args()

meta = json.load(open(os.path.join(args.model_dir, 'metadata.json')))
cols = meta['features']

model = joblib.load(os.path.join(args.model_dir, 'risk_model.joblib'))
scaler = joblib.load(os.path.join(args.model_dir, 'scaler.joblib'))

df = pd.read_csv(args.data)
X = scaler.transform(df[cols].values)
y = df['label'].values
p = model.predict_proba(X)[:,1]

auc = roc_auc_score(y, p)
brier = brier_score_loss(y, p)

print(f'AUC: {auc:.4f}, Brier: {brier:.4f}')

df_eval = pd.DataFrame({'y': y, 'p': p})
quant = pd.qcut(df_eval['p'], 10, duplicates='drop')
calib = df_eval.groupby(quant).agg(obs_rate=('y', 'mean'), pred_mean=('p', 'mean'), n=('y', 'size'))
print('\nCalibration by decile:', calib)
