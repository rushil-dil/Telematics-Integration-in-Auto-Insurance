import argparse, os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--rows', type=int, default=20000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out', type=str, default='data/trips.csv')
args = parser.parse_args()

rs = np.random.RandomState(args.seed)

n = args.rows
now = datetime.now()

drivers = [f"D{d: 04d}" for d in range(200)]

rows = []

for i in range(n):
    driver = rs.choice(drivers)
    start = now - timedelta(days=rs.randint(1, 60), minutes=rs.randint(0, 1440))
    dur_min = max(5, int(rs.normal(28, 12)))
    end = start + timedelta(minutes=dur_min)
    miles = max(0.5, np.round(rs.gamma(3, 1.2), 2))
    night = 1 if start.hour >= 22 or start.hour < 5 else 0
    night_mi = np.round(miles * (rs.beta(2, 8) if night else rs.beta(1, 12)), 2)
    mean_speed = np.clip(rs.normal(34, 8), 5, 80)
    max_over = max(0.0, np.round(rs.gamma(1.8, 4) - 3, 1))
    accel = np.round(np.clip(rs.normal(8, 5), 0, 60), 1)
    brake = np.round(np.clip(rs.normal(10, 6), 0, 80), 1)
    phone = np.round(np.clip(rs.beta(2, 10), 0, 1), 3)
    rain = int(rs.rand() < 0.18)
    incident = np.round(np.clip(rs.normal(0.6 if rain else 0.4, 0.2), 0, 1.5), 3)

    # Syntheic risk score (latent), then Bernoulli label
    risk_latent = (0.04*max_over + 0.03*accel + 0.035*brake + 0.5*phone + 0.02*night_mi + 0.4*rain + 0.3*incident + rs.normal(0, 0.3))

    p = 1/(1 + np.exp(-risk_latent + 2.5)) # shift for class balance
    label = int(rs.rand() < p)

    rows.append({'driver_id': driver, 'trip_id': f'T{i:06d}', 'start_ts': start.isoformat(), 'end_ts': end.isoformat(), 'miles': miles, 'night_miles': night_mi, 'mean_speed': np.round(mean_speed, 1), 'max_speed_over_limit': max_over, 'accel_events_per_100mi': accel, 'brake_events_per_100mi': brake, 'phone_use_ratio': phone, 'rain_flag': rain, 'incident_density': incident, 'label': label,})

os.makedirs('data', exist_ok=True)
pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"Wrote {args.out} with {n} rows")
