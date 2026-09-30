import pandas as pd
import numpy as np

# Load the real results
df = pd.read_csv("output/flagged_events_features.csv", parse_dates=['start_time', 'end_time', 'peak_time'])

# Convert string timestamps to datetime
for col in ['start_time', 'end_time', 'peak_time']:
    df[col] = pd.to_datetime(df[col], format='ISO8601')

# Calculate event duration in seconds
df['duration_s'] = (df['end_time'] - df['start_time']).dt.total_seconds()

# Calculate summary stats per cluster
summary = df.groupby('cluster_id').agg(
    count=('event_id', 'count'),
    mean_duration_s=('duration_s', 'mean'),
    mean_peak_voltage=('peak_voltage', lambda x: np.mean(np.abs(x))),
    mean_spectral_centroid=('spectral_centroid_mean', 'mean'),
    mean_rms=('rms_mean', 'mean'),
    mean_zcr=('zcr_mean', 'mean')
).round(4)

print("Cluster Profiles:")
print("-" * 60)
print(summary.to_string())
print("-" * 60)
