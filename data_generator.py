"""
Module 1: Synthetic Energy Demand Data Generator
Generates realistic hourly electricity demand data with:
  - Seasonal patterns (winter/summer peaks)
  - Daily patterns (morning/evening peaks)
  - Weekend vs weekday differences
  - Weather influence (temperature)
  - Random anomalies (spikes/drops)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def get_season(month):
    """Return season name from month number."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def get_season_code(month):
    """Return numeric season code (0=Spring, 1=Summer, 2=Autumn, 3=Winter)."""
    seasons = {12: 3, 1: 3, 2: 3, 3: 0, 4: 0, 5: 0,
               6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2}
    return seasons[month]


def simulate_temperature(month, hour):
    """Simulate hourly temperature based on month and time of day."""
    # Base temperature by month (India-like climate)
    monthly_base = {1: 12, 2: 15, 3: 22, 4: 30, 5: 36,
                    6: 33, 7: 28, 8: 27, 9: 28, 10: 24, 11: 18, 12: 13}
    base = monthly_base[month]
    # Diurnal cycle: cooler at night, warmer afternoon
    diurnal = 6 * np.sin(np.pi * (hour - 6) / 12)
    noise = np.random.normal(0, 1.5)
    return round(base + diurnal + noise, 1)


def generate_demand(timestamp, temperature, is_holiday=False):
    """
    Generate realistic electricity demand (MW) for a given timestamp.
    Incorporates: time-of-day, day-of-week, seasonality, temperature, holidays.
    """
    hour = timestamp.hour
    month = timestamp.month
    weekday = timestamp.weekday()  # 0=Monday, 6=Sunday

    # Base demand (MW)
    base = 3000

    # --- Hourly pattern ---
    # Morning ramp-up (6-9 AM), midday, evening peak (18-21), overnight dip
    if 0 <= hour < 5:
        hour_factor = 0.65
    elif 5 <= hour < 9:
        hour_factor = 0.65 + 0.35 * (hour - 5) / 4
    elif 9 <= hour < 12:
        hour_factor = 0.95 + 0.05 * np.sin(np.pi * (hour - 9) / 3)
    elif 12 <= hour < 14:
        hour_factor = 0.90  # lunch dip
    elif 14 <= hour < 18:
        hour_factor = 0.93 + 0.07 * (hour - 14) / 4
    elif 18 <= hour < 21:
        hour_factor = 1.10  # evening peak
    else:
        hour_factor = 1.10 - 0.45 * (hour - 21) / 3

    # --- Day-of-week factor ---
    if weekday >= 5:  # Weekend
        day_factor = 0.85
    elif is_holiday:
        day_factor = 0.78
    else:
        day_factor = 1.0

    # --- Seasonal / monthly factor ---
    monthly_factor = {
        1: 1.20, 2: 1.15, 3: 1.05, 4: 1.10,
        5: 1.25, 6: 1.30, 7: 1.35, 8: 1.30,
        9: 1.15, 10: 1.05, 11: 1.10, 12: 1.18
    }[month]

    # --- Temperature effect (cooling/heating demand) ---
    # Comfort zone: 18-24 C. Heating below, cooling above.
    if temperature < 18:
        temp_factor = 1 + 0.015 * (18 - temperature)
    elif temperature > 24:
        temp_factor = 1 + 0.020 * (temperature - 24)
    else:
        temp_factor = 1.0

    demand = base * hour_factor * day_factor * monthly_factor * temp_factor
    # Add random noise
    demand += np.random.normal(0, demand * 0.02)

    return max(demand, 500)


def generate_dataset(start_date='2022-01-01', end_date='2024-12-31',
                     anomaly_rate=0.005, seed=42):
    """
    Generate full synthetic dataset with timestamps, demand, weather, and labels.

    Parameters
    ----------
    start_date : str
    end_date   : str
    anomaly_rate : float  Fraction of points that are anomalies
    seed       : int

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, demand_mw, temperature_c, is_weekend, is_holiday,
        hour, day_of_week, month, season, season_code,
        hour_sin, hour_cos, month_sin, month_cos,
        is_anomaly, anomaly_type
    """
    np.random.seed(seed)

    # Indian public holidays (simplified)
    holidays = set([
        '2022-01-26', '2022-03-18', '2022-08-15', '2022-10-05',
        '2022-11-08', '2022-12-25',
        '2023-01-26', '2023-03-08', '2023-08-15', '2023-10-24',
        '2023-11-27', '2023-12-25',
        '2024-01-26', '2024-03-25', '2024-08-15', '2024-10-12',
        '2024-11-15', '2024-12-25',
    ])

    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    records = []

    for ts in timestamps:
        date_str = ts.strftime('%Y-%m-%d')
        is_holiday = date_str in holidays
        temp = simulate_temperature(ts.month, ts.hour)
        demand = generate_demand(ts, temp, is_holiday)
        records.append({
            'timestamp': ts,
            'demand_mw': round(demand, 2),
            'temperature_c': temp,
            'is_weekend': int(ts.weekday() >= 5),
            'is_holiday': int(is_holiday),
        })

    df = pd.DataFrame(records)

    # ── Temporal feature extraction ──────────────────────────────────────
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek   # 0=Mon
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['week_of_year']= df['timestamp'].dt.isocalendar().week.astype(int)
    df['month']       = df['timestamp'].dt.month
    df['year']        = df['timestamp'].dt.year
    df['season']      = df['month'].apply(get_season)
    df['season_code'] = df['month'].apply(get_season_code)

    # Cyclical encoding — preserves circular nature of time features
    df['hour_sin']    = np.sin(2 * np.pi * df['hour']        / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour']        / 24)
    df['dow_sin']     = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']     = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin']   = np.sin(2 * np.pi * df['month']       / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month']       / 12)
    df['doy_sin']     = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos']     = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # ── Lagged demand features ───────────────────────────────────────────
    df['lag_1h']      = df['demand_mw'].shift(1)
    df['lag_24h']     = df['demand_mw'].shift(24)
    df['lag_168h']    = df['demand_mw'].shift(168)  # same hour last week
    df['rolling_24h'] = df['demand_mw'].shift(1).rolling(24).mean()
    df['rolling_7d']  = df['demand_mw'].shift(1).rolling(168).mean()

    # ── Inject anomalies ─────────────────────────────────────────────────
    n_anomalies = int(len(df) * anomaly_rate)
    anomaly_idx = np.random.choice(df.index[200:], size=n_anomalies, replace=False)

    df['is_anomaly']   = 0
    df['anomaly_type'] = 'normal'

    spike_idx = anomaly_idx[:n_anomalies // 2]
    drop_idx  = anomaly_idx[n_anomalies // 2:]

    df.loc[spike_idx, 'demand_mw']    *= np.random.uniform(1.4, 2.0, len(spike_idx))
    df.loc[spike_idx, 'is_anomaly']   = 1
    df.loc[spike_idx, 'anomaly_type'] = 'spike'

    df.loc[drop_idx, 'demand_mw']    *= np.random.uniform(0.2, 0.5, len(drop_idx))
    df.loc[drop_idx, 'is_anomaly']   = 1
    df.loc[drop_idx, 'anomaly_type'] = 'drop'

    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = generate_dataset()
    df.to_csv('/home/claude/smart_energy_forecasting/energy_data.csv', index=False)
    print(f"Dataset generated: {df.shape}")
    print(df[['timestamp', 'demand_mw', 'temperature_c', 'hour', 'season',
              'is_anomaly', 'anomaly_type']].head(10).to_string())
