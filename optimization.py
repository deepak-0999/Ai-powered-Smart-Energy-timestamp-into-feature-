"""
Module 4: Energy Distribution Optimization
Rule-based + greedy Model Predictive Control (MPC) for:
  - Battery charge/discharge scheduling
  - Demand response activation
  - Renewable dispatch optimization
  - Peak shaving
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BatteryConfig:
    """Battery storage system configuration."""
    capacity_mwh: float    = 500.0    # Total storage capacity
    max_charge_mw: float   = 100.0    # Max charge rate
    max_discharge_mw: float = 120.0  # Max discharge rate
    efficiency: float      = 0.92     # Round-trip efficiency
    min_soc: float         = 0.10     # Minimum state of charge (10%)
    max_soc: float         = 0.95     # Maximum state of charge (95%)
    initial_soc: float     = 0.50     # Initial state of charge


@dataclass
class GridConfig:
    """Grid and tariff configuration."""
    base_supply_mw: float      = 4000.0   # Baseline grid supply
    renewable_capacity_mw: float = 800.0  # Solar + wind installed
    peak_threshold_mw: float   = 3800.0  # Demand above this = peak
    # Time-of-use tariff (Rs/MWh per hour category)
    off_peak_tariff: float     = 2000.0
    mid_peak_tariff: float     = 3500.0
    peak_tariff: float         = 6000.0
    demand_response_reward: float = 1500.0  # Rs/MWh shed


def get_tariff_period(hour: int) -> str:
    """Return tariff period for a given hour."""
    if 0 <= hour < 6:
        return 'off_peak'
    elif 6 <= hour < 9 or 14 <= hour < 17:
        return 'mid_peak'
    elif 9 <= hour < 14 or 17 <= hour < 22:
        return 'peak'
    else:
        return 'off_peak'


def renewable_availability(hour: int, month: int, season: str) -> float:
    """
    Estimate renewable generation fraction (0–1) based on time and season.
    Combines solar (daytime) and wind (evening/night).
    """
    # Solar profile
    if 6 <= hour <= 18:
        solar = np.sin(np.pi * (hour - 6) / 12) * 0.85
    else:
        solar = 0.0

    # Seasonal solar scaling
    solar_seasonal = {'Winter': 0.65, 'Spring': 0.85,
                      'Summer': 1.0,  'Autumn': 0.80}.get(season, 0.80)
    solar *= solar_seasonal

    # Wind profile (stronger at night and shoulders)
    if hour < 8 or hour > 19:
        wind = 0.60
    elif 8 <= hour <= 11 or 17 <= hour <= 19:
        wind = 0.40
    else:
        wind = 0.25

    # Combined (60% solar, 40% wind of total renewable capacity)
    return min(0.60 * solar + 0.40 * wind, 1.0)


class EnergyOptimizer:
    """
    Greedy MPC-style optimizer for one forecast horizon.

    At each hour:
      1. Compute renewable generation
      2. Compute net demand = forecast - renewable
      3. Decide battery action (charge / discharge / idle)
      4. Activate demand response if still in peak and demand too high
      5. Calculate cost and record all decisions
    """

    def __init__(self, battery: BatteryConfig = None, grid: GridConfig = None):
        self.battery = battery or BatteryConfig()
        self.grid    = grid    or GridConfig()

    def optimize_horizon(self, forecast_df: pd.DataFrame,
                         anomaly_flags: np.ndarray = None) -> pd.DataFrame:
        """
        Run optimization over forecast horizon.

        Parameters
        ----------
        forecast_df  : DataFrame with columns timestamp, forecast_mw,
                       lower_mw, upper_mw, hour, month, season
        anomaly_flags: 1=anomaly at that timestep; use conservative dispatch

        Returns
        -------
        DataFrame with optimization decisions and costs per hour.
        """
        if anomaly_flags is None:
            anomaly_flags = np.zeros(len(forecast_df), dtype=int)

        soc = self.battery.initial_soc       # State of charge (fraction)
        records = []

        for i, row in forecast_df.iterrows():
            hour   = int(row['hour'])
            month  = int(row['month'])
            season = row.get('season', 'Summer')
            is_anomaly = bool(anomaly_flags[i] if i < len(anomaly_flags) else 0)

            # Use upper quantile during anomalies for conservative planning
            demand = row['upper_mw'] if is_anomaly else row['forecast_mw']
            demand = max(demand, 0)

            # ── Renewable generation ──────────────────────────────────────
            renew_frac   = renewable_availability(hour, month, season)
            renewable_mw = renew_frac * self.grid.renewable_capacity_mw
            net_demand   = max(demand - renewable_mw, 0)

            # ── Tariff ───────────────────────────────────────────────────
            tariff_period = get_tariff_period(hour)
            tariff_map = {
                'off_peak': self.grid.off_peak_tariff,
                'mid_peak': self.grid.mid_peak_tariff,
                'peak':     self.grid.peak_tariff,
            }
            tariff = tariff_map[tariff_period]

            # ── Battery dispatch ──────────────────────────────────────────
            battery_mw     = 0.0
            battery_action = 'idle'
            soc_kwh_avail  = (soc - self.battery.min_soc) * self.battery.capacity_mwh

            if tariff_period == 'off_peak' and soc < self.battery.max_soc:
                # Charge battery during off-peak hours
                headroom   = (self.battery.max_soc - soc) * self.battery.capacity_mwh
                charge_mw  = min(self.battery.max_charge_mw, headroom)
                battery_mw = -charge_mw                 # negative = charging
                soc        = soc + (charge_mw * self.battery.efficiency
                                    / self.battery.capacity_mwh)
                battery_action = 'charging'

            elif tariff_period == 'peak' and net_demand > self.grid.peak_threshold_mw * 0.85:
                # Discharge during peak if net demand is high
                needed     = net_demand - self.grid.peak_threshold_mw * 0.85
                discharge  = min(self.battery.max_discharge_mw,
                                 needed, soc_kwh_avail)
                battery_mw = discharge
                soc        = soc - discharge / self.battery.capacity_mwh
                battery_action = 'discharging'

            soc = np.clip(soc, self.battery.min_soc, self.battery.max_soc)

            # ── Demand response ───────────────────────────────────────────
            grid_demand   = net_demand - battery_mw
            dr_shed_mw    = 0.0
            dr_activated  = False

            if tariff_period == 'peak' and grid_demand > self.grid.peak_threshold_mw:
                # Shed up to 5% of demand via demand-response
                dr_shed_mw   = min(grid_demand * 0.05,
                                   grid_demand - self.grid.peak_threshold_mw)
                grid_demand  -= dr_shed_mw
                dr_activated = True

            # ── Cost calculation ──────────────────────────────────────────
            procurement_cost = max(grid_demand, 0) * tariff / 1000   # Rs (MWh * Rs/MWh / 1000)
            dr_reward        = dr_shed_mw * self.grid.demand_response_reward / 1000
            net_cost         = procurement_cost - dr_reward

            records.append({
                'timestamp':      row.get('timestamp', i),
                'forecast_mw':    round(demand, 1),
                'renewable_mw':   round(renewable_mw, 1),
                'net_demand_mw':  round(net_demand, 1),
                'battery_mw':     round(battery_mw, 1),
                'battery_action': battery_action,
                'soc_pct':        round(soc * 100, 1),
                'dr_shed_mw':     round(dr_shed_mw, 1),
                'dr_activated':   int(dr_activated),
                'grid_supply_mw': round(max(grid_demand, 0), 1),
                'tariff_period':  tariff_period,
                'tariff_rs_mwh':  tariff,
                'procurement_cost_rs': round(procurement_cost, 0),
                'dr_reward_rs':   round(dr_reward, 0),
                'net_cost_rs':    round(net_cost, 0),
                'is_anomaly':     int(is_anomaly),
            })

        return pd.DataFrame(records)

    def summary_report(self, opt_df: pd.DataFrame) -> Dict:
        """Summarise optimization outcomes."""
        total_hours  = len(opt_df)
        peak_hours   = (opt_df['tariff_period'] == 'peak').sum()
        dr_events    = opt_df['dr_activated'].sum()
        total_cost   = opt_df['net_cost_rs'].sum()
        total_renew  = opt_df['renewable_mw'].sum()
        total_demand = opt_df['forecast_mw'].sum()
        renewable_pct = (total_renew / total_demand * 100) if total_demand > 0 else 0

        discharge_hours = (opt_df['battery_action'] == 'discharging').sum()
        charge_hours    = (opt_df['battery_action'] == 'charging').sum()

        summary = {
            'total_hours':          total_hours,
            'peak_hours':           peak_hours,
            'dr_events':            int(dr_events),
            'total_cost_rs':        round(total_cost, 0),
            'avg_cost_per_hr_rs':   round(total_cost / total_hours, 0),
            'renewable_coverage_%': round(renewable_pct, 1),
            'battery_charge_hours': int(charge_hours),
            'battery_discharge_hrs':int(discharge_hours),
            'avg_soc_%':            round(opt_df['soc_pct'].mean(), 1),
        }
        return summary


if __name__ == '__main__':
    from data_generator import generate_dataset
    from forecasting_model import EnergyForecaster

    df = generate_dataset()
    df = df.dropna().reset_index(drop=True)

    split = -7 * 24    # Use last 7 days for optimization demo
    df_test = df.iloc[split:].copy()

    forecaster = EnergyForecaster(n_estimators=100)
    forecaster.fit(df.iloc[:split])
    preds = forecaster.predict(df_test)

    forecast_df = df_test[['timestamp', 'hour', 'month', 'season']].copy()
    forecast_df['forecast_mw'] = preds['forecast_mw'].values
    forecast_df['lower_mw']    = preds['lower_mw'].values
    forecast_df['upper_mw']    = preds['upper_mw'].values

    optimizer = EnergyOptimizer()
    opt_result = optimizer.optimize_horizon(forecast_df)

    print("=== Optimization Results (first 24h) ===")
    cols = ['timestamp', 'forecast_mw', 'renewable_mw', 'battery_mw',
            'battery_action', 'soc_pct', 'dr_shed_mw', 'net_cost_rs']
    print(opt_result[cols].head(24).to_string(index=False))

    print("\n=== 7-Day Summary ===")
    summary = optimizer.summary_report(opt_result)
    for k, v in summary.items():
        print(f"  {k}: {v}")
