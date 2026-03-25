"""
Module 5: Main Pipeline
Runs the full AI-Powered Smart Energy Demand Forecasting & Optimization system:
  1. Generate / load data
  2. Feature engineering (temporal features)
  3. Train forecasting model + cross-validate
  4. Train anomaly detector + evaluate
  5. Run optimization on test horizon
  6. Visualize and save all results
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Make local imports work regardless of cwd ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_generator  import generate_dataset
from forecasting_model import (EnergyForecaster, ExponentialSmoothingBaseline,
                                evaluate_forecast, walk_forward_validate,
                                FEATURE_COLS)
from anomaly_detection import EnsembleAnomalyDetector
from optimization      import EnergyOptimizer, BatteryConfig, GridConfig

OUTPUT_DIR = '/home/claude/smart_energy_forecasting'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(" STEP 1: Generating Synthetic Dataset")
print("=" * 65)

df = generate_dataset(start_date='2022-01-01', end_date='2024-12-31',
                      anomaly_rate=0.008, seed=42)
df = df.dropna().reset_index(drop=True)
df.to_csv(f'{OUTPUT_DIR}/energy_data.csv', index=False)
print(f"  Total records : {len(df):,}")
print(f"  Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  Anomalies     : {df['is_anomaly'].sum():,} "
      f"({df['is_anomaly'].mean()*100:.2f}%)")
print(f"  Demand (MW)   : min={df['demand_mw'].min():.0f}  "
      f"mean={df['demand_mw'].mean():.0f}  max={df['demand_mw'].max():.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 2: Train / Test Split")
print("=" * 65)

split = -60 * 24     # last 60 days for testing
df_train = df.iloc[:split].copy()
df_test  = df.iloc[split:].copy()
print(f"  Train: {len(df_train):,} hours  "
      f"({df_train['timestamp'].min().date()} → {df_train['timestamp'].max().date()})")
print(f"  Test : {len(df_test):,}  hours  "
      f"({df_test['timestamp'].min().date()} → {df_test['timestamp'].max().date()})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 3: Walk-Forward Cross Validation")
print("=" * 65)
cv_results = walk_forward_validate(df_train, n_splits=5)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN FINAL FORECASTER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 4: Training Final Forecasting Model")
print("=" * 65)
forecaster = EnergyForecaster(n_estimators=300, max_depth=5, learning_rate=0.04)
forecaster.fit(df_train)

# Also train baseline
baseline = ExponentialSmoothingBaseline(alpha=0.3, seasonal_period=24)
baseline.fit(df_train['demand_mw'].values)

# Predict on test set
preds_test = forecaster.predict(df_test)
df_test = df_test.copy()
df_test['forecast_mw'] = preds_test['forecast_mw'].values
df_test['lower_mw']    = preds_test['lower_mw'].values
df_test['upper_mw']    = preds_test['upper_mw'].values

# Baseline predictions
baseline_preds = baseline.predict(
    steps=len(df_test),
    start_hour=int(df_test['hour'].iloc[0])
)
df_test['baseline_mw'] = baseline_preds

print("\n  Test-Set Metrics — GBM Forecaster:")
metrics_gbm = evaluate_forecast(df_test['demand_mw'].values,
                                df_test['forecast_mw'].values)
for k, v in metrics_gbm.items():
    print(f"    {k}: {v}")

print("\n  Test-Set Metrics — Exponential Smoothing Baseline:")
metrics_bl = evaluate_forecast(df_test['demand_mw'].values,
                               df_test['baseline_mw'].values)
for k, v in metrics_bl.items():
    print(f"    {k}: {v}")

print("\n  Top 10 Feature Importances:")
print(forecaster.feature_importance_.head(10).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 5: Anomaly Detection")
print("=" * 65)

# Compute residuals
preds_train = forecaster.predict(df_train)
df_train = df_train.copy()
df_train['forecast_mw'] = preds_train['forecast_mw'].values
df_train['residual']    = df_train['demand_mw'] - df_train['forecast_mw']
df_test['residual']     = df_test['demand_mw']  - df_test['forecast_mw']

detector = EnsembleAnomalyDetector(contamination=0.01, z_threshold=3.0,
                                   iqr_k=2.5, min_votes=2)
detector.fit(df_train, df_train['residual'])

print("\n  Evaluation on Test Set:")
anomaly_results = detector.evaluate(df_test, df_test['residual'],
                                    df_test['is_anomaly'].values)
df_test = pd.concat([df_test.reset_index(drop=True),
                     anomaly_results.reset_index(drop=True)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 6: Energy Distribution Optimization")
print("=" * 65)

battery = BatteryConfig(capacity_mwh=500, max_charge_mw=100,
                        max_discharge_mw=120, efficiency=0.92,
                        min_soc=0.10, max_soc=0.95, initial_soc=0.50)
grid    = GridConfig(base_supply_mw=4000, renewable_capacity_mw=800,
                     peak_threshold_mw=3800)

optimizer    = EnergyOptimizer(battery=battery, grid=grid)
forecast_opt = df_test[['timestamp', 'hour', 'month', 'season',
                         'forecast_mw', 'lower_mw', 'upper_mw']].copy()

opt_result = optimizer.optimize_horizon(
    forecast_opt,
    anomaly_flags=df_test['anomaly_pred'].values
)

print("\n  Optimization Summary:")
summary = optimizer.summary_report(opt_result)
for k, v in summary.items():
    print(f"    {k}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 7: Generating Visualizations")
print("=" * 65)

PALETTE = {
    'actual':    '#1f77b4',
    'forecast':  '#ff7f0e',
    'baseline':  '#9467bd',
    'anomaly':   '#d62728',
    'band':      '#ff7f0e',
    'renewable': '#2ca02c',
    'battery':   '#17becf',
    'dr':        '#bcbd22',
    'cost':      '#8c564b',
}

# ── Figure 1: Temporal Feature Analysis ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle('Temporal Feature Analysis — Energy Demand Patterns',
             fontsize=16, fontweight='bold', y=1.01)

# Hourly average demand
hourly = df.groupby('hour')['demand_mw'].mean()
axes[0, 0].bar(hourly.index, hourly.values, color=PALETTE['actual'], alpha=0.8)
axes[0, 0].set_title('Average Demand by Hour of Day')
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Demand (MW)')
axes[0, 0].set_xticks(range(0, 24, 3))
axes[0, 0].grid(axis='y', alpha=0.3)

# Day-of-week average
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow = df.groupby('day_of_week')['demand_mw'].mean()
colors_dow = [PALETTE['actual'] if i < 5 else PALETTE['anomaly']
              for i in range(7)]
axes[0, 1].bar(dow_labels, dow.values, color=colors_dow, alpha=0.85)
axes[0, 1].set_title('Average Demand by Day of Week')
axes[0, 1].set_ylabel('Demand (MW)')
axes[0, 1].grid(axis='y', alpha=0.3)

# Monthly average
monthly = df.groupby('month')['demand_mw'].mean()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
axes[0, 2].bar(month_labels, monthly.values, color=PALETTE['forecast'], alpha=0.85)
axes[0, 2].set_title('Average Demand by Month')
axes[0, 2].set_ylabel('Demand (MW)')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(axis='y', alpha=0.3)

# Seasonal boxplot
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
season_colors = ['#aec7e8', '#98df8a', '#ffbb78', '#ff9896']
season_data = [df[df['season'] == s]['demand_mw'].values for s in season_order]
bp = axes[1, 0].boxplot(season_data, labels=season_order, patch_artist=True)
for patch, color in zip(bp['boxes'], season_colors):
    patch.set_facecolor(color)
axes[1, 0].set_title('Demand Distribution by Season')
axes[1, 0].set_ylabel('Demand (MW)')
axes[1, 0].grid(axis='y', alpha=0.3)

# Cyclical encoding scatter
axes[1, 1].scatter(df['hour_sin'].values[::10],
                   df['hour_cos'].values[::10],
                   c=df['hour'].values[::10], cmap='hsv', s=6, alpha=0.5)
axes[1, 1].set_title('Cyclical Hour Encoding (sin vs cos)')
axes[1, 1].set_xlabel('sin(2π·hour/24)')
axes[1, 1].set_ylabel('cos(2π·hour/24)')
cb = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cb.set_label('Hour')

# Temperature vs demand scatter
sample = df.sample(2000, random_state=42)
sc = axes[1, 2].scatter(sample['temperature_c'], sample['demand_mw'],
                         c=sample['season_code'], cmap='Set1', s=6, alpha=0.5)
axes[1, 2].set_title('Temperature vs Demand (coloured by Season)')
axes[1, 2].set_xlabel('Temperature (°C)')
axes[1, 2].set_ylabel('Demand (MW)')
legend_elements = [Patch(facecolor=plt.cm.Set1(i/4), label=s)
                   for i, s in enumerate(['Spring','Summer','Autumn','Winter'])]
axes[1, 2].legend(handles=legend_elements, fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_temporal_feature_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_temporal_feature_analysis.png")


# ── Figure 2: Forecast vs Actual (7 days) ───────────────────────────────────
PLOT_HOURS = 7 * 24
plot_df    = df_test.iloc[:PLOT_HOURS].copy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
fig.suptitle('Demand Forecasting — Actual vs Predicted (7-Day Window)',
             fontsize=15, fontweight='bold')

xs = range(PLOT_HOURS)
ax1.plot(xs, plot_df['demand_mw'].values, color=PALETTE['actual'],
         lw=1.5, label='Actual Demand', alpha=0.9)
ax1.plot(xs, plot_df['forecast_mw'].values, color=PALETTE['forecast'],
         lw=1.5, label='GBM Forecast', linestyle='--')
ax1.plot(xs, plot_df['baseline_mw'].values, color=PALETTE['baseline'],
         lw=1.0, label='Exp. Smoothing Baseline', linestyle=':', alpha=0.7)
ax1.fill_between(xs, plot_df['lower_mw'].values, plot_df['upper_mw'].values,
                 alpha=0.18, color=PALETTE['band'], label='80% Prediction Interval')

# Shade weekends
for h in range(0, PLOT_HOURS, 24):
    if h // 24 < len(plot_df):
        if plot_df.iloc[h]['is_weekend']:
            ax1.axvspan(h, min(h + 24, PLOT_HOURS), alpha=0.07, color='gray')

ax1.set_ylabel('Demand (MW)')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(alpha=0.3)

# Residuals
residuals_plot = plot_df['demand_mw'].values - plot_df['forecast_mw'].values
ax2.bar(xs, residuals_plot, color=np.where(residuals_plot > 0, '#2ca02c', '#d62728'),
        alpha=0.7, width=1.0)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_ylabel('Residual (MW)')
ax2.set_xlabel('Hour')
ax2.grid(alpha=0.3)
ax2.set_xticks(range(0, PLOT_HOURS, 24))
ax2.set_xticklabels([plot_df['timestamp'].iloc[h].strftime('%a %b %d')
                      for h in range(0, PLOT_HOURS, 24)], rotation=30)

# Metric box
metric_text = '\n'.join([f"{k}: {v}" for k, v in metrics_gbm.items()])
ax1.text(0.01, 0.97, metric_text, transform=ax1.transAxes,
         verticalalignment='top', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_forecast_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_forecast_vs_actual.png")


# ── Figure 3: Anomaly Detection ──────────────────────────────────────────────
ANOM_HOURS = 30 * 24
anom_df    = df_test.iloc[:ANOM_HOURS].copy()

fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
fig.suptitle('Anomaly Detection — Ensemble Method Results',
             fontsize=15, fontweight='bold')

xs = range(ANOM_HOURS)

# Panel 1: Demand with anomalies highlighted
axes[0].plot(xs, anom_df['demand_mw'].values, color=PALETTE['actual'], lw=1, alpha=0.8,
             label='Actual Demand')
axes[0].plot(xs, anom_df['forecast_mw'].values, color=PALETTE['forecast'],
             lw=1, linestyle='--', alpha=0.7, label='Forecast')

# True anomalies
true_anom = anom_df['is_anomaly'].values == 1
axes[0].scatter(np.where(true_anom)[0], anom_df['demand_mw'].values[true_anom],
                color='red', s=25, zorder=5, label='True Anomaly', marker='x')
# Predicted anomalies
pred_anom = anom_df['anomaly_pred'].values == 1
axes[0].scatter(np.where(pred_anom)[0], anom_df['demand_mw'].values[pred_anom],
                color='orange', s=20, zorder=4, label='Predicted Anomaly',
                marker='o', alpha=0.7, facecolors='none', edgecolors='orange')
axes[0].set_ylabel('Demand (MW)')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(alpha=0.3)

# Panel 2: Residuals and Z-score threshold
axes[1].plot(xs, anom_df['residual'].values, color='steelblue', lw=0.8, alpha=0.8,
             label='Residual (Actual - Forecast)')
axes[1].axhline(0, color='black', lw=0.7)
axes[1].scatter(np.where(true_anom)[0], anom_df['residual'].values[true_anom],
                color='red', s=25, zorder=5, marker='x', label='True Anomaly')
axes[1].set_ylabel('Residual (MW)')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# Panel 3: Detector votes
axes[2].bar(xs, anom_df['votes'].values, color='darkorange', alpha=0.7, label='Votes')
axes[2].axhline(2, color='red', lw=1.2, linestyle='--', label='Decision threshold (≥2)')
axes[2].set_ylabel('Detector Votes')
axes[2].set_ylim(-0.1, 3.5)
axes[2].set_xlabel('Hour')
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)
axes[2].set_xticks(range(0, ANOM_HOURS, 24 * 5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_anomaly_detection.png")


# ── Figure 4: Optimization Results ──────────────────────────────────────────
OPT_HOURS = 7 * 24
opt7 = opt_result.iloc[:OPT_HOURS].copy()

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Energy Distribution Optimization — 7-Day Dispatch Plan',
             fontsize=15, fontweight='bold')

ax_demand = fig.add_subplot(gs[0, :])
xs = range(OPT_HOURS)
ax_demand.fill_between(xs, 0, opt7['renewable_mw'].values,
                        label='Renewable', color=PALETTE['renewable'], alpha=0.65)
ax_demand.fill_between(xs, opt7['renewable_mw'].values,
                        opt7['renewable_mw'].values + np.clip(opt7['battery_mw'].values, 0, None),
                        label='Battery Discharge', color=PALETTE['battery'], alpha=0.65)
ax_demand.plot(xs, opt7['forecast_mw'].values, color='black', lw=1.5,
               label='Total Demand Forecast', linestyle='--')
ax_demand.axhline(3800, color='red', lw=1, linestyle=':', label='Peak Threshold')
ax_demand.set_ylabel('MW')
ax_demand.set_title('Demand vs Supply Stack')
ax_demand.legend(loc='upper right', fontsize=9)
ax_demand.grid(alpha=0.3)

ax_soc = fig.add_subplot(gs[1, 0])
ax_soc.plot(xs, opt7['soc_pct'].values, color=PALETTE['battery'], lw=1.5)
ax_soc.fill_between(xs, 10, opt7['soc_pct'].values, alpha=0.25, color=PALETTE['battery'])
ax_soc.axhline(10, color='red', lw=0.8, linestyle='--', label='Min SoC')
ax_soc.axhline(95, color='green', lw=0.8, linestyle='--', label='Max SoC')
ax_soc.set_title('Battery State of Charge (%)')
ax_soc.set_ylabel('SoC (%)')
ax_soc.set_ylim(0, 100)
ax_soc.legend(fontsize=8)
ax_soc.grid(alpha=0.3)

ax_bat = fig.add_subplot(gs[1, 1])
bat_vals = opt7['battery_mw'].values
ax_bat.bar(xs, np.clip(bat_vals, 0, None), color=PALETTE['battery'],
           alpha=0.8, label='Discharge', width=1)
ax_bat.bar(xs, np.clip(bat_vals, None, 0), color='#aec7e8',
           alpha=0.8, label='Charge', width=1)
ax_bat.axhline(0, color='black', lw=0.7)
ax_bat.set_title('Battery Charge / Discharge')
ax_bat.set_ylabel('MW (+ discharge, − charge)')
ax_bat.legend(fontsize=8)
ax_bat.grid(alpha=0.3)

ax_dr = fig.add_subplot(gs[2, 0])
ax_dr.bar(xs, opt7['dr_shed_mw'].values, color=PALETTE['dr'], alpha=0.85)
ax_dr.set_title('Demand Response — Load Shed')
ax_dr.set_ylabel('MW shed')
ax_dr.grid(alpha=0.3)

ax_cost = fig.add_subplot(gs[2, 1])
ax_cost.bar(xs, opt7['net_cost_rs'].values / 1000, color=PALETTE['cost'], alpha=0.8)
ax_cost.set_title('Hourly Net Procurement Cost')
ax_cost.set_ylabel('Cost (Rs × 1000)')
ax_cost.grid(alpha=0.3)

plt.savefig(f'{OUTPUT_DIR}/04_optimization_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_optimization_results.png")


# ── Figure 5: Feature Importance + CV Results ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Diagnostics', fontsize=14, fontweight='bold')

# Feature importance
feat_imp = forecaster.feature_importance_.head(15)
colors_fi = ['#ff7f0e' if 'lag' in f or 'rolling' in f else
             '#1f77b4' if 'sin' in f or 'cos' in f else
             '#2ca02c' for f in feat_imp.index]
ax1.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors_fi[::-1])
ax1.set_title('Top 15 Feature Importances (GBM)')
ax1.set_xlabel('Importance')
ax1.grid(axis='x', alpha=0.3)
legend_fi = [Patch(facecolor='#ff7f0e', label='Lagged/Rolling'),
             Patch(facecolor='#1f77b4', label='Cyclical Temporal'),
             Patch(facecolor='#2ca02c', label='Other')]
ax1.legend(handles=legend_fi, fontsize=9)

# CV fold results
folds = cv_results.index.values
ax2.plot(folds, cv_results['RMSE'].values, 'o-', color='#1f77b4', label='RMSE')
ax2b = ax2.twinx()
ax2b.plot(folds, cv_results['MAPE (%)'].values, 's--', color='#ff7f0e', label='MAPE (%)')
ax2.set_title('Walk-Forward CV: RMSE and MAPE per Fold')
ax2.set_xlabel('Fold')
ax2.set_ylabel('RMSE (MW)', color='#1f77b4')
ax2b.set_ylabel('MAPE (%)', color='#ff7f0e')
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_model_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_model_diagnostics.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" STEP 8: Saving Results")
print("=" * 65)

# Save test predictions
test_out = df_test[['timestamp', 'demand_mw', 'temperature_c', 'hour',
                     'day_of_week', 'month', 'season', 'is_weekend', 'is_holiday',
                     'forecast_mw', 'lower_mw', 'upper_mw', 'baseline_mw',
                     'residual', 'is_anomaly', 'anomaly_pred', 'votes',
                     'zscore_flag', 'iforest_flag', 'iqr_flag']].copy()
test_out.to_csv(f'{OUTPUT_DIR}/test_predictions.csv', index=False)

# Save optimization results
opt_result.to_csv(f'{OUTPUT_DIR}/optimization_results.csv', index=False)

# Save metrics summary
summary_lines = [
    "=" * 55,
    " AI-Powered Smart Energy Forecasting — Results Summary",
    "=" * 55,
    "",
    f"Dataset: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
    f"Total hours: {len(df):,}  |  Anomalies: {df['is_anomaly'].sum():,}",
    "",
    "FORECASTING METRICS (GBM on 60-day test set):",
    *[f"  {k}: {v}" for k, v in metrics_gbm.items()],
    "",
    "FORECASTING METRICS (Exp. Smoothing Baseline):",
    *[f"  {k}: {v}" for k, v in metrics_bl.items()],
    "",
    "OPTIMIZATION SUMMARY (60-day horizon):",
    *[f"  {k}: {v}" for k, v in optimizer.summary_report(opt_result).items()],
    "",
    "CROSS-VALIDATION (5-fold walk-forward):",
    f"  Mean RMSE:   {cv_results['RMSE'].mean():.2f} MW",
    f"  Mean MAPE:   {cv_results['MAPE (%)'].mean():.2f}%",
    f"  Mean R²:     {cv_results['R2'].mean():.4f}",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
]
with open(f'{OUTPUT_DIR}/results_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines))
print("  Saved: results_summary.txt")
print("  Saved: test_predictions.csv")
print("  Saved: optimization_results.csv")

print("\n" + "=" * 65)
print(" PIPELINE COMPLETE ✓")
print("=" * 65)
print(f"\n  GBM RMSE  : {metrics_gbm['RMSE']} MW")
print(f"  GBM MAPE  : {metrics_gbm['MAPE (%)']}%")
print(f"  GBM R²    : {metrics_gbm['R2']}")
print(f"  Renewable : {optimizer.summary_report(opt_result)['renewable_coverage_%']}% coverage")
print()
