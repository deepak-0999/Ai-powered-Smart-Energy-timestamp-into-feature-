"""
Module 2: Demand Forecasting Models
- GradientBoostingRegressor (primary, lightweight ML model)
- Exponential Smoothing baseline (Holt-Winters-like via scipy)
- Walk-forward validation
- Probabilistic forecasting via quantile regression
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


# ─── Feature columns used for ML forecasting ────────────────────────────────
FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'season_code',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    'temperature_c', 'is_weekend', 'is_holiday',
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_24h', 'rolling_7d',
]


class EnergyForecaster:
    """
    Gradient Boosting-based short-term electricity demand forecaster.
    Includes probabilistic forecasting (lower/upper quantiles).
    """

    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.05):
        self.model_median = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            loss='squared_error',
            random_state=42,
            subsample=0.8,
        )
        self.model_lower = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            loss='quantile',
            alpha=0.10,           # 10th percentile
            random_state=42,
        )
        self.model_upper = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            loss='quantile',
            alpha=0.90,           # 90th percentile
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.is_fitted = False

    def fit(self, df_train):
        """
        Train all three models on training DataFrame.

        Parameters
        ----------
        df_train : pd.DataFrame  Must contain FEATURE_COLS and 'demand_mw'
        """
        X = df_train[FEATURE_COLS].values
        y = df_train['demand_mw'].values

        X_scaled = self.scaler.fit_transform(X)

        print("  Training median model ...")
        self.model_median.fit(X_scaled, y)
        print("  Training lower quantile (10th) ...")
        self.model_lower.fit(X_scaled, y)
        print("  Training upper quantile (90th) ...")
        self.model_upper.fit(X_scaled, y)

        # Feature importances from median model
        self.feature_importance_ = pd.Series(
            self.model_median.feature_importances_,
            index=FEATURE_COLS
        ).sort_values(ascending=False)

        self.is_fitted = True
        print("  Forecaster fitted successfully.")

    def predict(self, df):
        """
        Return DataFrame with forecast_mw, lower_mw, upper_mw.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        X = df[FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)

        result = df[['timestamp']].copy() if 'timestamp' in df.columns else df.copy()
        result['forecast_mw'] = self.model_median.predict(X_scaled)
        result['lower_mw']    = self.model_lower.predict(X_scaled)
        result['upper_mw']    = self.model_upper.predict(X_scaled)
        result['forecast_mw'] = result['forecast_mw'].clip(lower=0)
        result['lower_mw']    = result['lower_mw'].clip(lower=0)
        result['upper_mw']    = result['upper_mw'].clip(lower=0)
        return result


class ExponentialSmoothingBaseline:
    """
    Simple exponential smoothing baseline for benchmarking.
    Uses seasonal naive as the trend + alpha-smoothed residual.
    """

    def __init__(self, alpha=0.3, seasonal_period=24):
        self.alpha = alpha
        self.seasonal_period = seasonal_period
        self.seasonal_factors = None
        self.base_level = None

    def fit(self, series):
        """Fit on a numpy array or pd.Series of demand values."""
        s = np.array(series)
        sp = self.seasonal_period

        # Compute hourly seasonal factors from first 4 weeks
        n_weeks = min(4 * 7 * 24, len(s)) // (7 * 24) * (7 * 24)
        chunk = s[:n_weeks].reshape(-1, sp)
        self.seasonal_factors = chunk.mean(axis=0) / chunk.mean()

        # De-seasonalise and smooth
        de_seas = s / np.tile(self.seasonal_factors, len(s) // sp + 1)[:len(s)]
        level = de_seas[0]
        for val in de_seas[1:]:
            level = self.alpha * val + (1 - self.alpha) * level
        self.base_level = level

    def predict(self, steps, start_hour=0):
        """Generate `steps` one-step-ahead forecasts."""
        sp = self.seasonal_period
        hours = [(start_hour + i) % sp for i in range(steps)]
        sf = self.seasonal_factors
        return np.array([self.base_level * sf[h] for h in hours])


def evaluate_forecast(y_true, y_pred):
    """Return dict with RMSE, MAE, MAPE, R2."""
    from sklearn.metrics import r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    r2   = r2_score(y_true, y_pred)
    return {'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
            'MAPE (%)': round(mape, 2), 'R2': round(r2, 4)}


def walk_forward_validate(df, n_splits=5):
    """
    Time-series walk-forward cross validation.
    Returns aggregated metrics across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = df[FEATURE_COLS].values
    y = df['demand_mw'].values
    scaler = StandardScaler()

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4,
            learning_rate=0.08, random_state=42
        )
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_val_s)

        metrics = evaluate_forecast(y_val, y_pred)
        metrics['fold'] = fold + 1
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}: RMSE={metrics['RMSE']}, "
              f"MAPE={metrics['MAPE (%)']:.2f}%, R2={metrics['R2']:.4f}")

    summary = pd.DataFrame(fold_metrics).set_index('fold')
    print("\n  Cross-Validation Summary:")
    print(summary.mean().round(3).to_string())
    return summary


if __name__ == '__main__':
    from data_generator import generate_dataset
    df = generate_dataset()

    # Remove rows with NaN (from lag features)
    df = df.dropna().reset_index(drop=True)

    # Train/test split: last 30 days as test
    split = -30 * 24
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    print("=== Walk-Forward Cross Validation ===")
    cv_results = walk_forward_validate(df_train)

    print("\n=== Training Final Forecaster ===")
    forecaster = EnergyForecaster()
    forecaster.fit(df_train)

    print("\n=== Evaluating on Test Set ===")
    preds = forecaster.predict(df_test)
    metrics = evaluate_forecast(df_test['demand_mw'].values,
                                preds['forecast_mw'].values)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Top 10 Feature Importances ===")
    print(forecaster.feature_importance_.head(10).to_string())
