"""
S&P 500 Regime Detection — Historical Data Pipeline (1980-2026)
===============================================================
Transforms 45 years of daily S&P 500 history and monthly macro
data into a clean feature matrix for regime prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────

def load_historical_data(path: str = "data/raw/sp500_historical_1980_2026.csv") -> pd.DataFrame:
    """Load the historical S&P 500 dataset (1980-2026)."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    print(f"[Data] Loaded {len(df):,} rows | {df.index.min().date()} → {df.index.max().date()}")
    return df


# ─────────────────────────────────────────────────
# 2. TECHNICAL FEATURES
# ─────────────────────────────────────────────────

def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-series features from the historical S&P 500 data.
    Many are already pre-calculated in the CSV, we just extract them
    and add a few regime-specific transforms.
    """
    ts = pd.DataFrame(index=df.index)
    
    # Log returns
    ts["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Momentum (cumulative log returns)
    for window in [5, 10, 20]:
        ts[f"cum_ret_{window}d"] = ts["log_return"].rolling(window).sum()
        
    # Realized Volatility
    for window in [5, 10, 21, 63]:
        ts[f"rvol_{window}d"] = ts["log_return"].rolling(window).apply(
            lambda x: np.sqrt((x**2).sum()) * np.sqrt(252 / window), raw=True
        )
        
    # Volatility Regimes (short vs long term)
    ts["vol_ratio_5_21"] = ts["rvol_5d"] / (ts["rvol_21d"] + 1e-8)
    ts["vol_ratio_5_63"] = ts["rvol_5d"] / (ts["rvol_63d"] + 1e-8)
    
    # Pre-calculated technicals from the dataset
    if "RSI_14" in df.columns:
        ts["rsi_14"] = df["RSI_14"]
        
    if "MACD_Hist" in df.columns:
        ts["macd_hist"] = df["MACD_Hist"]
        
    # Moving Average Distances
    for sma in [21, 50, 200]:
        col = f"SMA_{sma}"
        if col in df.columns:
            ts[f"price_vs_sma{sma}"] = (df["Close"] - df[col]) / (df[col] + 1e-8)
            
    if "BB_Width" in df.columns:
        ts["bb_width"] = df["BB_Width"]
        
    # Normalized Volume (ratio to 50d MA to handle long-term growth)
    if "Volume" in df.columns:
        vol_ma50 = df["Volume"].rolling(50).mean()
        ts["vol_ratio_50d"] = df["Volume"] / (vol_ma50 + 1e-8)
        
    # Autocorrelation (Mean-reversion vs Momentum regime indicator)
    ts["autocorr_10"] = ts["log_return"].rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Lagged Returns
    for lag in range(1, 6):
        ts[f"ret_lag_{lag}"] = ts["log_return"].shift(lag)
        
    print(f"[Features] Built {len(ts.columns)} technical features")
    return ts


# ─────────────────────────────────────────────────
# 3. MACROECONOMIC FEATURES
# ─────────────────────────────────────────────────

def build_macro_features(daily_index: pd.DatetimeIndex,
                         macro_path: str = "data/raw/data.csv") -> pd.DataFrame:
    """
    Load monthly macroeconomic data and align it to daily frequency.
    Avoids calculating short-term differences on step-function forward-filled data,
    which creates artificial spikes.
    """
    macro = pd.read_csv(macro_path)
    macro["Date"] = pd.to_datetime(macro["Date"])
    macro = macro.set_index("Date").sort_index()
    
    # Keep history from 1975 to allow for rolling calculations
    macro = macro.loc["1975-01-01":].copy()
    
    mf = pd.DataFrame(index=macro.index)
    
    # Valuation
    mf["pe10"] = macro["PE10"]
    mf["earnings_yield"] = np.where(macro["PE10"] > 0, 1.0 / macro["PE10"], 0)
    
    # Interest rates (level and 1-year change, NOT 3-month change)
    mf["long_rate"] = macro["Long Interest Rate"]
    mf["long_rate_chg_1y"] = macro["Long Interest Rate"].diff(12)
    
    # Dividend yield
    mf["dividend_yield"] = np.where(
        macro["SP500"] > 0,
        macro["Dividend"] / macro["SP500"],
        0
    )
    
    # Inflation
    mf["cpi_yoy"] = macro["Consumer Price Index"].pct_change(12)
    mf["cpi_acceleration"] = mf["cpi_yoy"].diff(12)
    
    # Real price momentum
    mf["real_price_mom_1y"] = macro["Real Price"].pct_change(12)
    
    # Danger interaction
    mf["rate_pe_interaction"] = mf["long_rate"] * mf["pe10"]
    
    # Forward fill to daily
    mf = mf.reindex(daily_index.union(mf.index)).sort_index()
    mf = mf.ffill()
    mf = mf.reindex(daily_index)
    
    print(f"[Features] Built {len(mf.columns)} macroeconomic features")
    return mf


# ─────────────────────────────────────────────────
# 4. TARGET CONSTRUCTION
# ─────────────────────────────────────────────────

def build_target(index_return: pd.Series, horizon: int = 20) -> pd.Series:
    """
    y_t = 1 if cumulative return over next `horizon` days > 0, else 0.
    With historical data, h=20 (1 month) is a stable target for macro features.
    """
    # Sum of returns over the next 'horizon' days
    future_cum_return = index_return.rolling(horizon).sum().shift(-horizon)
    target = (future_cum_return > 0).astype(int)
    target.name = "target"
    
    pct_up = target.mean()
    print(f"[Target] Horizon={horizon}d | Up={pct_up:.1%} | Down={1-pct_up:.1%}")
    return target


# ─────────────────────────────────────────────────
# 5. NORMALIZATION
# ─────────────────────────────────────────────────

def rolling_zscore_normalize(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Trailing 252-day z-score normalization.
    Adapts to long-term regime shifts without look-ahead bias.
    """
    rolling_mean = df.rolling(window, min_periods=60).mean()
    rolling_std = df.rolling(window, min_periods=60).std()
    normalized = (df - rolling_mean) / (rolling_std + 1e-8)
    normalized = normalized.clip(-3, 3)
    return normalized


# ─────────────────────────────────────────────────
# 6. PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────

def run_pipeline(data_path: str = "data/raw/sp500_historical_1980_2026.csv",
                 macro_path: str = "data/raw/data.csv",
                 horizon: int = 20,
                 normalize: bool = True) -> tuple:
    
    print("=" * 60)
    print(f"S&P 500 REGIME DETECTION (1980-2026) | Horizon={horizon}d")
    print("=" * 60)
    
    # Load
    df = load_historical_data(data_path)
    
    # Features
    ts_features = build_technical_features(df)
    macro_features = build_macro_features(df.index, macro_path)
    features = pd.concat([ts_features, macro_features], axis=1)
    
    # Target (h=20)
    target = build_target(ts_features["log_return"].fillna(0), horizon=horizon)
    
    # Align and drop early rows with NaNs (e.g., from 200d MA)
    combined = pd.concat([features, target], axis=1).dropna()
    print(f"\n[Pipeline] Valid dataset: {combined.shape[0]} days × {combined.shape[1]-1} features")
    
    X = combined.drop(columns=["target"])
    y = combined["target"]
    
    # Robust Split
    # Train: 1990 - 2015 (Covers Dot-Com & 2008 crashes)
    # Test: 2016 - 2026 (Covers COVID & 2022 drops)
    train_end = "2015-12-31"
    train_start = "1990-01-01"
    
    mask_train = (X.index >= train_start) & (X.index <= train_end)
    mask_test = X.index > train_end
    
    X_train = X.loc[mask_train]
    X_test = X.loc[mask_test]
    y_train = y.loc[mask_train]
    y_test = y.loc[mask_test]
    
    print(f"[Split] Train ({train_start} to {train_end}): {len(X_train)} days")
    print(f"[Split] Test  (> {train_end}): {len(X_test)} days")
    print(f"[Split] Train Up% = {y_train.mean():.1%} | Test Up% = {y_test.mean():.1%}")
    
    # Normalize
    if normalize:
        print("[Norm] Applying rolling z-score normalization...")
        X_all = pd.concat([X_train, X_test])
        X_all_norm = rolling_zscore_normalize(X_all, window=252)
        X_train = X_all_norm.loc[X_train.index].dropna()
        X_test = X_all_norm.loc[X_test.index].dropna()
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]
        
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    feature_names = X_train.columns.tolist()
    print(f"\n[Done] Final shapes: Train={X_train.shape}, Test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats = run_pipeline()
    print(f"Features: {feats}")
