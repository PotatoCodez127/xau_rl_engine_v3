import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import logging

logger = logging.getLogger("Feature_Pipeline")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def aggregate_m1_to_m15(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw 1-minute OHLC data into 15-minute closed candles.
    """
    logger.info("Resampling 1-minute raw data into 15-minute structural candles...")
    
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw.index = pd.to_datetime(df_raw.index)

    aggregation_dict = {
        'xau_open': 'first',
        'xau_high': 'max',
        'xau_low': 'min',
        'xau_close': 'last',
    }
    
    if 'dxy_close' in df_raw.columns:
        aggregation_dict['dxy_close'] = 'last'

    df_m15 = df_raw.resample('15min').agg(aggregation_dict).dropna()
    
    logger.info(f"Aggregation complete. Compressed {len(df_raw)} M1 bars into {len(df_m15)} M15 bars.")
    return df_m15

def rolling_z_score(series: pd.Series, window: int = 500, eps: float = 1e-8) -> pd.Series:
    """
    Robust Standardization: Normalizes continuous variables against macro regime shifts.
    FIX: Added Softsign/Tanh squashing to strictly bound outputs to ~[-3.0, 3.0]
    preventing Neural Network activation saturation during extreme macro volatility.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    raw_z = (series - rolling_mean) / (rolling_std + eps)
    
    # Soft clip using Tanh to compress extreme outliers gracefully
    return np.tanh(raw_z / 3.0) * 3.0

def build_xau_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms data into the stationary, normalized feature tensor required by V3.2.
    Refactored: Strict Rolling Z-Score normalization to prevent Oracle Activation Starvation.
    """
    logger.info("Initiating Quant Feature Engineering Pipeline...")
    
    # 1. Enforce UTC Temporal Integrity
    df = df_raw.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    df = df.sort_index(ascending=True)

    # 2. AGGREGATE TO 15M TIMEFRAME
    df = aggregate_m1_to_m15(df)

    # 3. Environment Variables (Used for sizing/gating, not for Neural Net)
    df['env_atr'] = ta.atr(df['xau_high'], df['xau_low'], df['xau_close'], length=14)
    df['env_h4_ema'] = ta.ema(df['xau_close'], length=800)
    # H4 Trend remains raw as it is strictly bounded to categorical [-1.0, 1.0]
    df['h4_trend'] = np.where(df['xau_close'] > df['env_h4_ema'], 1.0, -1.0) 

    # 4. Neural Network Features (Strictly Stationary & Normalized)
    
    # Oscillators (Strictly division-scaled to [0.0, 1.0])
    df['rsi_14_norm'] = ta.rsi(df['xau_close'], length=14) / 100.0
    
    # Continuous Variables (Strictly normalized via Rolling Z-Score to ~[-3.0, 3.0])
    close_frac_diff = np.log(df['xau_close'] / df['xau_close'].shift(1))
    df['close_frac_diff_norm'] = rolling_z_score(close_frac_diff)
    
    if 'dxy_close' in df.columns:
        dxy_pct = df['dxy_close'].pct_change()
        df['dxy_pct_change_15m_norm'] = rolling_z_score(dxy_pct)
    else:
        df['dxy_pct_change_15m_norm'] = 0.0 
    
    mom_1 = df['xau_close'].diff(1)
    df['mom_1_norm'] = rolling_z_score(mom_1)
    
    mom_4 = df['xau_close'].diff(4)
    df['mom_4_norm'] = rolling_z_score(mom_4)

    h1_vol_regime = df['env_atr'] / df['env_atr'].rolling(64).mean()
    df['h1_vol_regime_norm'] = rolling_z_score(h1_vol_regime)

    ema_50 = ta.ema(df['xau_close'], length=50)
    dist_ema_50 = (df['xau_close'] - ema_50) / df['xau_close']
    df['dist_ema_50_norm'] = rolling_z_score(dist_ema_50)

    # Volatility Scaled Distance Metrics
    rolling_max_15m = df['xau_high'].rolling(14).max()
    rolling_min_15m = df['xau_low'].rolling(14).min()
    df['dist_rolling_max_15m_norm'] = (rolling_max_15m - df['xau_close']) / df['env_atr']
    df['dist_rolling_min_15m_norm'] = (df['xau_close'] - rolling_min_15m) / df['env_atr']

    # 5. Supervised Target Labeling (1 Hour Forward Look)
    future_return = df['xau_close'].shift(-4) / df['xau_close'] - 1 
    df['target'] = 0
    df.loc[future_return > 0.0015, 'target'] = 1 
    df.loc[future_return < -0.0015, 'target'] = 2 

    # 6. Clean and Fill Structural Columns
    df.dropna(inplace=True)
    
    structural_cols = [
        "dist_res_zone_top_15m_norm", "dist_res_zone_bottom_15m_norm", "dist_sup_zone_top_15m_norm", "dist_sup_zone_bottom_15m_norm",
        "dist_res_zone_top_30m_norm", "dist_res_zone_bottom_30m_norm", "dist_sup_zone_top_30m_norm", "dist_sup_zone_bottom_30m_norm",
        "dist_res_zone_top_4h_norm", "dist_res_zone_bottom_4h_norm", "dist_sup_zone_top_4h_norm", "dist_sup_zone_bottom_4h_norm",
        "dist_daily_eq_norm", "dist_pivot_norm", "dist_R1_norm", "dist_S1_norm"
    ]
    for col in structural_cols:
        if col not in df.columns:
            df[col] = 0.0

    df['prob_long'] = 0.0
    df['prob_short'] = 0.0
    df['prob_hold'] = 1.0

    # 7. MACRO LEAKAGE PREVENTION 
    # Prefix all raw continuous variables with 'env_' so they are completely excluded 
    # from the Neural Network feature vector during the run_wfa.py ingestion phase.
    df.rename(columns={
        'xau_open': 'env_xau_open',
        'xau_high': 'env_xau_high',
        'xau_low': 'env_xau_low',
        'xau_close': 'env_xau_close',
        'dxy_close': 'env_dxy_close'
    }, inplace=True)

    logger.info(f"Feature Pipeline Complete. Generated {len(df)} stationary rows.")
    return df

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/XAUUSDr_M1_OG.csv" 
    OUT_PATH = "data/processed_features.parquet"
    
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH, sep=None, engine='python')
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        exit()
        
    logger.info(f"Raw CSV Columns Detected: {raw_df.columns.tolist()}")
    
    raw_df.columns = [c.lower().strip().replace('<', '').replace('>', '') for c in raw_df.columns]
    
    if 'date' in raw_df.columns and 'time' in raw_df.columns:
        raw_df['datetime'] = pd.to_datetime(raw_df['date'] + ' ' + raw_df['time'])
    elif 'time' in raw_df.columns:
        raw_df['datetime'] = pd.to_datetime(raw_df['time'])
    else:
        raise KeyError(f"Could not identify a valid time column. Available columns: {raw_df.columns}")

    rename_map = {
        'open': 'xau_open', 
        'high': 'xau_high', 
        'low': 'xau_low', 
        'close': 'xau_close',
        'tickvol': 'tick_volume',
        'vol': 'real_volume'
    }
    raw_df.rename(columns=rename_map, inplace=True)
    raw_df.set_index('datetime', inplace=True)
    
    processed_df = build_xau_features(raw_df)
    processed_df.to_parquet(OUT_PATH)