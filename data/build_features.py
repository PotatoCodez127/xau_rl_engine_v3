import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

logger = logging.getLogger("Feature_Pipeline")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def aggregate_m1_to_m15(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw 1-minute OHLC data into 15-minute closed candles.
    """
    logger.info("Resampling 1-minute raw data into 15-minute structural candles...")
    
    # Ensure datetime index
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        df_raw.index = pd.to_datetime(df_raw.index)

    # Define the aggregation logic for OHLC pricing
    aggregation_dict = {
        'xau_open': 'first',
        'xau_high': 'max',
        'xau_low': 'min',
        'xau_close': 'last',
    }
    
    # If DXY is included in your M1 data, take the last price of the 15m window
    if 'dxy_close' in df_raw.columns:
        aggregation_dict['dxy_close'] = 'last'

    # Resample to 15min (Pandas 2.2+ compliant) and drop any empty periods
    df_m15 = df_raw.resample('15min').agg(aggregation_dict).dropna()
    
    logger.info(f"Aggregation complete. Compressed {len(df_raw)} M1 bars into {len(df_m15)} M15 bars.")
    return df_m15

def build_xau_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms data into the stationary, normalized feature tensor required by V3.2.
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

    # 3. Environment Variables (Used for sizing/gating)
    df['env_atr'] = ta.atr(df['xau_high'], df['xau_low'], df['xau_close'], length=14)
    df['h4_ema'] = ta.ema(df['xau_close'], length=800)
    df['h4_trend'] = np.where(df['xau_close'] > df['h4_ema'], 1.0, -1.0)

    # 4. Neural Network Features (Stationary & Normalized)
    df['close_frac_diff'] = np.log(df['xau_close'] / df['xau_close'].shift(1))
    
    if 'dxy_close' in df.columns:
        df['dxy_pct_change_15m'] = df['dxy_close'].pct_change()
    else:
        # Fallback if DXY is missing from your raw data
        df['dxy_pct_change_15m'] = 0.0 
    
    df['mom_1'] = df['xau_close'].diff(1)
    df['mom_4'] = df['xau_close'].diff(4)
    df['mom_1_norm'] = (df['mom_1'] - df['mom_1'].rolling(1000).mean()) / df['mom_1'].rolling(1000).std()
    df['mom_4_norm'] = (df['mom_4'] - df['mom_4'].rolling(1000).mean()) / df['mom_4'].rolling(1000).std()

    df['h1_vol_regime'] = df['env_atr'] / df['env_atr'].rolling(64).mean()

    df['ema_50'] = ta.ema(df['xau_close'], length=50)
    df['dist_ema_50'] = (df['xau_close'] - df['ema_50']) / df['xau_close']
    df['dist_ema_50_norm'] = (df['dist_ema_50'] - df['dist_ema_50'].rolling(1000).mean()) / df['dist_ema_50'].rolling(1000).std()

    df['rolling_max_15m'] = df['xau_high'].rolling(14).max()
    df['rolling_min_15m'] = df['xau_low'].rolling(14).min()
    df['dist_rolling_max_15m_norm'] = (df['rolling_max_15m'] - df['xau_close']) / df['env_atr']
    df['dist_rolling_min_15m_norm'] = (df['xau_close'] - df['rolling_min_15m']) / df['env_atr']

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

    logger.info(f"Feature Pipeline Complete. Generated {len(df)} stationary rows.")
    return df

if __name__ == "__main__":
    RAW_DATA_PATH = "/content/drive/MyDrive/XAU_RL_V3/data/raw/xauusd_1m.csv" 
    OUT_PATH = "/content/drive/MyDrive/XAU_RL_V3/data/processed_features.parquet"
    
    # 1. Load the CSV without forcing the datetime parse yet
    # \t separator is added as a fallback because MT5 often exports tab-separated files
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH, sep=None, engine='python')
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        exit()
        
    logger.info(f"Raw CSV Columns Detected: {raw_df.columns.tolist()}")
    
    # 2. Standardize MT5 Column Headers dynamically
    # Convert all columns to lowercase and strip whitespace for easier matching
    raw_df.columns = [c.lower().strip().replace('<', '').replace('>', '') for c in raw_df.columns]
    
    # Handle Split Date/Time columns (Standard MT5 Export)
    if 'date' in raw_df.columns and 'time' in raw_df.columns:
        raw_df['datetime'] = pd.to_datetime(raw_df['date'] + ' ' + raw_df['time'])
    # Handle single 'time' column
    elif 'time' in raw_df.columns:
        raw_df['datetime'] = pd.to_datetime(raw_df['time'])
    else:
        raise KeyError(f"Could not identify a valid time column. Available columns: {raw_df.columns}")

    # Map standard OHLC headers to the 'xau_' prefix expected by the pipeline
    rename_map = {
        'open': 'xau_open', 
        'high': 'xau_high', 
        'low': 'xau_low', 
        'close': 'xau_close',
        'tickvol': 'tick_volume',
        'vol': 'real_volume'
    }
    raw_df.rename(columns=rename_map, inplace=True)
    
    # Set the index to our newly standardized datetime column
    raw_df.set_index('datetime', inplace=True)
    
    # Run the feature pipeline
    processed_df = build_xau_features(raw_df)
    processed_df.to_parquet(OUT_PATH)