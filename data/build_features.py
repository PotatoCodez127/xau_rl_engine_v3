import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from datetime import timezone

logger = logging.getLogger("Feature_Pipeline")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def build_xau_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw 15m XAUUSD and DXY data into the stationary, normalized 
    feature tensor required by the V3.2 architecture.
    """
    logger.info("Initiating Quant Feature Engineering Pipeline...")
    df = df_raw.copy()

    # 1. Enforce UTC Temporal Integrity
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    df = df.sort_index(ascending=True)

    # 2. Environment Variables (Used for sizing/gating, not fed to neural nets)
    # ATR for dynamic Stop Loss scaling
    df['env_atr'] = ta.atr(df['xau_high'], df['xau_low'], df['xau_close'], length=14)
    
    # The Absolute Macro Gate (H4 Trend)
    # Since we are on a 15m chart, 4 Hours = 16 periods. 50-EMA on H4 = 16 * 50 = 800 periods.
    df['h4_ema'] = ta.ema(df['xau_close'], length=800)
    df['h4_trend'] = np.where(df['xau_close'] > df['h4_ema'], 1.0, -1.0)

    # 3. Neural Network Features (Stationary & Normalized)
    # Fractional Differentiation proxy (Log returns)
    df['close_frac_diff'] = np.log(df['xau_close'] / df['xau_close'].shift(1))
    
    # Intermarket Correlation
    df['dxy_pct_change_15m'] = df['dxy_close'].pct_change()
    
    # Normalized Momentum
    df['mom_1'] = df['xau_close'].diff(1)
    df['mom_4'] = df['xau_close'].diff(4)
    df['mom_1_norm'] = (df['mom_1'] - df['mom_1'].rolling(1000).mean()) / df['mom_1'].rolling(1000).std()
    df['mom_4_norm'] = (df['mom_4'] - df['mom_4'].rolling(1000).mean()) / df['mom_4'].rolling(1000).std()

    # Volatility Regime
    df['h1_vol_regime'] = df['env_atr'] / df['env_atr'].rolling(64).mean() # 64 periods = 16 hours

    # Distance to Structural Moving Averages (Normalized)
    df['ema_50'] = ta.ema(df['xau_close'], length=50)
    df['dist_ema_50'] = (df['xau_close'] - df['ema_50']) / df['xau_close']
    df['dist_ema_50_norm'] = (df['dist_ema_50'] - df['dist_ema_50'].rolling(1000).mean()) / df['dist_ema_50'].rolling(1000).std()

    # Rolling Min/Max Distances (Wick Zones)
    df['rolling_max_15m'] = df['xau_high'].rolling(14).max()
    df['rolling_min_15m'] = df['xau_low'].rolling(14).min()
    df['dist_rolling_max_15m_norm'] = (df['rolling_max_15m'] - df['xau_close']) / df['env_atr']
    df['dist_rolling_min_15m_norm'] = (df['xau_close'] - df['rolling_min_15m']) / df['env_atr']

    # Generate Dummy Target for Oracle Training (Supervised Labeling)
    # 0 = Hold, 1 = Long Breakout, 2 = Short Breakout
    future_return = df['xau_close'].shift(-4) / df['xau_close'] - 1 # 1 hour forward looking
    df['target'] = 0
    df.loc[future_return > 0.0015, 'target'] = 1 # +0.15% momentum expansion
    df.loc[future_return < -0.0015, 'target'] = 2 # -0.15% momentum expansion

    # 4. Clean and Export
    df.dropna(inplace=True)
    
    # Ensure all required cols exist, filling missing structural zones with 0.0 for safety
    structural_cols = [
        "dist_res_zone_top_15m_norm", "dist_res_zone_bottom_15m_norm", "dist_sup_zone_top_15m_norm", "dist_sup_zone_bottom_15m_norm",
        "dist_res_zone_top_30m_norm", "dist_res_zone_bottom_30m_norm", "dist_sup_zone_top_30m_norm", "dist_sup_zone_bottom_30m_norm",
        "dist_res_zone_top_4h_norm", "dist_res_zone_bottom_4h_norm", "dist_sup_zone_top_4h_norm", "dist_sup_zone_bottom_4h_norm",
        "dist_daily_eq_norm", "dist_pivot_norm", "dist_R1_norm", "dist_S1_norm"
    ]
    for col in structural_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Initialize Phase A probabilities to 0 (These get injected during the WFA pipeline)
    df['prob_long'] = 0.0
    df['prob_short'] = 0.0
    df['prob_hold'] = 1.0

    logger.info(f"Feature Pipeline Complete. Generated {len(df)} stationary rows.")
    return df

if __name__ == "__main__":
    # Example execution for Google Colab
    RAW_DATA_PATH = "/content/drive/MyDrive/XAU_RL_V3/data/raw/xauusd_15m.csv"
    OUT_PATH = "/content/drive/MyDrive/XAU_RL_V3/data/processed_features.parquet"
    
    # Load MT5 CSV export
    raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=['datetime'], index_col='datetime')
    processed_df = build_xau_features(raw_df)
    processed_df.to_parquet(OUT_PATH)