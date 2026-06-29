# live_feed.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import timezone

class LiveMT5Feed:
    def __init__(self, xau_symbol="XAUUSD", dxy_symbol="DXY"):
        if not mt5.initialize():
            raise ConnectionError(f"MT5 Initialization failed: {mt5.last_error()}")
        self.xau = xau_symbol
        self.dxy = dxy_symbol

    def get_latest_tick(self):
        """Fetches the current tick and formats it for the StreamingFeatureEngine."""
        tick_xau = mt5.symbol_info_tick(self.xau)
        tick_dxy = mt5.symbol_info_tick(self.dxy)

        if tick_xau is None or tick_dxy is None:
            return None, None

        # Enforce UTC temporal synchronization
        timestamp = pd.to_datetime(tick_xau.time, unit='s').tz_localize('UTC')
        
        tick_row = {
            "xau_open": tick_xau.bid,
            "xau_high": tick_xau.ask,
            "xau_low": tick_xau.bid,
            "xau_close": tick_xau.bid,
            "dxy_close": tick_dxy.bid
        }
        
        return timestamp, tick_row
    
    def fetch_historical_warmup(self, bars=15000):
        """Fetches the last N minutes of data to instantly saturate the engine's memory."""
        print(f"[MT5] Fetching last {bars} M1 bars for Instant Warmup...")
        
        rates_xau = mt5.copy_rates_from_pos(self.xau, mt5.TIMEFRAME_M1, 0, bars)
        rates_dxy = mt5.copy_rates_from_pos(self.dxy, mt5.TIMEFRAME_M1, 0, bars)
        
        if rates_xau is None or rates_dxy is None:
            raise ValueError("Failed to fetch historical data. Check MT5 connection.")
            
        # Format XAU
        df_xau = pd.DataFrame(rates_xau)
        df_xau['datetime'] = pd.to_datetime(df_xau['time'], unit='s').dt.tz_localize('UTC')
        df_xau.set_index('datetime', inplace=True)
        df_xau.rename(columns={'open': 'xau_open', 'high': 'xau_high', 'low': 'xau_low', 'close': 'xau_close'}, inplace=True)
        
        # Format DXY
        df_dxy = pd.DataFrame(rates_dxy)
        df_dxy['datetime'] = pd.to_datetime(df_dxy['time'], unit='s').dt.tz_localize('UTC')
        df_dxy.set_index('datetime', inplace=True)
        df_dxy.rename(columns={'close': 'dxy_close'}, inplace=True)
        
        # Synchronize and drop non-overlapping times
        master_df = df_xau[['xau_open', 'xau_high', 'xau_low', 'xau_close']].join(df_dxy[['dxy_close']], how='inner').dropna()
        return master_df