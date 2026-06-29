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