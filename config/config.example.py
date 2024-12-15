# Binance API Configuration
API_KEY = 'YOUR_BINANCE_API_KEY'
API_SECRET = 'YOUR_BINANCE_API_SECRET'

# Default Strategy Parameters
DEFAULT_PARAMS = {
    'starting_balance': 100,
    'leverage': 10,
    'bb_period': 20,
    'bb_std': 2.5,
    'adx_period': 20,
    'adx_threshold': 25,
    'rsi_period': 20,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04,
    'atr_period': 20,
    'atr_multiplier': 2.5,
    'keltner_period': 20,
    'keltner_atr_factor': 2.0,
    'hma_period': 20,
    'vwap_std': 2.0,
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'mfi_period': 14,
    'obv_ma_period': 14,
}

# Trading Settings
SYMBOL = "YOUR_SYMBOL" # Example: "BTCUSDT"
BACKTEST_START = "1 Jan, 2024"