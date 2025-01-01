from binance.client import Client
from binance.enums import *
import time
import pandas as pd
import numpy as np
from strategy.multistrategy_manager import Multistrategy_manager
from config.config import API_KEY, API_SECRET, SYMBOL, BACKTEST_START, MULTISTRAT_PARAMS, MULTISTRAT_PARAMS_2, MULTISTRAT_PARAMS_3, DEFAULT_PARAMS

class BinanceBot(Multistrategy_manager):
    def __init__(self):
        super().__init__()
    
    def run_strategies(self, df: pd.DataFrame):
        pass






client = Client(API_KEY, API_SECRET, tld='com')

def fetch_data(client, symbol, interval, start_str):
    klines = client.get_historical_klines(symbol, interval, start_str)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                        'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def main():
    data = fetch_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START)

    bot = BinanceBot()

    bot.add_strategy(
        bb_period=MULTISTRAT_PARAMS['bb_period'],
        bb_std=MULTISTRAT_PARAMS['bb_std'],
        adx_period=MULTISTRAT_PARAMS['adx_period'],
        adx_threshold=MULTISTRAT_PARAMS['adx_threshold'],
        rsi_period=MULTISTRAT_PARAMS['rsi_period'],
        rsi_overbought=MULTISTRAT_PARAMS['rsi_overbought'],
        rsi_oversold=MULTISTRAT_PARAMS['rsi_oversold'],
        stop_loss_pct=MULTISTRAT_PARAMS['stop_loss_pct'],
        take_profit_pct=MULTISTRAT_PARAMS['take_profit_pct'],
        atr_period=MULTISTRAT_PARAMS['atr_period'],
        atr_multiplier=MULTISTRAT_PARAMS['atr_multiplier'],
        keltner_period=MULTISTRAT_PARAMS['keltner_period'],
        keltner_atr_factor=MULTISTRAT_PARAMS['keltner_atr_factor'],
        hma_period=MULTISTRAT_PARAMS['hma_period'],
        vwap_std=MULTISTRAT_PARAMS['vwap_std'],
        macd_fast_period=MULTISTRAT_PARAMS['macd_fast_period'],
        macd_slow_period=MULTISTRAT_PARAMS['macd_slow_period'],
        macd_signal_period=MULTISTRAT_PARAMS['macd_signal_period'],
        mfi_period=MULTISTRAT_PARAMS['mfi_period'],
        obv_ma_period=MULTISTRAT_PARAMS['obv_ma_period']
    )
    bot.add_strategy(
        bb_period=MULTISTRAT_PARAMS_2['bb_period'],
        bb_std=MULTISTRAT_PARAMS_2['bb_std'],
        adx_period=MULTISTRAT_PARAMS_2['adx_period'],
        adx_threshold=MULTISTRAT_PARAMS_2['adx_threshold'],
        rsi_period=MULTISTRAT_PARAMS_2['rsi_period'],
        rsi_overbought=MULTISTRAT_PARAMS_2['rsi_overbought'],
        rsi_oversold=MULTISTRAT_PARAMS_2['rsi_oversold'],
        stop_loss_pct=MULTISTRAT_PARAMS_2['stop_loss_pct'],
        take_profit_pct=MULTISTRAT_PARAMS_2['take_profit_pct'],
        atr_period=MULTISTRAT_PARAMS_2['atr_period'],
        atr_multiplier=MULTISTRAT_PARAMS_2['atr_multiplier'],
        keltner_period=MULTISTRAT_PARAMS_2['keltner_period'],
        keltner_atr_factor=MULTISTRAT_PARAMS_2['keltner_atr_factor'],
        hma_period=MULTISTRAT_PARAMS_2['hma_period'],
        vwap_std=MULTISTRAT_PARAMS_2['vwap_std'],
        macd_fast_period=MULTISTRAT_PARAMS_2['macd_fast_period'],
        macd_slow_period=MULTISTRAT_PARAMS_2['macd_slow_period'],
        macd_signal_period=MULTISTRAT_PARAMS_2['macd_signal_period'],
        mfi_period=MULTISTRAT_PARAMS_2['mfi_period'],
        obv_ma_period=MULTISTRAT_PARAMS_2['obv_ma_period']
    )
    bot.add_strategy(
        bb_period=MULTISTRAT_PARAMS_3['bb_period'],
        bb_std=MULTISTRAT_PARAMS_3['bb_std'],
        adx_period=MULTISTRAT_PARAMS_3['adx_period'],
        adx_threshold=MULTISTRAT_PARAMS_3['adx_threshold'],
        rsi_period=MULTISTRAT_PARAMS_3['rsi_period'],
        rsi_overbought=MULTISTRAT_PARAMS_3['rsi_overbought'],
        rsi_oversold=MULTISTRAT_PARAMS_3['rsi_oversold'],
        stop_loss_pct=MULTISTRAT_PARAMS_3['stop_loss_pct'],
        take_profit_pct=MULTISTRAT_PARAMS_3['take_profit_pct'],
        atr_period=MULTISTRAT_PARAMS_3['atr_period'],
        atr_multiplier=MULTISTRAT_PARAMS_3['atr_multiplier'],
        keltner_period=MULTISTRAT_PARAMS_3['keltner_period'],
        keltner_atr_factor=MULTISTRAT_PARAMS_3['keltner_atr_factor'],
        hma_period=MULTISTRAT_PARAMS_3['hma_period'],
        vwap_std=MULTISTRAT_PARAMS_3['vwap_std'],
        macd_fast_period=MULTISTRAT_PARAMS_3['macd_fast_period'],
        macd_slow_period=MULTISTRAT_PARAMS_3['macd_slow_period'],
        macd_signal_period=MULTISTRAT_PARAMS_3['macd_signal_period'],
        mfi_period=MULTISTRAT_PARAMS_3['mfi_period'],
        obv_ma_period=MULTISTRAT_PARAMS_3['obv_ma_period']
    )




if __name__ == '__main__':
    main()

