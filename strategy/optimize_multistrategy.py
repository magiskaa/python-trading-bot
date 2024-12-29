from strategy.strategy_base import Strategy_base
from config.config import DEFAULT_PARAMS
import numpy as np
import pandas as pd

class Optimize_multistrategy(Strategy_base):
    def __init__(self, bb_period, bb_std, adx_period, adx_threshold, 
                 rsi_period, rsi_overbought, rsi_oversold, stop_loss_pct, take_profit_pct, 
                 atr_period, atr_multiplier, keltner_period, keltner_atr_factor, hma_period, 
                 vwap_std, macd_fast_period, macd_slow_period, macd_signal_period, mfi_period, obv_ma_period):
        # Initialize the base class with common parameters
        super().__init__(DEFAULT_PARAMS['starting_balance'], DEFAULT_PARAMS['leverage'])

        # Set strategy parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.keltner_period = keltner_period
        self.keltner_atr_factor = keltner_atr_factor
        self.hma_period = hma_period
        self.vwap_std = vwap_std
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.mfi_period = mfi_period
        self.obv_ma_period = obv_ma_period
        
        # Initialize additional attributes
        self.current_position = 0
        self.entry_price = 0.0
        self.current_balance = self.starting_balance
        self.balance_history = []
        self.last_df = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # Bollinger Bands
        if self.bb_period is not None:
            df['bb_middle'], df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(
                df['close'],
                period=self.bb_period,
                std_dev=self.bb_std
            )
        
        # ADX
        if self.adx_period is not None:
            df['adx'] = self.calculate_adx(
                df['high'],
                df['low'],
                df['close'],
                period=self.adx_period
            )
        
        # RSI
        if self.rsi_period is not None:
            df['rsi'] = self.calculate_rsi(df['close'], period=self.rsi_period)
        
        # ATR
        df['atr'] = self.calculate_atr(
            df['high'], df['low'], df['close'], period=self.atr_period
        )

        # Keltner Channels
        if self.keltner_period is not None:
            df['kc_middle'], df['kc_upper'], df['kc_lower'] = self.calculate_keltner_channels(
                df['high'], df['low'], df['close'],
                period=self.keltner_period,
                atr_factor=self.keltner_atr_factor
            )

        # HMA
        if self.hma_period is not None:
            df['hma'] = self.calculate_hma(
                df['close'], period=self.hma_period
            )
        
        # VWAP
        if self.vwap_std is not None:
            df['vwap'], df['vwap_upper'], df['vwap_lower'] = self.calculate_vwap(df)

        # Calculate MACD
        if self.macd_fast_period is not None:
            df['macd_line'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(
                df['close'],
                fast_period=self.macd_fast_period,
                slow_period=self.macd_slow_period,
                signal_period=self.macd_signal_period
            )

        # Calculate OBV
        if self.obv_ma_period is not None:
            df['obv'] = self.calculate_obv(df['close'], df['volume'])
            # Calculate OBV trend (difference from previous value)
            df['obv_trend'] = df['obv'].diff()
            # Calculate OBV Moving Average
            df['obv_ma'] = df['obv'].rolling(window=self.obv_ma_period).mean()

        # Calculate MFI
        if self.mfi_period is not None:
            df['mfi'] = self.calculate_mfi(
                df['high'],
                df['low'],
                df['close'],
                df['volume'],
                period=self.mfi_period
            )

        return df
    
    def check_entry_automated(self, row: pd.Series) -> int:
        """Entry conditions check for automated_optimization"""
        buy_signals = []
        sell_signals = []

        if self.bb_period is not None:
            bb_condition_long = row['close'] < row['bb_lower']
            buy_signals.append(bb_condition_long)
            bb_condition_short = row['close'] > row['bb_upper']
            sell_signals.append(bb_condition_short)

        if self.adx_period is not None:
            adx_condition = row['adx'] > self.adx_threshold
            buy_signals.append(adx_condition)
            sell_signals.append(adx_condition)
        
        if self.rsi_period is not None:
            rsi_condition_long = row['rsi'] < self.rsi_oversold
            buy_signals.append(rsi_condition_long)
            rsi_condition_short = row['rsi'] > self.rsi_overbought
            sell_signals.append(rsi_condition_short)
            
        if self.keltner_period is not None:
            keltner_condition_long = row['close'] < row['kc_lower']
            buy_signals.append(keltner_condition_long)
            keltner_condition_short = row['close'] > row['kc_upper']
            sell_signals.append(keltner_condition_short)

        if self.hma_period is not None:
            hma_trend_long = row['close'] > row['hma']
            buy_signals.append(hma_trend_long)
            hma_trend_short = row['close'] < row['hma']
            sell_signals.append(hma_trend_short)

        if self.vwap_std is not None:
            vwap_condition_long = row['close'] < row['vwap_lower']
            buy_signals.append(vwap_condition_long)
            vwap_condition_short = row['close'] > row['vwap_upper']
            sell_signals.append(vwap_condition_short)

        if self.macd_fast_period is not None:
            macd_condition_long = row['macd_line'] > row['macd_signal']
            buy_signals.append(macd_condition_long)
            macd_condition_short = row['macd_line'] < row['macd_signal']
            sell_signals.append(macd_condition_short)

        if self.mfi_period is not None:
            mfi_condition_long = row['mfi'] < 20
            buy_signals.append(mfi_condition_long)
            mfi_condition_short = row['mfi'] > 80
            sell_signals.append(mfi_condition_short)

        if self.obv_ma_period is not None:
            obv_ma_condition_long = row['obv'] > row['obv_ma']
            buy_signals.append(obv_ma_condition_long)
            obv_ma_condition_short = row['obv'] < row['obv_ma']
            sell_signals.append(obv_ma_condition_short)

        if all(buy_signals):
            self.current_position = 1
            self.entry_price = row['close']
            return 1  # Buy signal
        
        elif all(sell_signals):
            self.current_position = -1
            self.entry_price = row['close']
            return -1  # Sell signal
        
        return 0  # Hold
  
    def calculate_dynamic_stop_loss(self, row: pd.Series, position: int) -> float:
        """Calculate dynamic stop loss based on ATR"""
        if position == 1:
            return row['close'] * (1 - max(self.stop_loss_pct, (row['atr'] * self.atr_multiplier) / row['close']))
        elif position == -1:
            return row['close'] * (1 + max(self.stop_loss_pct, (row['atr'] * self.atr_multiplier) / row['close']))
        return 0
    

