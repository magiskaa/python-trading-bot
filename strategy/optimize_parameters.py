from strategy.strategy_base import Strategy_base
import numpy as np
import pandas as pd
from config.config import DEFAULT_PARAMS

class Optimize_parameters(Strategy_base):
    def __init__(self,
                 starting_balance: float = DEFAULT_PARAMS['starting_balance'],
                 leverage: int = DEFAULT_PARAMS['leverage'],
                 bb_period: int = DEFAULT_PARAMS['bb_period'],
                 bb_std: float = DEFAULT_PARAMS['bb_std'],
                 adx_period: int = DEFAULT_PARAMS['adx_period'],
                 adx_threshold: int = DEFAULT_PARAMS['adx_threshold'],
                 rsi_period: int = DEFAULT_PARAMS['rsi_period'],
                 rsi_overbought: int = DEFAULT_PARAMS['rsi_overbought'],
                 rsi_oversold: int = DEFAULT_PARAMS['rsi_oversold'],
                 stop_loss_pct: float = DEFAULT_PARAMS['stop_loss_pct'],
                 take_profit_pct: float = DEFAULT_PARAMS['take_profit_pct'],
                 atr_period: int = DEFAULT_PARAMS['atr_period'],
                 atr_multiplier: float = DEFAULT_PARAMS['atr_multiplier'],
                 keltner_period: int = DEFAULT_PARAMS['keltner_period'],
                 keltner_atr_factor: float = DEFAULT_PARAMS['keltner_atr_factor'],
                 hma_period: int = DEFAULT_PARAMS['hma_period'],
                 vwap_std: float = DEFAULT_PARAMS['vwap_std'],
                 macd_fast_period: int = DEFAULT_PARAMS['macd_fast_period'],
                 macd_slow_period: int = DEFAULT_PARAMS['macd_slow_period'],
                 macd_signal_period: int = DEFAULT_PARAMS['macd_signal_period'],
                 mfi_period: int = DEFAULT_PARAMS['mfi_period'],
                 obv_ma_period: int = DEFAULT_PARAMS['obv_ma_period'],):
        
        # Initialize the base class with common parameters
        super().__init__(starting_balance, leverage)
        
        # Initialize strategy-specific parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Enhanced strategy parameters
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
        # Calculate base indicators
        df = super().calculate_indicators(df)
        
        # Calculate additional indicators from base class methods
        df['atr'] = self.calculate_atr(
            df['high'], df['low'], df['close'], period=self.atr_period
        )
        df['kc_middle'], df['kc_upper'], df['kc_lower'] = self.calculate_keltner_channels(
            df['high'], df['low'], df['close'],
            period=self.keltner_period,
            atr_factor=self.keltner_atr_factor
        )
        df['hma'] = self.calculate_hma(
            df['close'], period=self.hma_period
        )
        df['vwap'], df['vwap_upper'], df['vwap_lower'] = self.calculate_vwap(df)

        # Calculate MACD
        df['macd_line'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(
            df['close'],
            fast_period=self.macd_fast_period,
            slow_period=self.macd_slow_period,
            signal_period=self.macd_signal_period
        )

        # Calculate OBV
        df['obv'] = self.calculate_obv(df['close'], df['volume'])

        # Calculate OBV trend (difference from previous value)
        df['obv_trend'] = df['obv'].diff()

        # Calculate OBV Moving Average
        df['obv_ma'] = df['obv'].rolling(window=self.obv_ma_period).mean()

        # Calculate MFI
        df['mfi'] = self.calculate_mfi(
            df['high'],
            df['low'],
            df['close'],
            df['volume'],
            period=self.mfi_period
        )

        return df

    def check_entry_conditions(self, row: pd.Series) -> int:
        """Enhanced entry conditions check"""
        bb_condition_long = row['close'] < row['bb_lower']
        bb_condition_short = row['close'] > row['bb_upper']
        adx_condition = row['adx'] > self.adx_threshold
        rsi_condition_long = row['rsi'] < self.rsi_oversold
        rsi_condition_short = row['rsi'] > self.rsi_overbought
        keltner_condition_long = row['close'] < row['kc_lower']
        keltner_condition_short = row['close'] > row['kc_upper']
        hma_trend = 1 if row['close'] > row['hma'] else -1
        vwap_condition_long = row['close'] < row['vwap_lower']
        vwap_condition_short = row['close'] > row['vwap_upper']
        macd_condition_long = row['macd_line'] > row['macd_signal']
        macd_condition_short = row['macd_line'] < row['macd_signal']
        mfi_condition_long = row['mfi'] < 20
        mfi_condition_short = row['mfi'] > 80
        obv_condition_long = row['obv_trend'] > 0
        obv_condition_short = row['obv_trend'] < 0
        obv_ma_condition_long = row['obv'] > row['obv_ma']
        obv_ma_condition_short = row['obv'] < row['obv_ma']
        
        # Long entry
        if (bb_condition_long and adx_condition and rsi_condition_long and
            obv_ma_condition_long):
            return 1
            
        # Short entry
        elif (bb_condition_short and adx_condition and rsi_condition_short and
              obv_ma_condition_short):
            return -1

        return 0

    def calculate_dynamic_stop_loss(self, row: pd.Series, position: int) -> float:
        """Calculate dynamic stop loss based on ATR"""
        if position == 1:
            return row['close'] * (1 - max(self.stop_loss_pct, (row['atr'] * self.atr_multiplier) / row['close']))
        elif position == -1:
            return row['close'] * (1 + max(self.stop_loss_pct, (row['atr'] * self.atr_multiplier) / row['close']))
        return 0
    
    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the enhanced trading strategy with dynamic stops"""
        # Store dataframe and calculate indicators
        self.last_df = df
        df = self.calculate_indicators(df)
        laskuri = 0
        
        # Initialize arrays
        positions = np.zeros(len(df))
        self.balance_history = [self.starting_balance] * len(df)
        stop_losses = np.zeros(len(df))
        
        # Start after warmup period to ensure all indicators are available
        start_idx = max(
            self.bb_period, 
            self.adx_period, 
            self.rsi_period,
            #self.keltner_period,
            #self.hma_period,
            #self.macd_slow_period,
            #self.mfi_period,
            self.obv_ma_period
        )
        
        for i in range(start_idx, len(df)):
            current_price = df['close'].iloc[i]
            current_row = df.iloc[i]
            
            # Check exit conditions first
            if self.current_position != 0:
                # Check stop loss and take profit
                stop_hit = (
                    (self.current_position == 1 and current_price < stop_losses[i-1]) or
                    (self.current_position == -1 and current_price > stop_losses[i-1])
                )
                
                take_profit_hit = (
                    (self.current_position == 1 and 
                    (current_price - self.entry_price) / self.entry_price > self.take_profit_pct) or
                    (self.current_position == -1 and 
                    (self.entry_price - current_price) / self.entry_price > self.take_profit_pct)
                )
                
                if stop_hit or take_profit_hit:
                    # Calculate PnL
                    pnl = self.calculate_pnl(current_price)
                    self.pnl.append(pnl)
                    self.current_balance += pnl
                    
                    # Update peak balance and drawdown
                    self.peak_balance = max(self.peak_balance, self.current_balance)
                    current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                    self.max_drawdown = max(self.max_drawdown, current_drawdown)
                    
                    # Record trade
                    self.trades.append({
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'position': self.current_position,
                        'balance_after': self.current_balance,
                        'exit_type': 'stop_loss' if stop_hit else 'take_profit'
                    })
                    if len(self.trades) < 4:
                        print("enp:", self.trades[laskuri]['entry_price'])
                        print("exp:", self.trades[laskuri]['exit_price'])
                        print("pnl:", self.trades[laskuri]['pnl'])
                        print("pos:", self.trades[laskuri]['position'])
                        print("bal:", self.trades[laskuri]['balance_after'])
                        print("ext:", self.trades[laskuri]['exit_type'])
                    laskuri += 1

                    # Reset position
                    self.current_position = 0
                    positions[i] = 0
                else:
                    positions[i] = self.current_position
            else:
                positions[i] = 0
                    
            # Update balance history
            self.balance_history[i] = self.current_balance

            arvot = [800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000]
            lisättävä = 1000
            for arvo in arvot:
                if self.current_balance > arvo:
                    self.position_size = self.starting_balance / 2 * self.leverage + lisättävä
                    lisättävä += 1000

            # Check entry conditions if not in position
            if self.current_position == 0:
                entry_signal = self.check_entry_conditions(current_row)
                
                if entry_signal != 0:
                    self.current_position = entry_signal
                    self.entry_price = current_price
                    positions[i] = entry_signal
                    
                    # Set initial stop loss
                    stop_losses[i] = self.calculate_dynamic_stop_loss(
                        current_row, 
                        entry_signal
                    )
                    if len(self.trades) < 3:
                        print("Current price", current_price)
                        print("Initial stop loss:", stop_losses[i])
                else:
                    positions[i] = 0
            else:
                positions[i] = self.current_position
                    
            # Update trailing stop if in position
            if self.current_position != 0:
                new_stop = self.calculate_dynamic_stop_loss(
                    current_row, 
                    self.current_position
                )
                if len(self.trades) < 3:
                    print("Current price", current_price)
                    print("New stop loss:", new_stop)
                if self.current_position == 1:
                    stop_losses[i] = max(new_stop, stop_losses[i-1])
                else:
                    stop_losses[i] = min(new_stop, stop_losses[i-1])
            
        # Add results to dataframe
        df['position'] = positions
        df['balance'] = self.balance_history
        df['stop_loss'] = stop_losses
        
        return df


