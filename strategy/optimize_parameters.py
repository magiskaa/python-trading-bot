import json
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

    def run_strategy(self, df: pd.DataFrame, isMetrics=False) -> pd.DataFrame:
        """Run the enhanced trading strategy with dynamic stops"""
        # Store dataframe and calculate indicators
        df = self.calculate_indicators(df)

        # Counter for printing trade details (for debugging)
        counter = 0

        isHighlow = True

        trades = {}
        
        # Initialize arrays
        positions = np.zeros(len(df))
        self.balance_history = [self.starting_balance] * len(df)
        stop_losses = np.zeros(len(df))
        take_profits = np.zeros(len(df))
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            lowest_price = df['low'].iloc[i]
            highest_price = df['high'].iloc[i]
            current_row = df.iloc[i]

            isInitial = False
            
            # Check exit conditions first
            if self.current_position != 0:
                # Check stop loss and take profit
                stop_hit = (
                    (self.current_position == 1 and lowest_price < stop_losses[i-1]) or
                    (self.current_position == -1 and highest_price > stop_losses[i-1])
                )
                take_profit_hit = (
                    (self.current_position == 1 and highest_price > take_profits[i-1]) or
                    (self.current_position == -1 and lowest_price < take_profits[i-1])
                )
        
                if take_profit_hit:
                    self.stop_loss_or_take_profit_hit(take_profits[i-1], type='take_profit')

                    if isMetrics:
                        trade = self.trades[counter]
                        trades[str(counter+1)] = {
                            "position": trade["position"],
                            "position_size": self.position_size,
                            "entry_price": trade["entry_price"],
                            "exit_price": trade["exit_price"],
                            "exit_type": trade["exit_type"],
                            "stop_loss": stop_losses[i-1],
                            "take_profit": take_profits[i-1],
                            "pnl": trade["pnl"],
                            "balance_after": trade["balance_after"],
                        }
                        with open('data/trades.json', 'w') as f:
                            json.dump(trades, f, ensure_ascii=False, indent=4)
                            
                        counter += 1

                    self.current_position = 0
                    positions[i] = 0
                elif stop_hit:
                    self.stop_loss_or_take_profit_hit(stop_losses[i-1], type='stop_loss')

                    if isMetrics:
                        trade = self.trades[counter]
                        trades[str(counter+1)] = {
                            "position": trade["position"],
                            "position_size": self.position_size,
                            "entry_price": trade["entry_price"],
                            "exit_price": trade["exit_price"],
                            "exit_type": trade["exit_type"],
                            "stop_loss": stop_losses[i-1],
                            "take_profit": take_profits[i-1],
                            "pnl": trade["pnl"],
                            "balance_after": trade["balance_after"],
                        }
                        with open('data/trades.json', 'w') as f:
                            json.dump(trades, f, ensure_ascii=False, indent=4)

                        counter += 1

                    self.current_position = 0
                    positions[i] = 0
                else:
                    positions[i] = self.current_position
            else:
                positions[i] = 0
                    
            # Update balance history
            self.balance_history[i] = self.current_balance
            if self.current_balance <= 0:
                self.current_balance = 0
                self.balance_history[i] = 0
                break

            # Update position size
            if self.position_size < 25000:
                self.position_size = self.current_balance * self.leverage * 0.7
            else:
                self.position_size = 25000

            # Check entry conditions if not in position
            if self.current_position == 0:
                entry_signal = self.check_entry_automated(current_row)
                if entry_signal != 0:
                    self.current_position = entry_signal
                    self.entry_price = current_price
                    positions[i] = entry_signal
                    # Set initial stop loss
                    if isHighlow:
                        stop_losses[i] = self.calculate_dynamic_stop_loss_highlow(current_row, self.current_position)
                    else:
                        stop_losses[i] = self.calculate_dynamic_stop_loss(current_row, self.current_position)

                    if self.current_position == 1:
                        take_profits[i] = self.entry_price * (1 + self.take_profit_pct)
                        if stop_losses[i] > self.entry_price:
                            stop_losses[i] = self.entry_price - self.entry_price * self.stop_loss_pct
                    elif self.current_position == -1:
                        take_profits[i] = self.entry_price * (1 - self.take_profit_pct)
                        if stop_losses[i] < self.entry_price:
                            stop_losses[i] = self.entry_price + self.entry_price * self.stop_loss_pct
                    
                    stop_losses[i-1] = stop_losses[i]
                    take_profits[i-1] = take_profits[i]

                    isInitial = True
                else:
                    positions[i] = 0
            else:
                positions[i] = self.current_position
                    
            # Update trailing stop if in position
            if self.current_position != 0 and isInitial == False:
                if isHighlow:
                    new_stop = self.calculate_dynamic_stop_loss_highlow(current_row, self.current_position)
                else:
                    new_stop = self.calculate_dynamic_stop_loss(current_row, self.current_position)
                if self.current_position == 1:
                    stop_losses[i] = max(new_stop, stop_losses[i-1])
                    if stop_losses[i] > current_price:
                        stop_losses[i] = current_price - current_price * 0.001
                else:
                    stop_losses[i] = min(new_stop, stop_losses[i-1])
                    if stop_losses[i] < current_price:
                        stop_losses[i] = current_price + current_price * 0.001
                take_profits[i] = take_profits[i-1]
            
        # Add results to dataframe
        df['position'] = positions
        df['balance'] = self.balance_history
        df['stop_loss'] = stop_losses
        
        return df
