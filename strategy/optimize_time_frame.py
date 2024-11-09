from strategy.strategy_base import Strategy_base
import pandas as pd
from typing import Dict
from config.config import DEFAULT_PARAMS

class Optimize_time_frame(Strategy_base):
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
                 keltner_atr_factor: float = DEFAULT_PARAMS['keltner_atr_factor']):
        
        super().__init__(starting_balance, leverage)

        # Strategy-specific parameters
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
        
        self.last_df = None
        self.reset_state()

    def calculate_dynamic_stop_loss(self, row: pd.Series, position: int) -> float:
        """Calculate dynamic stop loss based on ATR"""
        if position == 1:
            return row['close'] - (row['atr'] * self.atr_multiplier)
        elif position == -1:
            return row['close'] + (row['atr'] * self.atr_multiplier)
        return 0

    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the trading strategy"""
        # Store the dataframe for later use in plotting
        self.last_df = df
        
        # Calculate indicators first
        df = self.calculate_indicators(df)
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], period=self.atr_period)
        df['kc_middle'], df['kc_upper'], df['kc_lower'] = self.calculate_keltner_channels(
            df['high'], df['low'], df['close'],
            period=self.keltner_period,
            atr_factor=self.keltner_atr_factor
        )
        
        
        # Initialize positions array with the same length as the dataframe
        positions = [0] * len(df)
        stop_losses = [0.0] * len(df)
        
        # Initialize balance history with starting balance
        self.balance_history = [self.starting_balance] * len(df)
        
        # Start from the maximum period to ensure all indicators are available
        start_idx = max(self.bb_period, self.adx_period, self.rsi_period, self.atr_period, self.keltner_period)
        
        for i in range(start_idx, len(df)):
            current_price = df['close'].iloc[i]
            current_row = df.iloc[i]
            
            # Update balance history
            self.balance_history[i] = self.current_balance
            
            # Check if in a position
            if self.current_position != 0:
                # Update trailing stop loss
                new_stop = self.calculate_dynamic_stop_loss(current_row, self.current_position)
                if self.current_position == 1:
                    stop_losses[i] = max(new_stop, stop_losses[i - 1])
                else:
                    stop_losses[i] = min(new_stop, stop_losses[i - 1])
                
                # Check if stop loss is hit
                stop_hit = (
                    (self.current_position == 1 and current_price <= stop_losses[i]) or
                    (self.current_position == -1 and current_price >= stop_losses[i])
                )
                
                # Check take profit
                price_change = (current_price - self.entry_price) / self.entry_price
                take_profit_hit = (
                    (self.current_position == 1 and price_change >= self.take_profit_pct) or
                    (self.current_position == -1 and price_change <= -self.take_profit_pct)
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
                    
                    # Reset position
                    self.current_position = 0
                    positions[i] = 0
                else:
                    positions[i] = self.current_position
            else:
                # Entry conditions
                if (pd.notna(df['adx'].iloc[i]) and 
                    pd.notna(df['bb_lower'].iloc[i]) and 
                    pd.notna(df['rsi'].iloc[i]) and 
                    pd.notna(df['kc_lower'].iloc[i])):
                    
                    # Long entry
                    if (df['adx'].iloc[i] > self.adx_threshold and
                        df['close'].iloc[i] < df['bb_lower'].iloc[i] and
                        df['rsi'].iloc[i] < self.rsi_oversold and
                        df['close'].iloc[i] < df['kc_lower'].iloc[i]):
                        
                        self.current_position = 1
                        self.entry_price = current_price
                        positions[i] = 1
                        
                        # Set initial stop loss
                        stop_losses[i] = self.calculate_dynamic_stop_loss(current_row, self.current_position)
                    
                    # Short entry
                    elif (df['adx'].iloc[i] > self.adx_threshold and
                          df['close'].iloc[i] > df['bb_upper'].iloc[i] and
                          df['rsi'].iloc[i] > self.rsi_overbought and
                          df['close'].iloc[i] > df['kc_upper'].iloc[i]):
                        
                        self.current_position = -1
                        self.entry_price = current_price
                        positions[i] = -1
                        
                        # Set initial stop loss
                        stop_losses[i] = self.calculate_dynamic_stop_loss(current_row, self.current_position)
                    else:
                        positions[i] = 0
                else:
                    positions[i] = 0
                    stop_losses[i] = stop_losses[i - 1] if i > start_idx else 0
        
        # Add results to dataframe
        df['position'] = positions
        df['balance'] = self.balance_history
        df['stop_loss'] = stop_losses
        
        return df
            
    def evaluate_parameters(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Evaluate a single set of parameters"""
        try:
            # Update strategy parameters
            self.bb_period = params['bb_period']
            self.bb_std = params['bb_std']
            self.adx_period = params['adx_period']
            self.adx_threshold = params['adx_threshold']
            self.rsi_period = params['rsi_period']
            self.rsi_overbought = params['rsi_overbought']
            self.rsi_oversold = params['rsi_oversold']
            self.stop_loss_pct = params['stop_loss_pct']
            self.take_profit_pct = params['take_profit_pct']

            # Store the dataframe
            self.last_df = df.copy()  # Store a copy of the dataframe

            # Get performance metrics
            performance = self.get_strategy_performance(df)

            return {**params, **performance}
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            return {**params, 'total_pnl': None, 'win_rate': None, 'profit_factor': None, 'num_trades': None, 'sharpe_ratio': None}


