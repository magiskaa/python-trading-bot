from strategy.strategy_base import Strategy_base
from strategy.optimize_multistrategy import Optimize_multistrategy
from config.config import DEFAULT_PARAMS, MULTISTRAT_PARAMS, MULTISTRAT_PARAMS_2
import numpy as np
import pandas as pd

class Multistrategy_manager(Optimize_multistrategy):
    def __init__(self):
        self.strategies = []

        self.reset_state()

    def reset_state(self):
        """Reset the strategy state for new runs"""
        self.current_position = 0
        self.entry_price = 0
        self.take_profit_pct = MULTISTRAT_PARAMS['take_profit_pct']
        self.pnl = []
        self.trades = []
        self.active_strategy = None
        self.current_balance = DEFAULT_PARAMS['starting_balance']
        self.peak_balance = DEFAULT_PARAMS['starting_balance']
        self.max_drawdown = 0
        self.balance_history = [DEFAULT_PARAMS['starting_balance']]
        self.position_size = DEFAULT_PARAMS['starting_balance'] * DEFAULT_PARAMS['leverage'] * 0.8

    def add_strategy(self, bb_period, bb_std, adx_period, adx_threshold, 
                     rsi_period, rsi_overbought, rsi_oversold, stop_loss_pct, 
                     take_profit_pct, atr_period, atr_multiplier, keltner_period, 
                     keltner_atr_factor, hma_period, vwap_std, macd_fast_period, 
                     macd_slow_period, macd_signal_period, mfi_period, obv_ma_period):
        # Creates a new strategy
        strategy = Optimize_multistrategy(
            bb_period,
            bb_std,
            adx_period,
            adx_threshold,
            rsi_period,
            rsi_overbought,
            rsi_oversold,
            stop_loss_pct,
            take_profit_pct,
            atr_period,
            atr_multiplier,
            keltner_period,
            keltner_atr_factor,
            hma_period,
            vwap_std,
            macd_fast_period,
            macd_slow_period,
            macd_signal_period,
            mfi_period,
            obv_ma_period
        )
        # Adds the strategy to the used strategies list
        self.strategies.append(strategy)
        
    def calculate_metrics(self):
        """Calculate and print strategy metrics"""
        total_pnl = sum(self.pnl)
        num_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        max_drawd = self.max_drawdown

        if num_trades > 0:
            win_rate = (winning_trades / num_trades) * 100
            avg_win = (
                np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0])
                if winning_trades > 0 else 0
            )
            avg_loss = (
                np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0])
                if num_trades - winning_trades > 0 else 0
            )

            print(f"\nStrategy Metrics:")
            print(f"Total PnL: ${total_pnl:.2f}")
            print(f"Number of trades: {num_trades}")
            print(f"Win rate: {win_rate:.2f}%")
            print(f"Average winning trade: ${avg_win:.2f}")
            print(f"Average losing trade: ${avg_loss:.2f}")
            print(f"Final balance: ${(DEFAULT_PARAMS['starting_balance'] + total_pnl):.2f}")
            print(f"Max drawdown: {max_drawd * 100:.2f}%")
        else:
            print("No trades were made.")

    def run_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run multiple strategies at the same time"""
        # Store dataframe and calculate indicators
        self.strategies[0].last_df = df
        self.strategies[1].last_df = df
        df0 = self.strategies[0].calculate_indicators(df.copy())
        df1 = self.strategies[1].calculate_indicators(df.copy())

        # Counter and for printing trade details (for debugging)
        counter = 0
        isDebug = False # Change to True if you want to print trades

        # Initialize arrays
        df_len = len(df)
        positions = np.zeros(df_len)
        self.balance_history = [DEFAULT_PARAMS['starting_balance']] * df_len
        stop_losses = np.zeros(df_len)

        for i in range(df_len):
            current_price = df['close'].iloc[i]
            current_row_0 = df0.iloc[i]
            current_row_1 = df1.iloc[i]

            # Check exit conditions first
            if self.current_position != 0:
                # Check stop loss and take profit 
                stop_hit = (
                    (self.current_position == 1 and current_price < stop_losses[i-1]) or
                    (self.current_position == -1 and current_price > stop_losses[i-1])
                )

                take_profit_hit = (
                    (self.current_position == 1 and (current_price - self.entry_price) / self.entry_price > self.take_profit_pct) or
                    (self.current_position == -1 and (self.entry_price - current_price) / self.entry_price > self.take_profit_pct)
                )

                if stop_hit or take_profit_hit:
                    # Calculate PnL
                    price_change = (current_price - self.entry_price) / self.entry_price
                    pnl = self.position_size * price_change * self.current_position
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

                    # Print trade details (for debugging)
                    if isDebug:
                        print("enp:", self.trades[counter]['entry_price'])
                        print("exp:", self.trades[counter]['exit_price'])
                        print("pnl:", self.trades[counter]['pnl'])
                        print("pos:", self.trades[counter]['position'])
                        print("bal:", self.trades[counter]['balance_after'])
                        print("ext:", self.trades[counter]['exit_type'])
                        print("pos_s:", self.position_size)
                        counter += 1

                    # Reset position
                    self.current_position = 0
                    positions[i] = 0
                else:
                    positions[i] = self.current_position
            else: 
                positions[i] = 0

            # Update balance history
            self.balance_history[i] = self.current_balance

            # Update position size
            self.position_size = self.current_balance * DEFAULT_PARAMS['leverage'] * 0.8
            
            # Check entry conditions if not in position
            if self.current_position == 0:
                entry_signal_0 = self.strategies[0].check_entry_automated(current_row_0)
                entry_signal_1 = self.strategies[1].check_entry_automated(current_row_1)
                if entry_signal_0 != 0:
                    self.current_position = entry_signal_0
                    self.entry_price = current_price
                    positions[i] = entry_signal_0
                    self.active_strategy = 0
                    self.take_profit_pct = self.strategies[0].take_profit_pct
                    stop_losses[i] = self.strategies[0].calculate_dynamic_stop_loss(current_row_0, entry_signal_0)

                elif entry_signal_1 != 0:
                    self.current_position = entry_signal_1
                    self.entry_price = current_price
                    positions[i] = entry_signal_1
                    self.active_strategy = 1
                    self.take_profit_pct = self.strategies[1].take_profit_pct
                    stop_losses[i] = self.strategies[1].calculate_dynamic_stop_loss(current_row_1, entry_signal_1)
                else:
                    positions[i] = 0
            else:
                positions[i] = self.current_position

            # Update trailing stop if in position
            if self.current_position != 0:
                if self.active_strategy == 0:
                    new_stop = self.strategies[0].calculate_dynamic_stop_loss(current_row_0, self.current_position)
                else:
                    new_stop = self.strategies[1].calculate_dynamic_stop_loss(current_row_1, self.current_position)

                if self.current_position == 1:
                    stop_losses[i] = max(new_stop, stop_losses[i-1])
                else:    
                    stop_losses[i] = min(new_stop, stop_losses[i-1])
                
        # Add results to dataframe
        df['position'] = positions
        df['balance'] = self.balance_history
        df['stop_loss'] = stop_losses

        return df


