from strategy.optimize_multistrategy import Optimize_multistrategy
from config.config import DEFAULT_PARAMS, MULTISTRAT_PARAMS, MULTISTRAT_PARAMS_2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Multistrategy_manager(Optimize_multistrategy):
    def __init__(self):
        self.strategies = []
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

    def plot_results(self, df: pd.DataFrame):
        """Plot price and account balance history"""
        plt.figure(figsize=(15, 8))
        
        # Create multiple y-axes
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # Offset the right axes to make room for both scales
        ax3.spines['right'].set_position(('outward', 60))
        
        # Plot asset price (e.g., BTC price)
        line1 = ax1.plot(
            df.index, 
            df['close'], 
            label='Asset Price', 
            color='gray', 
            alpha=0.6
        )
        
        # Calculate percentage gains and PnL using the balance history
        balance_series = pd.Series(self.balance_history, index=df.index)
        pct_gains = ((balance_series / DEFAULT_PARAMS['starting_balance'] - 1) * 100)
        pnl_history = (balance_series - DEFAULT_PARAMS['starting_balance'])
        
        # Plot balance as percentage gain
        line2 = ax2.plot(
            df.index, 
            pct_gains,
            label='Account Balance (%)', 
            color='green',
            linewidth=2
        )
        
        # Plot PnL in dollars
        line3 = ax3.plot(
            df.index,
            pnl_history,
            label='PnL ($)',
            color='blue',
            linewidth=2,
            linestyle='--'
        )
        
        # Set labels and title
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Asset Price ($)', color='gray')
        ax2.set_ylabel('Account Balance (%)', color='green')
        ax3.set_ylabel('PnL ($)', color='blue')
        
        plt.title('Trading Strategy Performance', pad=20)
        
        # Combine all lines and labels for the legend
        lines = line1 + line2 + line3
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # Format axis colors
        ax1.tick_params(axis='y', labelcolor='gray')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        # Add grid
        ax2.grid(True, alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.savefig('data/multistrategy_results.png')
        except Exception as e:
            print(f"Error saving figure: {e}")
        finally:
            plt.close()

    def run_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run multiple strategies at the same time"""
        # Store dataframe and calculate indicators for each strategy
        dfs = [df.copy()] * len(self.strategies)
        for i, strategy in enumerate(self.strategies):
            strategy.last_df = df
            dfs[i] = strategy.calculate_indicators(df.copy())

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
            current_rows = [df.copy().iloc[i]] * len(self.strategies)
            for j, strategy in enumerate(self.strategies):
                current_rows[j] = dfs[j].iloc[i]

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
                for i, strategy in enumerate(self.strategies):
                    entry_signal = strategy.check_entry_automated(current_rows[i])
                    if entry_signal != 0:
                        self.current_position = entry_signal
                        self.entry_price = current_price
                        positions[i] = entry_signal
                        self.active_strategy = i
                        self.take_profit_pct = strategy.take_profit_pct
                        stop_losses[i] = strategy.calculate_dynamic_stop_loss(current_rows[i], entry_signal)
                        break
            else:
                positions[i] = 0

            # Update trailing stop if in position
            if self.current_position != 0:
                new_stop = self.strategies[self.active_strategy].calculate_dynamic_stop_loss(current_rows[self.active_strategy], self.current_position)
                if self.current_position == 1:
                    stop_losses[i] = max(new_stop, stop_losses[i-1])
                else:    
                    stop_losses[i] = min(new_stop, stop_losses[i-1])

        # Add results to dataframe
        df['position'] = positions
        df['balance'] = self.balance_history
        df['stop_loss'] = stop_losses

        return df


