import json
import pandas as pd
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from config.config import HIGHLOW

class Strategy_base:
    def __init__(self, starting_balance: float, leverage: int):
        self.starting_balance = starting_balance
        self.leverage = leverage
        self.position_size = self.starting_balance * self.leverage * 0.8

        self.reset_state()

    def reset_state(self):
        """Reset the strategy state for new runs"""
        self.current_position = 0
        self.entry_price = 0
        self.pnl = []
        self.trades = []
        self.current_balance = self.starting_balance
        self.peak_balance = self.starting_balance
        self.max_drawdown = 0
        self.balance_history = [self.starting_balance]
        self.position_size = self.starting_balance * self.leverage * 0.8

    def calculate_bollinger_bands(self, series, period=20, std_dev=2.0):
        rolling_mean = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return rolling_mean, upper_band, lower_band

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_adx(self, high, low, close, period=14):
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate Plus Directional Movement (+DM)
        plus_dm = high.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -low.diff()), 0)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)

        # Calculate Minus Directional Movement (-DM)
        minus_dm = -low.diff()
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > high.diff()), 0)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, atr_factor: float = 2.0) -> tuple:
        """Calculate Keltner Channels"""
        typical_price = (high + low + close) / 3
        middle = typical_price.rolling(window=period).mean()
        atr = self.calculate_atr(high, low, close, period=self.atr_period)
        upper = middle + (atr * atr_factor)
        lower = middle - (atr * atr_factor)
        return middle, upper, lower

    def calculate_hma(self, close: pd.Series, period: int = 21) -> pd.Series:
        """Calculate Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = close.rolling(window=half_period).mean()
        wma_full = close.rolling(window=period).mean()
        hma = (2 * wma_half - wma_full).rolling(window=sqrt_period).mean()
        return hma

    def calculate_vwap(self, df: pd.DataFrame) -> tuple:
        """Calculate VWAP with standard deviation bands"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Calculate standard deviation
        cum_vol = df['volume'].cumsum()
        cum_vol_tp = (df['volume'] * typical_price).cumsum()
        cum_vol_tp2 = (df['volume'] * typical_price ** 2).cumsum()
        std_dev = np.sqrt((cum_vol_tp2 - (cum_vol_tp ** 2) / cum_vol) / cum_vol)

        upper_band = vwap + (std_dev * self.vwap_std)
        lower_band = vwap - (std_dev * self.vwap_std)
        return vwap, upper_band, lower_band

    def calculate_macd(self, series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = series.ewm(span=fast_period, adjust=False).mean()
        exp2 = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume (OBV)"""
        obv = pd.Series(index=close.index, dtype='float64')
        obv.iloc[0] = 0
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index (MFI)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi

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
            return 1  # Buy signal
        
        elif all(sell_signals):
            return -1  # Sell signal
        
        return 0  # Hold

    def check_entry_automated_print(self, row: pd.Series) -> int:
        """Entry conditions check for automated_optimization"""
        buy_signals = []
        sell_signals = []

        if self.bb_period is not None:
            bb_condition_long = row['close'] < row['bb_lower']
            buy_signals.append(bb_condition_long)
            bb_condition_short = row['close'] > row['bb_upper']
            sell_signals.append(bb_condition_short)
            print(f"bb_l: {row['close']} < {row['bb_lower']} {bb_condition_long}")
            print(f"bb_s: {row['close']} > {row['bb_upper']} {bb_condition_short}")

        if self.adx_period is not None:
            adx_condition = row['adx'] > self.adx_threshold
            buy_signals.append(adx_condition)
            sell_signals.append(adx_condition)            
            print(f"adx: {row['adx']} > {self.adx_threshold} {adx_condition}")
        
        if self.rsi_period is not None:
            rsi_condition_long = row['rsi'] < self.rsi_oversold
            buy_signals.append(rsi_condition_long)
            rsi_condition_short = row['rsi'] > self.rsi_overbought
            sell_signals.append(rsi_condition_short)
            print(f"rsi_l: {row['rsi']} < {self.rsi_oversold} {rsi_condition_long}")
            print(f"rsi_s: {row['rsi']} > {self.rsi_overbought} {rsi_condition_short}")
            
        if self.keltner_period is not None:
            keltner_condition_long = row['close'] < row['kc_lower']
            buy_signals.append(keltner_condition_long)
            keltner_condition_short = row['close'] > row['kc_upper']
            sell_signals.append(keltner_condition_short)
            print(f"kc_l: {row['close']} < {row['kc_lower']} {keltner_condition_long}")
            print(f"kc_s: {row['close']} > {row['kc_upper']} {keltner_condition_short}")

        if self.hma_period is not None:
            hma_trend_long = row['close'] > row['hma']
            buy_signals.append(hma_trend_long)
            hma_trend_short = row['close'] < row['hma']
            sell_signals.append(hma_trend_short)
            print(f"hma_l: {row['close']} > {row['hma']} {hma_trend_long}")
            print(f"hma_s: {row['close']} < {row['hma']} {hma_trend_short}")

        if self.vwap_std is not None:
            vwap_condition_long = row['close'] < row['vwap_lower']
            buy_signals.append(vwap_condition_long)
            vwap_condition_short = row['close'] > row['vwap_upper']
            sell_signals.append(vwap_condition_short)
            print(f"vwap_l: {row['close']} < {row['vwap_lower']} {vwap_condition_long}")
            print(f"vwap_s: {row['close']} > {row['vwap_upper']} {vwap_condition_short}")

        if self.macd_fast_period is not None:
            macd_condition_long = row['macd_line'] > row['macd_signal']
            buy_signals.append(macd_condition_long)
            macd_condition_short = row['macd_line'] < row['macd_signal']
            sell_signals.append(macd_condition_short)
            print(f"macd_l: {row['macd_line']} > {row['macd_signal']} {macd_condition_long}")
            print(f"macd_s: {row['macd_line']} < {row['macd_signal']} {macd_condition_short}")

        if self.mfi_period is not None:
            mfi_condition_long = row['mfi'] < 20
            buy_signals.append(mfi_condition_long)
            mfi_condition_short = row['mfi'] > 80
            sell_signals.append(mfi_condition_short)
            print(f"mfi_l: {row['mfi']} < {20} {mfi_condition_long}")
            print(f"mfi_s: {row['mfi']} > {80} {mfi_condition_short}")

        if self.obv_ma_period is not None:
            obv_ma_condition_long = row['obv'] > row['obv_ma']
            buy_signals.append(obv_ma_condition_long)
            obv_ma_condition_short = row['obv'] < row['obv_ma']
            sell_signals.append(obv_ma_condition_short)
            print(f"obv_l: {row['obv']} < {row['obv_ma']} {obv_ma_condition_long}")
            print(f"obv_s: {row['obv']} > {row['obv_ma']} {obv_ma_condition_short}")

        if all(buy_signals):
            return 1  # Buy signal
        
        elif all(sell_signals):
            return -1  # Sell signal
        
        return 0  # Hold

    def calculate_atr_stop_loss(self, row: pd.Series, position: int, entry: float) -> float:
        if position == 1:
            return entry - row['atr'] * self.atr_multiplier
        elif position == -1:
            return entry + row['atr'] * self.atr_multiplier
        return 0

    def calculate_dynamic_stop_loss_highlow(self, row: pd.Series, position: int) -> float:
        """Calculate dynamic stop loss based on ATR"""
        if position == 1:
            return row['high'] * (1 - max(self.stop_loss_pct, (row['atr'] * self.atr_multiplier) / row['close']))
        elif position == -1:
            return row['low'] * (1 + max(self.stop_loss_pct, (row['atr'] * self.atr_multiplier) / row['close']))
        return 0

    def calculate_dynamic_stop_loss(self, row: pd.Series, position: int) -> float:
        """Calculate dynamic stop loss based on ATR"""
        if position == 1:
            return row['close'] * (1 - max(self.stop_loss_pct, (row['atr'] * self.atr_multiplier) / row['close']))
        elif position == -1:
            return row['close'] * (1 + max(self.stop_loss_pct, (row['atr'] * self.atr_multiplier) / row['close']))
        return 0

    def stop_loss_or_take_profit_hit(self, exit_price, type):
        # Calculate PnL
        pnl = self.calculate_pnl(exit_price)
        self.pnl.append(pnl)
        self.current_balance += pnl

        # Enforce margin call
        if self.current_balance <= 0:
            pnl_adjustment = -self.current_balance
            self.current_balance = 0
            pnl -= pnl_adjustment

        # Update peak balance and drawdown
        self.peak_balance = max(self.peak_balance, self.current_balance)
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Record trade
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'position': self.current_position,
            'balance_after': self.current_balance,
            'exit_type': type
        })
        
    def calculate_pnl(self, exit_price: float) -> float:
        """Calculate PnL for a closed trade"""
        price_change = (exit_price - self.entry_price) / self.entry_price
        pnl = self.position_size * price_change * self.current_position
        pnl -= self.position_size * 0.0005 # Taker fees
        return pnl
    
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
        pct_gains = ((balance_series / self.starting_balance - 1) * 100)
        pnl_history = (balance_series - self.starting_balance)
        
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
            plt.savefig('data/chart_strategy_results.png')
        except Exception as e:
            print(f"Error saving figure: {e}")
        finally:
            plt.close()

    def update_balance_and_drawdown(self, pnl: float):
        """Update account balance and track drawdown"""
        self.current_balance += pnl
        self.balance_history.append(self.current_balance)
        
        # Update peak balance if we've reached a new high
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Calculate current drawdown
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        # Update max drawdown if current drawdown is larger
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def calculate_metrics(self, df: pd.DataFrame):
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
            print(f"Final balance: ${(self.starting_balance + total_pnl):.2f}")
            print(f"Max drawdown: {max_drawd * 100:.2f}%")
        else:
            print("No trades were made.")

    def get_strategy_performance(self, df: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        total_pnl = sum(self.pnl)
        num_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        
        if num_trades == 0:
            return {
                'total_pnl': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        win_rate = (winning_trades / num_trades) * 100
        
        # Calculate profit factor
        gross_profits = sum([t['pnl'] for t in self.trades if t['pnl'] > 0]) or 0
        gross_losses = abs(sum([t['pnl'] for t in self.trades if t['pnl'] < 0])) or 1
        profit_factor = gross_profits / gross_losses
        
        # Calculate Sharpe Ratio (assuming daily returns)
        daily_returns = pd.Series(self.pnl).fillna(0)
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 1 else 0
        

        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown * 100
        }

    def optimize_step_by_step(self, data, initial_params):
        """Optimize parameters using parallel execution."""
        
        best_params = {**initial_params}
        improved = True
        iteration = 1

        while improved and iteration <= 5:
            improved = False
            print(f"\nIteration {iteration}")
            print("Current best parameters:")
            for param, value in best_params.items():
                if value is not None:
                    print(f"    '{param}': {value},")

            for param_name, current_value in best_params.items():
                # Determine the range of test values based on the parameter type
                if isinstance(current_value, int):
                    test_values = [current_value + i for i in range(-20, 21) if current_value + i > 0]
                elif isinstance(current_value, float):
                    test_values = [round(current_value * (0.70 + i * 0.01), 5) for i in range(61) if round(current_value * (0.70 + i * 0.01), 5) >= 0.003]
                else:
                    continue

                print(f"\nTesting {param_name}:")

                best_metric = float('-inf')
                best_value = current_value

                # Prepare tasks for parallel execution
                with ProcessPoolExecutor() as executor:
                    futures = {
                        executor.submit(self.optimize_parameters, data, best_params, param_name, value): value
                        for value in test_values
                    }

                    for future in as_completed(futures):
                        value = futures[future]
                        try:
                            result = future.result()
                            if result is None:
                                continue

                            combined_metric = result['combined_metric']

                            if combined_metric > best_metric:
                                best_metric = combined_metric
                                best_value = value

                            print(
                                f"Value: {value}, PnL: ${result['pnl']:.2f}, "
                                f"MDD: {result['mdd']:.2f}%, "
                                f"Combined Metric: {combined_metric:.6f}, "
                                f"Trades: {result['num_trades']}, "
                                f"Win rate: {result['win_rate']:.2f}%"
                            )
                        except Exception as e:
                            print(f"Error testing value {value}: {str(e)}")
                            continue

                if best_value != current_value:
                    best_params[param_name] = best_value
                    improved = True
                    print(f"Updated {param_name} to {best_value}")

            iteration += 1

        # Update strategy with the best parameters found
        for param, value in best_params.items():
            setattr(self, param, value)

        return best_params

    def optimize_parameters(self, data, best_params, param_name, value):
        """Evaluate parameters and return performance metrics."""
        # Copy parameters and update the current parameter
        test_params = best_params.copy()
        test_params[param_name] = value

        # Create a new instance to avoid shared state issues
        strategy = self.__class__(**test_params)
        strategy.reset_state()
        strategy.run_strategy(data)
        
        performance = strategy.get_strategy_performance(data)
        pnl = performance['total_pnl']
        max_drawdown = performance['max_drawdown']
        win_rate = performance['win_rate']
        num_trades = performance['num_trades']

        if pnl <= 0 or max_drawdown >= 50 or num_trades < 5:
            return None

        normalized_pnl = np.log1p(max(0, pnl)) / np.log1p(10000)
        normalized_mdd = max(0, 1 - (max_drawdown / 100))
        normalized_wr = win_rate / 100

        combined_metric = (
            0.6 * normalized_pnl +
            0.3 * normalized_mdd +
            0.09 * normalized_wr +
            0.01 * (num_trades / 1000)
        )

        if combined_metric <= 0:
            return None

        return {
            'pnl': pnl,
            'mdd': max_drawdown,
            'combined_metric': combined_metric,
            'num_trades': num_trades,
            'win_rate': win_rate
        }

    def run_strategy(self, df: pd.DataFrame, isMetrics=False) -> pd.DataFrame:
        """Run the enhanced trading strategy with dynamic stops"""
        # Store dataframe and calculate indicators
        df = self.calculate_indicators(df)

        # Counter for printing trade details (for debugging)
        counter = 0

        isHighlow = HIGHLOW

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
        
                if take_profit_hit and take_profits[i-1] != 0:
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
                elif stop_hit and stop_losses[i-1] != 0:
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
