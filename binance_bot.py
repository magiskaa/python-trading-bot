from binance.client import Client
from binance.enums import *
from binance.um_futures import UMFutures
from binance.error import ClientError
import time
import os
import json
from datetime import datetime, timedelta
import pandas as pd
from strategy.multistrategy_manager import Multistrategy_manager
from config.config import (
    API_KEY_FUTURES, API_SECRET_FUTURES, SYMBOL, MULTISTRAT_PARAMS, MULTISTRAT_PARAMS_2, 
    MULTISTRAT_PARAMS_3, MULTISTRAT_PARAMS_4, MULTISTRAT_PARAMS_5, MULTISTRAT_PARAMS_6, 
    MULTISTRAT_PARAMS_7, MULTISTRAT_PARAMS_8, MULTISTRAT_PARAMS_9, MULTISTRAT_PARAMS_10,
    MULTISTRAT_PARAMS_11, DEFAULT_PARAMS, BACKTEST_START
)

class BinanceBot(Multistrategy_manager):
    def __init__(self):
        super().__init__()
        self.active_SL_order = None
        self.active_TP_order = None
        self.position_size = 0
        self.stop_loss = 0
        self.quantity = 0
        self.initial = False

    def account_balance(self, balance_client):
        try:
            balance = balance_client.balance()
            for i in balance:
                if i['asset'] == 'BNFCR':
                    balance = float(i['balance'])
            
            return round(balance, 2)
        except ClientError as e:
            print(f"Balance error: {e}")
            return None
    
    def create_order(self, client, side, quantity, symbol, order_type=FUTURE_ORDER_TYPE_MARKET):
        try:
            # Check minimum order size
            min_qty = float(client.futures_exchange_info()['symbols'][0]['filters'][2]['minQty'])
            if quantity < min_qty:
                print(f"Order quantity {quantity} below minimum {min_qty}")
                return None
            # Create order
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
            print(f"Order created: {symbol} {side} {order_type} {quantity}")
            return order
        except Exception as e:
            print(f"Order creation error: {e}")
            return None

    def place_stop_loss_order(self, client, symbol, side, quantity, stop_price, order_type=FUTURE_ORDER_TYPE_STOP_MARKET):
        try:
            # Create stop loss order
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                stopPrice=stop_price,
                closePosition=True
            )
            self.active_SL_order = order['orderId']
            print(f"Stop loss order created: {symbol} {side} {order_type} {quantity} {stop_price}")
            return order
        except Exception as e:
            print(f"Stop loss error: {e}")
            return None
        
    def cancel_stop_loss_order(self, client, symbol):
        # Cancel stop loss order if exists
        if self.active_SL_order is not None:
            try:
                client.futures_cancel_order(symbol=symbol, orderId=self.active_SL_order)
                self.active_SL_order = None
                print("Stop loss order cancelled")
            except Exception as e:
                print(f"Stop loss cancel error: {e}")
        
    def place_take_profit_order(self, client, symbol, side, quantity, price, order_type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET):
        # Create take profit order
        try:
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                stopPrice=price,
                closePosition=True
            )
            self.active_TP_order = order['orderId']
            print(f"Take profit order created: {symbol} {side} {order_type} {quantity} {price}")
            return order
        except Exception as e:
            print(f"Take profit error: {e}")
            return None
        
    def calculate_position_quantity(self, client, symbol, current_price, usd_size):
        try:
            # Calculate quantity in base asset
            quantity = usd_size / current_price
            
            # Get symbol info for precision
            symbol_info = next(
                filter(lambda x: x['symbol'] == symbol, 
                client.futures_exchange_info()['symbols'])
            )
            precision = int(symbol_info['quantityPrecision'])
            
            # Round to valid precision
            quantity = round(quantity, precision)

            print(f"Position quantity: {quantity}")
            return quantity
        except Exception as e:
            print(f"Quantity calculation error: {e}")
            return None

    def run_strategies(self, df: pd.DataFrame, client, balance_client, data):
        acc_bal = self.account_balance(balance_client)
        print(f"Account balance: ${acc_bal}")
        self.initial = False

        # Calculate indicators for each strategy
        dfs = [None] * len(self.strategies)
        for i, strategy in enumerate(self.strategies):
            strategy.last_df = df
            dfs[i] = strategy.calculate_indicators(df.copy())

        # Get current price and rows for each strategy
        current_price = df['close'].iloc[-1]
        print(f"{SYMBOL} price: ${current_price}\n")
        current_rows = [df.copy().iloc[-1]] * len(self.strategies)
        for j, strategy in enumerate(self.strategies):
            current_rows[j] = dfs[j].iloc[-1]

        # Check if position has been closed
        if self.current_position != 0:
            sl_hit = client.futures_get_order(symbol=SYMBOL, orderId=self.active_SL_order)
            tp_hit = client.futures_get_order(symbol=SYMBOL, orderId=self.active_TP_order)
            if sl_hit["status"] == "FILLED" or tp_hit["status"] == "FILLED":
                self.current_position = 0
                self.quantity = 0
                self.active_SL_order = None
                self.active_TP_order = None
                self.active_strategy = None
                client.futures_cancel_all_open_orders(symbol=SYMBOL)
                with open('data/trade_details.json', 'w') as f:
                    f.write("{}")
                print("Position has been closed and all orders cancelled")

        # Calculate position size
        if self.position_size < 50000:
            self.position_size = acc_bal * self.leverage * 0.9
        else:
            self.position_size = 50000

        # Check if a new position should be opened
        if self.current_position == 0:
            for i, strategy in enumerate(self.strategies):
                entry_signal = strategy.check_entry_automated(current_rows[i])
                print(f"Entry signal: {entry_signal}")
                if entry_signal != 0:
                    self.current_position = entry_signal
                    self.entry_price = current_price
                    self.active_strategy = i
                    self.take_profit_pct = strategy.take_profit_pct
                    if self.active_strategy in [0, 2, 8, 9, 10]:
                        self.stop_loss = strategy.calculate_dynamic_stop_loss_highlow(current_rows[i], self.current_position)
                    else:
                        self.stop_loss = strategy.calculate_dynamic_stop_loss(current_rows[i], self.current_position)
                    if self.current_position == 1:
                        self.quantity = self.calculate_position_quantity(client, SYMBOL, current_price, self.position_size)
                        self.create_order(client, SIDE_BUY, self.quantity, SYMBOL)
                        self.place_take_profit_order(client, SYMBOL, SIDE_SELL, self.quantity, round(self.entry_price * (1 + self.take_profit_pct), 1))
                        self.place_stop_loss_order(client, SYMBOL, SIDE_SELL, self.quantity, round(self.stop_loss, 1))
                    elif self.current_position == -1:
                        self.quantity = self.calculate_position_quantity(client, SYMBOL, current_price, self.position_size)
                        self.create_order(client, SIDE_SELL, self.quantity, SYMBOL)
                        self.place_take_profit_order(client, SYMBOL, SIDE_BUY, self.quantity, round(self.entry_price * (1 - self.take_profit_pct), 1))
                        self.place_stop_loss_order(client, SYMBOL, SIDE_BUY, self.quantity, round(self.stop_loss, 1))
                    self.initial = True
                    data["trade_details"] = {
                        "current_position": self.current_position,
                        "entry_price": self.entry_price,
                        "active_strategy": self.active_strategy,
                        "take_profit_pct": self.take_profit_pct,
                        "stop_loss": self.stop_loss,
                        "quantity": self.quantity,
                        "active_SL_order": self.active_SL_order,
                        "active_TP_order": self.active_TP_order
                    }
                    with open('data/trade_details.json', 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"Trade details saved: {data['trade_details']}")
                    break

        # Create or update stop loss order
        if self.current_position != 0 and self.initial == False:
            if self.active_strategy in [0, 2, 8, 9, 10]:
                new_stop = self.strategies[self.active_strategy].calculate_dynamic_stop_loss_highlow(current_rows[self.active_strategy], self.current_position)
            else:
                new_stop = self.strategies[self.active_strategy].calculate_dynamic_stop_loss(current_rows[self.active_strategy], self.current_position)
            if self.current_position == 1 and new_stop > self.stop_loss:
                self.stop_loss = new_stop
                self.cancel_stop_loss_order(client, SYMBOL)
                self.place_stop_loss_order(client, SYMBOL, SIDE_SELL, self.quantity, round(new_stop, 1))
            elif self.current_position == -1 and new_stop < self.stop_loss:
                self.stop_loss = new_stop
                self.cancel_stop_loss_order(client, SYMBOL)
                self.place_stop_loss_order(client, SYMBOL, SIDE_BUY, self.quantity, round(new_stop, 1))

def fetch_and_store_data(client, symbol, interval, start_str, filename):
    # Try loading existing data
    try:
        existing_df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        last_saved = existing_df.index[-1]
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        last_saved = None

    # Fetch new data starting from the last stored timestamp
    if last_saved:
        klines = client.get_historical_klines(symbol, interval, str(last_saved))
    else:
        klines = client.get_historical_klines(symbol, interval, start_str)

    new_df = pd.DataFrame(klines, columns=[
        'timestamp','open','high','low','close','volume','close_time',
        'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume','ignore'
    ])
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
    new_df.set_index('timestamp', inplace=True)
    new_df = new_df[['open','high','low','close','volume']].astype(float)

    # Merge new data with existing data and store
    combined_df = pd.concat([existing_df, new_df])
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
    combined_df.to_csv(filename)

    return combined_df

def wait_until_next_hour():
    next_hour = (datetime.now() + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
    time_to_wait = (next_hour - datetime.now()).total_seconds()
    time.sleep(time_to_wait)

def main():
    try:
        client = Client(API_KEY_FUTURES, API_SECRET_FUTURES, tld='com')
        balance_client = UMFutures(API_KEY_FUTURES, API_SECRET_FUTURES)
        bot = BinanceBot()

        # Test connection first
        client.futures_ping()

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
        bot.add_strategy(
            bb_period=MULTISTRAT_PARAMS_4['bb_period'],
            bb_std=MULTISTRAT_PARAMS_4['bb_std'],
            adx_period=MULTISTRAT_PARAMS_4['adx_period'],
            adx_threshold=MULTISTRAT_PARAMS_4['adx_threshold'],
            rsi_period=MULTISTRAT_PARAMS_4['rsi_period'],
            rsi_overbought=MULTISTRAT_PARAMS_4['rsi_overbought'],
            rsi_oversold=MULTISTRAT_PARAMS_4['rsi_oversold'],
            stop_loss_pct=MULTISTRAT_PARAMS_4['stop_loss_pct'],
            take_profit_pct=MULTISTRAT_PARAMS_4['take_profit_pct'],
            atr_period=MULTISTRAT_PARAMS_4['atr_period'],
            atr_multiplier=MULTISTRAT_PARAMS_4['atr_multiplier'],
            keltner_period=MULTISTRAT_PARAMS_4['keltner_period'],
            keltner_atr_factor=MULTISTRAT_PARAMS_4['keltner_atr_factor'],
            hma_period=MULTISTRAT_PARAMS_4['hma_period'],
            vwap_std=MULTISTRAT_PARAMS_4['vwap_std'],
            macd_fast_period=MULTISTRAT_PARAMS_4['macd_fast_period'],
            macd_slow_period=MULTISTRAT_PARAMS_4['macd_slow_period'],
            macd_signal_period=MULTISTRAT_PARAMS_4['macd_signal_period'],
            mfi_period=MULTISTRAT_PARAMS_4['mfi_period'],
            obv_ma_period=MULTISTRAT_PARAMS_4['obv_ma_period']
        )
        bot.add_strategy(
            bb_period=MULTISTRAT_PARAMS_5['bb_period'],
            bb_std=MULTISTRAT_PARAMS_5['bb_std'],
            adx_period=MULTISTRAT_PARAMS_5['adx_period'],
            adx_threshold=MULTISTRAT_PARAMS_5['adx_threshold'],
            rsi_period=MULTISTRAT_PARAMS_5['rsi_period'],
            rsi_overbought=MULTISTRAT_PARAMS_5['rsi_overbought'],
            rsi_oversold=MULTISTRAT_PARAMS_5['rsi_oversold'],
            stop_loss_pct=MULTISTRAT_PARAMS_5['stop_loss_pct'],
            take_profit_pct=MULTISTRAT_PARAMS_5['take_profit_pct'],
            atr_period=MULTISTRAT_PARAMS_5['atr_period'],
            atr_multiplier=MULTISTRAT_PARAMS_5['atr_multiplier'],
            keltner_period=MULTISTRAT_PARAMS_5['keltner_period'],
            keltner_atr_factor=MULTISTRAT_PARAMS_5['keltner_atr_factor'],
            hma_period=MULTISTRAT_PARAMS_5['hma_period'],
            vwap_std=MULTISTRAT_PARAMS_5['vwap_std'],
            macd_fast_period=MULTISTRAT_PARAMS_5['macd_fast_period'],
            macd_slow_period=MULTISTRAT_PARAMS_5['macd_slow_period'],
            macd_signal_period=MULTISTRAT_PARAMS_5['macd_signal_period'],
            mfi_period=MULTISTRAT_PARAMS_5['mfi_period'],
            obv_ma_period=MULTISTRAT_PARAMS_5['obv_ma_period']
        )
        bot.add_strategy(
            bb_period=MULTISTRAT_PARAMS_6['bb_period'],
            bb_std=MULTISTRAT_PARAMS_6['bb_std'],
            adx_period=MULTISTRAT_PARAMS_6['adx_period'],
            adx_threshold=MULTISTRAT_PARAMS_6['adx_threshold'],
            rsi_period=MULTISTRAT_PARAMS_6['rsi_period'],
            rsi_overbought=MULTISTRAT_PARAMS_6['rsi_overbought'],
            rsi_oversold=MULTISTRAT_PARAMS_6['rsi_oversold'],
            stop_loss_pct=MULTISTRAT_PARAMS_6['stop_loss_pct'],
            take_profit_pct=MULTISTRAT_PARAMS_6['take_profit_pct'],
            atr_period=MULTISTRAT_PARAMS_6['atr_period'],
            atr_multiplier=MULTISTRAT_PARAMS_6['atr_multiplier'],
            keltner_period=MULTISTRAT_PARAMS_6['keltner_period'],
            keltner_atr_factor=MULTISTRAT_PARAMS_6['keltner_atr_factor'],
            hma_period=MULTISTRAT_PARAMS_6['hma_period'],
            vwap_std=MULTISTRAT_PARAMS_6['vwap_std'],
            macd_fast_period=MULTISTRAT_PARAMS_6['macd_fast_period'],
            macd_slow_period=MULTISTRAT_PARAMS_6['macd_slow_period'],
            macd_signal_period=MULTISTRAT_PARAMS_6['macd_signal_period'],
            mfi_period=MULTISTRAT_PARAMS_6['mfi_period'],
            obv_ma_period=MULTISTRAT_PARAMS_6['obv_ma_period']
        )
        bot.add_strategy(
            bb_period=MULTISTRAT_PARAMS_7['bb_period'],
            bb_std=MULTISTRAT_PARAMS_7['bb_std'],
            adx_period=MULTISTRAT_PARAMS_7['adx_period'],
            adx_threshold=MULTISTRAT_PARAMS_7['adx_threshold'],
            rsi_period=MULTISTRAT_PARAMS_7['rsi_period'],
            rsi_overbought=MULTISTRAT_PARAMS_7['rsi_overbought'],
            rsi_oversold=MULTISTRAT_PARAMS_7['rsi_oversold'],
            stop_loss_pct=MULTISTRAT_PARAMS_7['stop_loss_pct'],
            take_profit_pct=MULTISTRAT_PARAMS_7['take_profit_pct'],
            atr_period=MULTISTRAT_PARAMS_7['atr_period'],
            atr_multiplier=MULTISTRAT_PARAMS_7['atr_multiplier'],
            keltner_period=MULTISTRAT_PARAMS_7['keltner_period'],
            keltner_atr_factor=MULTISTRAT_PARAMS_7['keltner_atr_factor'],
            hma_period=MULTISTRAT_PARAMS_7['hma_period'],
            vwap_std=MULTISTRAT_PARAMS_7['vwap_std'],
            macd_fast_period=MULTISTRAT_PARAMS_7['macd_fast_period'],
            macd_slow_period=MULTISTRAT_PARAMS_7['macd_slow_period'],
            macd_signal_period=MULTISTRAT_PARAMS_7['macd_signal_period'],
            mfi_period=MULTISTRAT_PARAMS_7['mfi_period'],
            obv_ma_period=MULTISTRAT_PARAMS_7['obv_ma_period']
        )
        bot.add_strategy(
            bb_period=MULTISTRAT_PARAMS_8['bb_period'],
            bb_std=MULTISTRAT_PARAMS_8['bb_std'],
            adx_period=MULTISTRAT_PARAMS_8['adx_period'],
            adx_threshold=MULTISTRAT_PARAMS_8['adx_threshold'],
            rsi_period=MULTISTRAT_PARAMS_8['rsi_period'],
            rsi_overbought=MULTISTRAT_PARAMS_8['rsi_overbought'],
            rsi_oversold=MULTISTRAT_PARAMS_8['rsi_oversold'],
            stop_loss_pct=MULTISTRAT_PARAMS_8['stop_loss_pct'],
            take_profit_pct=MULTISTRAT_PARAMS_8['take_profit_pct'],
            atr_period=MULTISTRAT_PARAMS_8['atr_period'],
            atr_multiplier=MULTISTRAT_PARAMS_8['atr_multiplier'],
            keltner_period=MULTISTRAT_PARAMS_8['keltner_period'],
            keltner_atr_factor=MULTISTRAT_PARAMS_8['keltner_atr_factor'],
            hma_period=MULTISTRAT_PARAMS_8['hma_period'],
            vwap_std=MULTISTRAT_PARAMS_8['vwap_std'],
            macd_fast_period=MULTISTRAT_PARAMS_8['macd_fast_period'],
            macd_slow_period=MULTISTRAT_PARAMS_8['macd_slow_period'],
            macd_signal_period=MULTISTRAT_PARAMS_8['macd_signal_period'],
            mfi_period=MULTISTRAT_PARAMS_8['mfi_period'],
            obv_ma_period=MULTISTRAT_PARAMS_8['obv_ma_period']
        )
        bot.add_strategy(
            bb_period=MULTISTRAT_PARAMS_9['bb_period'],
            bb_std=MULTISTRAT_PARAMS_9['bb_std'],
            adx_period=MULTISTRAT_PARAMS_9['adx_period'],
            adx_threshold=MULTISTRAT_PARAMS_9['adx_threshold'],
            rsi_period=MULTISTRAT_PARAMS_9['rsi_period'],
            rsi_overbought=MULTISTRAT_PARAMS_9['rsi_overbought'],
            rsi_oversold=MULTISTRAT_PARAMS_9['rsi_oversold'],
            stop_loss_pct=MULTISTRAT_PARAMS_9['stop_loss_pct'],
            take_profit_pct=MULTISTRAT_PARAMS_9['take_profit_pct'],
            atr_period=MULTISTRAT_PARAMS_9['atr_period'],
            atr_multiplier=MULTISTRAT_PARAMS_9['atr_multiplier'],
            keltner_period=MULTISTRAT_PARAMS_9['keltner_period'],
            keltner_atr_factor=MULTISTRAT_PARAMS_9['keltner_atr_factor'],
            hma_period=MULTISTRAT_PARAMS_9['hma_period'],
            vwap_std=MULTISTRAT_PARAMS_9['vwap_std'],
            macd_fast_period=MULTISTRAT_PARAMS_9['macd_fast_period'],
            macd_slow_period=MULTISTRAT_PARAMS_9['macd_slow_period'],
            macd_signal_period=MULTISTRAT_PARAMS_9['macd_signal_period'],
            mfi_period=MULTISTRAT_PARAMS_9['mfi_period'],
            obv_ma_period=MULTISTRAT_PARAMS_9['obv_ma_period']
        )
        bot.add_strategy(
            bb_period=MULTISTRAT_PARAMS_10['bb_period'],
            bb_std=MULTISTRAT_PARAMS_10['bb_std'],
            adx_period=MULTISTRAT_PARAMS_10['adx_period'],
            adx_threshold=MULTISTRAT_PARAMS_10['adx_threshold'],
            rsi_period=MULTISTRAT_PARAMS_10['rsi_period'],
            rsi_overbought=MULTISTRAT_PARAMS_10['rsi_overbought'],
            rsi_oversold=MULTISTRAT_PARAMS_10['rsi_oversold'],
            stop_loss_pct=MULTISTRAT_PARAMS_10['stop_loss_pct'],
            take_profit_pct=MULTISTRAT_PARAMS_10['take_profit_pct'],
            atr_period=MULTISTRAT_PARAMS_10['atr_period'],
            atr_multiplier=MULTISTRAT_PARAMS_10['atr_multiplier'],
            keltner_period=MULTISTRAT_PARAMS_10['keltner_period'],
            keltner_atr_factor=MULTISTRAT_PARAMS_10['keltner_atr_factor'],
            hma_period=MULTISTRAT_PARAMS_10['hma_period'],
            vwap_std=MULTISTRAT_PARAMS_10['vwap_std'],
            macd_fast_period=MULTISTRAT_PARAMS_10['macd_fast_period'],
            macd_slow_period=MULTISTRAT_PARAMS_10['macd_slow_period'],
            macd_signal_period=MULTISTRAT_PARAMS_10['macd_signal_period'],
            mfi_period=MULTISTRAT_PARAMS_10['mfi_period'],
            obv_ma_period=MULTISTRAT_PARAMS_10['obv_ma_period']
        )
        bot.add_strategy(
            bb_period=MULTISTRAT_PARAMS_11['bb_period'],
            bb_std=MULTISTRAT_PARAMS_11['bb_std'],
            adx_period=MULTISTRAT_PARAMS_11['adx_period'],
            adx_threshold=MULTISTRAT_PARAMS_11['adx_threshold'],
            rsi_period=MULTISTRAT_PARAMS_11['rsi_period'],
            rsi_overbought=MULTISTRAT_PARAMS_11['rsi_overbought'],
            rsi_oversold=MULTISTRAT_PARAMS_11['rsi_oversold'],
            stop_loss_pct=MULTISTRAT_PARAMS_11['stop_loss_pct'],
            take_profit_pct=MULTISTRAT_PARAMS_11['take_profit_pct'],
            atr_period=MULTISTRAT_PARAMS_11['atr_period'],
            atr_multiplier=MULTISTRAT_PARAMS_11['atr_multiplier'],
            keltner_period=MULTISTRAT_PARAMS_11['keltner_period'],
            keltner_atr_factor=MULTISTRAT_PARAMS_11['keltner_atr_factor'],
            hma_period=MULTISTRAT_PARAMS_11['hma_period'],
            vwap_std=MULTISTRAT_PARAMS_11['vwap_std'],
            macd_fast_period=MULTISTRAT_PARAMS_11['macd_fast_period'],
            macd_slow_period=MULTISTRAT_PARAMS_11['macd_slow_period'],
            macd_signal_period=MULTISTRAT_PARAMS_11['macd_signal_period'],
            mfi_period=MULTISTRAT_PARAMS_11['mfi_period'],
            obv_ma_period=MULTISTRAT_PARAMS_11['obv_ma_period']
        )

        client.futures_change_leverage(symbol=SYMBOL, leverage=DEFAULT_PARAMS['leverage'])

        df = fetch_and_store_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START, 'data/symbol_data.csv')

        string = input("Is a trade open? (y/n): ")
        data = {}
        if string == 'y':
            FILE_PATH = 'data/trade_details.json'
            if os.path.exists(FILE_PATH):
                with open(FILE_PATH, 'r') as f:
                    data = json.load(f)

                details = data["trade_details"]

                bot.current_position = details["current_position"]
                bot.entry_price = details["entry_price"]
                bot.active_strategy = details["active_strategy"]
                bot.take_profit_pct = details["take_profit_pct"]
                bot.stop_loss = details["stop_loss"]
                bot.quantity = details["quantity"]
                bot.active_SL_order = details["active_SL_order"]
                bot.active_TP_order = details["active_TP_order"]
            else:
                print("Trade details file not found.")

        while True:
            try:
                time.sleep(1)
                df = fetch_and_store_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START, 'data/symbol_data.csv')
                print("Running strategy @", (datetime.now() + timedelta(hours=3)))
                bot.run_strategies(df, client, balance_client, data)
                print("Strategy run, waiting for next hour\n")
                wait_until_next_hour()
            except Exception as e:
                print(f"Error in main loop: {e}")
                print("Retrying in 60s")
                time.sleep(60)  # Wait before retrying
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == '__main__':
    main()

