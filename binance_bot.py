from binance.client import Client
from binance.enums import *
from binance.um_futures import UMFutures
from binance.error import ClientError
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from strategy.multistrategy_manager import Multistrategy_manager
from config.config import API_KEY_FUTURES, API_SECRET_FUTURES, SYMBOL, MULTISTRAT_PARAMS, MULTISTRAT_PARAMS_2, MULTISTRAT_PARAMS_3, DEFAULT_PARAMS

class BinanceBot(Multistrategy_manager):
    def __init__(self):
        super().__init__()
        self.active_SL_order = None
        self.active_TP_order = None
        self.position_size = 0
        self.stop_loss = 0

    def account_balance(self, balance_client):
        try:
            balance = balance_client.balance()
            # Change 'BNB' to for example 'USDT' if using USDT as collateral
            for i in balance:
                if i['asset'] == 'BNB':
                    balance = float(i['balance'])
            # Convert BNB to USDT
            bnb_price = float(balance_client.mark_price('BNBUSDT')['markPrice'])
            balance *= bnb_price
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
            symbol_info = next(filter(lambda x: x['symbol'] == symbol, 
                                    client.futures_exchange_info()['symbols']))
            precision = int(symbol_info['quantityPrecision'])
            
            # Round to valid precision
            quantity = round(quantity, precision)

            print(f"Position quantity: {quantity}")
            return quantity
        except Exception as e:
            print(f"Quantity calculation error: {e}")
            return None

    def run_strategies(self, df: pd.DataFrame, client, balance_client):
        acc_bal = self.account_balance(balance_client)
        print(f"Account balance: ${acc_bal}")

        # Calculate indicators for each strategy
        dfs = [None] * len(self.strategies)
        for i, strategy in enumerate(self.strategies):
            strategy.last_df = df
            dfs[i] = strategy.calculate_indicators(df.copy())

        # Get current price and rows for each strategy
        current_price = df['close'].iloc[-1]
        current_rows = [df.copy().iloc[-1]] * len(self.strategies)
        for j, strategy in enumerate(self.strategies):
            current_rows[j] = dfs[j].iloc[-1]

        # Check if position has been closed
        if self.current_position != 0:
            positions_info = client.futures_position_information(symbol=SYMBOL)
            for pos in positions_info:
                if pos['symbol'] == SYMBOL and float(pos['positionAmt']) == 0:
                    self.current_position = 0
                    self.active_SL_order = None
                    self.active_TP_order = None
                    self.active_strategy = None
                    client.futures_cancel_all_open_orders(symbol=SYMBOL)
                    print("Position has been closed and all orders cancelled")

        # Calculate position size
        self.position_size = self.account_balance(balance_client) * self.leverage * 0.9

        # Check if a new position should be opened
        if self.current_position == 0:
            for i, strategy in enumerate(self.strategies):
                entry_signal = strategy.check_entry_automated(current_rows[i])
                if entry_signal != 0:
                    self.current_position = entry_signal
                    self.entry_price = current_price
                    self.active_strategy = i
                    self.take_profit_pct = strategy.take_profit_pct
                    self.stop_loss = strategy.calculate_dynamic_stop_loss(current_rows[i], entry_signal)
                    if self.current_position == 1:
                        quantity = self.calculate_position_quantity(client, SYMBOL, current_price, self.position_size)
                        self.create_order(client, SIDE_BUY, quantity, SYMBOL)
                        self.place_take_profit_order(client, SYMBOL, SIDE_SELL, quantity, round(self.entry_price * (1 + self.take_profit_pct), 1))
                    elif self.current_position == -1:
                        quantity = self.calculate_position_quantity(client, SYMBOL, current_price, self.position_size)
                        self.create_order(client, SIDE_SELL, quantity, SYMBOL)
                        self.place_take_profit_order(client, SYMBOL, SIDE_BUY, quantity, round(self.entry_price * (1 - self.take_profit_pct), 1))
                    break

        # Create or update stop loss order
        if self.current_position != 0:
            new_stop = self.strategies[self.active_strategy].calculate_dynamic_stop_loss(current_rows[self.active_strategy], self.current_position)
            if self.current_position == 1:
                self.stop_loss = max(self.stop_loss, new_stop)
                self.cancel_stop_loss_order(client, SYMBOL)
                quantity = self.calculate_position_quantity(client, SYMBOL, current_price, self.position_size)
                self.place_stop_loss_order(client, SYMBOL, SIDE_SELL, quantity, round(self.stop_loss, 1))
            else:
                self.stop_loss = min(self.stop_loss, new_stop)
                self.cancel_stop_loss_order(client, SYMBOL)
                quantity = self.calculate_position_quantity(client, SYMBOL, current_price, self.position_size)
                self.place_stop_loss_order(client, SYMBOL, SIDE_BUY, quantity, round(self.stop_loss, 1))


def fetch_historical_data(client, symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=100):
    # Fetch historical data for indicator calculation
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

def fetch_latest_price(client, symbol, interval=Client.KLINE_INTERVAL_1HOUR):
    # Fetch latest price
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=1)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

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

        client.futures_change_leverage(symbol=SYMBOL, leverage=DEFAULT_PARAMS['leverage'])

        df = fetch_historical_data(client, SYMBOL)
        if df is None or len(df) < 100:
            raise Exception("Failed to fetch initial data")

        while True:
            try:
                time.sleep(1)
                latest_df = fetch_latest_price(client, SYMBOL)
                if latest_df is None or len(latest_df) == 0:
                    raise Exception("Failed to fetch latest price")
                df = pd.concat([df.iloc[1:], latest_df])
                print("Running strategy @", (datetime.now() + timedelta(hours=2)))
                bot.run_strategies(df, client, balance_client)
                print("Strategy run, waiting for next hour\n")
                wait_until_next_hour()
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == '__main__':
    main()

