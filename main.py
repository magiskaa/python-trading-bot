from binance.client import Client
from strategy.optimize_parameters import Optimize_parameters
from strategy.optimize_time_frame import Optimize_time_frame
from strategy.strategy_utils import fetch_data, fetch_multi_timeframe_data, find_best_timeframe
from config.config import SYMBOL, BACKTEST_START, API_KEY, API_SECRET, DEFAULT_PARAMS

def run_parameter_optimization_strategy():
    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)

    data = fetch_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START)

    # Initialize strategy
    strategy = Optimize_parameters(
        starting_balance=DEFAULT_PARAMS['starting_balance'],
        leverage=DEFAULT_PARAMS['leverage']
    )

    # Initial parameters to start with
    initial_params = {
        #'leverage': 10,
        'macd_fast_period': 12,
        'macd_slow_period': 26,
        'macd_signal_period': 9,
        #'mfi_period': 14,
        'obv_ma_period': 10,
        'bb_period': 20,
        'bb_std': 1.7,
        'adx_period': 25,
        'adx_threshold': 29,
        'rsi_period': 21,
        'rsi_overbought': 75,
        'rsi_oversold': 30,
        'take_profit_pct': 0.03,
        'atr_period': 22,
        'atr_multiplier': 1.8,
        #'keltner_period': 21,
        #'keltner_atr_factor': 1.6,
        #'hma_period': 20,
        #'vwap_std': 2.0,
    }

    strategy.optimize_step_by_step(data, initial_params)
    print("\nOptimization done")

    #strategy.run_strategy(data)
    #strategy.calculate_metrics(data)
    #strategy.plot_results(data)

def run_timeframe_optimization_strategy():
    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)

    # Initialize strategy
    strategy = Optimize_time_frame(
        starting_balance=DEFAULT_PARAMS['starting_balance'],
        leverage=DEFAULT_PARAMS['leverage']
    )
    
    # Initial parameters to start with
    initial_params = {
        'bb_period': 20,
        'bb_std': 2.0,
        'adx_period': 20,
        'adx_threshold': 25,
        'rsi_period': 20,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'take_profit_pct': 0.05,
        'atr_period': 14,
        'atr_multiplier': 2.0
    }

    # First optimize on 1-hour timeframe
    #print("Fetching 1-hour data for initial optimization...")
    hour_data = fetch_multi_timeframe_data(
        client, 
        SYMBOL, 
        BACKTEST_START,  # Start with recent data for faster testing
        [Client.KLINE_INTERVAL_1HOUR]
    )[Client.KLINE_INTERVAL_1HOUR]

    # Find optimal parameters
    print("\nOptimizing parameters...")
    best_params = strategy.optimize_step_by_step(hour_data, initial_params)
    
    # Fetch data for different timeframes
    timeframes = [
        Client.KLINE_INTERVAL_15MINUTE,
        Client.KLINE_INTERVAL_30MINUTE,
        Client.KLINE_INTERVAL_1HOUR
    ]
    
    # Test on different timeframes
    #print("\nFetching multi-timeframe data...")
    timeframe_data = fetch_multi_timeframe_data(
        client,
        SYMBOL,
        "3 months ago UTC",
        timeframes
    )
    strategy.run_strategy(hour_data)
    strategy.calculate_metrics(hour_data)
    strategy.plot_results(hour_data)
    
    # Find best timeframe
    find_best_timeframe(strategy, timeframe_data, best_params)


if __name__ == '__main__':
    run_parameter_optimization_strategy()
    #run_timeframe_optimization_strategy()