import optuna
from binance.client import Client
import numpy as np
from strategy.optimize_parameters import Optimize_parameters
from strategy.optimize_time_frame import Optimize_time_frame
from strategy.multistrategy_manager import Multistrategy_manager
from strategy.strategy_utils import fetch_data, fetch_multi_timeframe_data, find_best_timeframe
from config.config import SYMBOL, BACKTEST_START, API_KEY, API_SECRET, DEFAULT_PARAMS, MULTISTRAT_PARAMS, MULTISTRAT_PARAMS_2
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
        'bb_period': 20,
        'bb_std': 2.5,
        'adx_period': 20,
        'adx_threshold': 25,
        'rsi_period': 20,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'atr_period': 20,
        'atr_multiplier': 2.5,
        'keltner_period': 20,
        'keltner_atr_factor': 2.0,
        'hma_period': 20,
        'vwap_std': 2.0,
        'macd_fast_period': 12,
        'macd_slow_period': 26,
        'macd_signal_period': 9,
        'mfi_period': 14,
        'obv_ma_period': 14,
    }

    #strategy.optimize_step_by_step(data, initial_params)
    print("\nOptimization done")

    strategy.run_strategy(data, False)
    strategy.calculate_metrics(data)
    strategy.plot_results(data)

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
    strategy.run_strategy(hour_data, False)
    strategy.calculate_metrics(hour_data)
    strategy.plot_results(hour_data)
    
    # Find best timeframe
    find_best_timeframe(strategy, timeframe_data, best_params)

def automated_optimization():
    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)
    global data
    data = fetch_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START)

    # Run optimization
    storage = 'sqlite:///data/parameter_optimization.db'
    study_name = 'parameter_optimization'

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize', 
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=500)

    # Print results
    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)

def objective(trial):
    # Indicators to use
    use_bb = trial.suggest_categorical('use_bb', [True, False])
    use_adx = trial.suggest_categorical('use_adx', [True, False])
    use_rsi = trial.suggest_categorical('use_rsi', [True, False])
    use_keltner = trial.suggest_categorical('use_keltner', [True, False])
    use_hma = trial.suggest_categorical('use_hma', [True, False])
    use_vwap = trial.suggest_categorical('use_vwap', [True, False])
    use_macd = trial.suggest_categorical('use_macd', [True, False])
    use_mfi = trial.suggest_categorical('use_mfi', [True, False])
    use_obv = trial.suggest_categorical('use_obv', [True, False])

    # Initialize strategy
    strategy = Optimize_parameters()

    # Set common parameters
    strategy.stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.001, 0.1)
    strategy.take_profit_pct = trial.suggest_float('take_profit_pct', 0.01, 0.15)
    strategy.atr_period = trial.suggest_int('atr_period', 1, 50)
    strategy.atr_multiplier = trial.suggest_float('atr_multiplier', 0.1, 3.5)

    # Conditional parameter definitions
    if use_bb:
        strategy.bb_period = trial.suggest_int('bb_period', 1, 40)
        strategy.bb_std = trial.suggest_float('bb_std', 0.1, 3.5)
    else:
        strategy.bb_period = None
        strategy.bb_std = None
    
    if use_adx:
        strategy.adx_period = trial.suggest_int('adx_period', 1, 60)
        strategy.adx_threshold = trial.suggest_int('adx_threshold', 1, 60)
    else:  
        strategy.adx_period = None
        strategy.adx_threshold = None

    if use_rsi:
        strategy.rsi_period = trial.suggest_int('rsi_period', 1, 50)
        strategy.rsi_overbought = trial.suggest_int('rsi_overbought', 55, 95)
        strategy.rsi_oversold = trial.suggest_int('rsi_oversold', 5, 45)
    else:
        strategy.rsi_period = None
        strategy.rsi_overbought = None
        strategy.rsi_oversold = None

    if use_keltner:
        strategy.keltner_period = trial.suggest_int('keltner_period', 1, 40)
        strategy.keltner_atr_factor = trial.suggest_float('keltner_atr_factor', 0.1, 3.5)
    else:
        strategy.keltner_period = None
        strategy.keltner_atr_factor = None

    if use_hma:
        strategy.hma_period = trial.suggest_int('hma_period', 1, 35)
    else:
        strategy.hma_period = None

    if use_vwap:
        strategy.vwap_std = trial.suggest_float('vwap_std', 0.1, 3.5)
    else:
        strategy.vwap_std = None

    if use_macd:
        strategy.macd_fast_period = trial.suggest_int('macd_fast_period', 1, 50)
        strategy.macd_slow_period = trial.suggest_int('macd_slow_period', 1, 50)
        strategy.macd_signal_period = trial.suggest_int('macd_signal_period', 1, 50)
    else:
        strategy.macd_fast_period = None
        strategy.macd_slow_period = None
        strategy.macd_signal_period = None

    if use_mfi:
        strategy.mfi_period = trial.suggest_int('mfi_period', 1, 40)
    else:
        strategy.mfi_period = None

    if use_obv:
        strategy.obv_ma_period = trial.suggest_int('obv_ma_period', 1, 50)
    else:
        strategy.obv_ma_period = None

    # Run strategy
    strategy.reset_state()
    strategy.run_strategy(data, True)

    performance = strategy.get_strategy_performance(data)
    pnl = performance['total_pnl']
    max_drawdown = performance['max_drawdown']

    print("\nPnL: ", pnl)
    print("Max Drawdown: ", max_drawdown)

    # Calculate combined metric
    weight_pnl = 0.5
    weight_mdd = 0.5

    # Logarithmic normalization for pnl
    normalized_pnl = np.log1p(max(0, pnl)) / np.log1p(10000)  # Adjust 10000 based on expected PnL range
    
    normalized_mdd = 1 - max_drawdown / 100

    combined_metric = (
        weight_pnl * normalized_pnl +
        weight_mdd * normalized_mdd)
    
    return combined_metric

def run_multistrategy_optimization():
    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)
    data = fetch_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START)

    # Initialize strategies
    manager = Multistrategy_manager()

    # Add strategies
    manager.add_strategy(
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

    manager.add_strategy(
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

    # Run strategies
    manager.run_strategies(data)
    manager.calculate_metrics()

    

if __name__ == '__main__':
    #run_parameter_optimization_strategy()
    #run_timeframe_optimization_strategy()
    #automated_optimization()
    run_multistrategy_optimization()