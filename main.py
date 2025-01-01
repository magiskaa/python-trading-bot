import optuna
from binance.client import Client
import numpy as np
import pandas as pd
from strategy.optimize_parameters import Optimize_parameters
from strategy.multistrategy_manager import Multistrategy_manager
from config.config import SYMBOL, BACKTEST_START, API_KEY, API_SECRET, DEFAULT_PARAMS, OPTIMIZE_PARAMS, MULTISTRAT_PARAMS, MULTISTRAT_PARAMS_2, MULTISTRAT_PARAMS_3
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def fetch_data(client, symbol, interval, start_str):
    klines = client.get_historical_klines(symbol, interval, start_str)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                        'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def run_parameter_optimization_strategy(mode):
    print("\nFetching data...\n")

    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)

    data = fetch_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START)

    print("Data fetched\n")

    # Initialize strategy
    strategy = Optimize_parameters(
        starting_balance=DEFAULT_PARAMS['starting_balance'],
        leverage=DEFAULT_PARAMS['leverage']
    )

    # Initial parameters to start with (change in config.py)
    initial_params = {
        'bb_period': OPTIMIZE_PARAMS['bb_period'],
        'bb_std': OPTIMIZE_PARAMS['bb_std'],
        'adx_period': OPTIMIZE_PARAMS['adx_period'],
        'adx_threshold': OPTIMIZE_PARAMS['adx_threshold'],
        'rsi_period': OPTIMIZE_PARAMS['rsi_period'],
        'rsi_overbought': OPTIMIZE_PARAMS['rsi_overbought'],
        'rsi_oversold': OPTIMIZE_PARAMS['rsi_oversold'],
        'stop_loss_pct': OPTIMIZE_PARAMS['stop_loss_pct'],
        'take_profit_pct': OPTIMIZE_PARAMS['take_profit_pct'],
        'atr_period': OPTIMIZE_PARAMS['atr_period'],
        'atr_multiplier': OPTIMIZE_PARAMS['atr_multiplier'],
        'keltner_period': OPTIMIZE_PARAMS['keltner_period'],
        'keltner_atr_factor': OPTIMIZE_PARAMS['keltner_atr_factor'],
        'hma_period': OPTIMIZE_PARAMS['hma_period'],
        'vwap_std': OPTIMIZE_PARAMS['vwap_std'],
        'macd_fast_period': OPTIMIZE_PARAMS['macd_fast_period'],
        'macd_slow_period': OPTIMIZE_PARAMS['macd_slow_period'],
        'macd_signal_period': OPTIMIZE_PARAMS['macd_signal_period'],
        'mfi_period': OPTIMIZE_PARAMS['mfi_period'],
        'obv_ma_period': OPTIMIZE_PARAMS['obv_ma_period'],
    }

    # Choose weather to optimize the parameters (1) or run the strategy, print the metrics and plot the results (2)
    # Case 2 uses the DEFAULT_PARAMS from config.py
    optimize = mode
    match optimize:
        case 1:
            print("Starting optimization...")
            strategy.optimize_step_by_step(data, initial_params)
            print("\nOptimization done\n")
        case 2:
            print("Calculating metrics...")
            strategy.run_strategy(data)
            strategy.calculate_metrics(data)
            strategy.plot_results(data)
            print("\nMetrics calculated\n")

def automated_optimization(trials):
    print("\nFetching data...\n")
    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)
    global data
    data = fetch_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START)

    print("Data fetched\n")

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

    print("Starting optimization...\n")

    study.optimize(objective, n_trials=trials)

    # Print results
    print("\nBest parameters: ", study.best_params)
    print("Best value: ", study.best_value, "\n")

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
    strategy.take_profit_pct = trial.suggest_float('take_profit_pct', 0.001, 0.2)
    strategy.atr_period = trial.suggest_int('atr_period', 1, 60)
    strategy.atr_multiplier = trial.suggest_float('atr_multiplier', 0.05, 4.0)

    # Conditional parameter definitions
    if use_bb:
        strategy.bb_period = trial.suggest_int('bb_period', 1, 50)
        strategy.bb_std = trial.suggest_float('bb_std', 0.05, 3.5)
    else:
        strategy.bb_period = None
        strategy.bb_std = None
    
    if use_adx:
        strategy.adx_period = trial.suggest_int('adx_period', 1, 70)
        strategy.adx_threshold = trial.suggest_int('adx_threshold', 1, 70)
    else:  
        strategy.adx_period = None
        strategy.adx_threshold = None

    if use_rsi:
        strategy.rsi_period = trial.suggest_int('rsi_period', 1, 60)
        strategy.rsi_overbought = trial.suggest_int('rsi_overbought', 50, 99)
        strategy.rsi_oversold = trial.suggest_int('rsi_oversold', 1, 50)
    else:
        strategy.rsi_period = None
        strategy.rsi_overbought = None
        strategy.rsi_oversold = None

    if use_keltner:
        strategy.keltner_period = trial.suggest_int('keltner_period', 1, 60)
        strategy.keltner_atr_factor = trial.suggest_float('keltner_atr_factor', 0.05, 4.0)
    else:
        strategy.keltner_period = None
        strategy.keltner_atr_factor = None

    if use_hma:
        strategy.hma_period = trial.suggest_int('hma_period', 1, 60)
    else:
        strategy.hma_period = None

    if use_vwap:
        strategy.vwap_std = trial.suggest_float('vwap_std', 0.05, 4.0)
    else:
        strategy.vwap_std = None

    if use_macd:
        strategy.macd_fast_period = trial.suggest_int('macd_fast_period', 1, 60)
        strategy.macd_slow_period = trial.suggest_int('macd_slow_period', 1, 60)
        strategy.macd_signal_period = trial.suggest_int('macd_signal_period', 1, 60)
    else:
        strategy.macd_fast_period = None
        strategy.macd_slow_period = None
        strategy.macd_signal_period = None

    if use_mfi:
        strategy.mfi_period = trial.suggest_int('mfi_period', 1, 60)
    else:
        strategy.mfi_period = None

    if use_obv:
        strategy.obv_ma_period = trial.suggest_int('obv_ma_period', 1, 60)
    else:
        strategy.obv_ma_period = None

    # Run strategy
    strategy.reset_state()
    strategy.run_strategy(data)

    performance = strategy.get_strategy_performance(data)
    pnl = performance['total_pnl']
    max_drawdown = performance['max_drawdown']

    print(f"\nPnL: {pnl:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")

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
    print("\nFetching data...\n")

    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)
    data = fetch_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, BACKTEST_START)

    print("Data fetched\n")

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

    manager.add_strategy(
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

    print("Strategies added\n")

    # Run strategies
    print("Calculating metrics...")
    manager.run_strategies(data)
    manager.calculate_metrics()
    manager.plot_results(data)
    print("\nMetrics calculated\n")


if __name__ == '__main__':
    # Choose optimization to run
    optimization = 3 # 1: Optimize a particular strategy, 2: Find the best strategy, 3: Combine and run multiple strategies together

    # Choose mode for optimization 1
    mode = 1 # 1: Run parameter optimization, 2: Print metrics and plot results 

    # Choose how many trials to run for optimization 2
    trials = 500
    
    match optimization:
        case 1:
            run_parameter_optimization_strategy(mode)   # Optimize a particular strategys parameters
        case 2:
            automated_optimization(trials)              # Find the best strategy
        case 3:
            run_multistrategy_optimization()            # Combine and run multiple strategies together