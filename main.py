import optuna
from binance.client import Client
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from strategy.optimize_parameters import Optimize_parameters
from strategy.multistrategy_manager import Multistrategy_manager
from config.config import (
    SYMBOL, BACKTEST_START, BACKTEST_END, API_KEY, API_SECRET, DEFAULT_PARAMS, OPTIMIZE_PARAMS, 
    MULTISTRAT_PARAMS, MULTISTRAT_PARAMS_2, MULTISTRAT_PARAMS_3, MULTISTRAT_PARAMS_4, 
    MULTISTRAT_PARAMS_5, MULTISTRAT_PARAMS_6, MULTISTRAT_PARAMS_7, MULTISTRAT_PARAMS_8,
    MULTISTRAT_PARAMS_9, MULTISTRAT_PARAMS_10, MULTISTRAT_PARAMS_11
)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def fetch_and_store_data(client, symbol, interval, start_str, filename, end_str=None):
    # Try loading existing data
    try:
        existing_df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        last_saved = existing_df.index[-1]
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        last_saved = None

    # Fetch new data starting from the last stored timestamp
    if last_saved:
        klines = client.get_historical_klines(symbol, interval, str(last_saved), end_str) if end_str else client.get_historical_klines(symbol, interval, str(last_saved))
    else:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str) if end_str else client.get_historical_klines(symbol, interval, start_str)

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

# Run parameter optimization or print metrics
def run_parameter_optimization(mode, backtest_start, backtest_end, filename='data/symbol_data.csv'):
    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)

    print("\nFetching data...\n")
    data = fetch_and_store_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, backtest_start, filename, backtest_end)
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
            strategy.run_strategy(data, isMetrics=True)
            strategy.calculate_metrics(data)
            strategy.plot_results(data)
            print("\nMetrics calculated\n")

# Run automated optimization with Optuna
def run_automated_optimization(trials, backtest_start, backtest_end, filename='data/symbol_data.csv'):
    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)
    
    print("\nFetching data...\n")
    global data
    data = fetch_and_store_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, backtest_start, filename, backtest_end)
    print("Data fetched\n")

    # Run optimization
    storage = 'sqlite:///data/parameter_optimization.db'
    study_name = 'walk_forward'

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
    strategy.stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.003, 0.15)
    strategy.take_profit_pct = trial.suggest_float('take_profit_pct', 0.003, 0.3)
    strategy.atr_period = trial.suggest_int('atr_period', 1, 100)
    strategy.atr_multiplier = trial.suggest_float('atr_multiplier', 0.01, 5.0)

    # Conditional parameter definitions
    if use_bb:
        strategy.bb_period = trial.suggest_int('bb_period', 1, 100)
        strategy.bb_std = trial.suggest_float('bb_std', 0.01, 5.0)
    else:
        strategy.bb_period = None
        strategy.bb_std = None
    
    if use_adx:
        strategy.adx_period = trial.suggest_int('adx_period', 1, 100)
        strategy.adx_threshold = trial.suggest_int('adx_threshold', 1, 100)
    else:  
        strategy.adx_period = None
        strategy.adx_threshold = None

    if use_rsi:
        strategy.rsi_period = trial.suggest_int('rsi_period', 1, 100)
        strategy.rsi_overbought = trial.suggest_int('rsi_overbought', 30, 99)
        strategy.rsi_oversold = trial.suggest_int('rsi_oversold', 1, 70)
    else:
        strategy.rsi_period = None
        strategy.rsi_overbought = None
        strategy.rsi_oversold = None

    if use_keltner:
        strategy.keltner_period = trial.suggest_int('keltner_period', 1, 100)
        strategy.keltner_atr_factor = trial.suggest_float('keltner_atr_factor', 0.01, 5.0)
    else:
        strategy.keltner_period = None
        strategy.keltner_atr_factor = None

    if use_hma:
        strategy.hma_period = trial.suggest_int('hma_period', 1, 100)
    else:
        strategy.hma_period = None

    if use_vwap:
        strategy.vwap_std = trial.suggest_float('vwap_std', 0.01, 5.0)
    else:
        strategy.vwap_std = None

    if use_macd:
        strategy.macd_fast_period = trial.suggest_int('macd_fast_period', 1, 100)
        strategy.macd_slow_period = trial.suggest_int('macd_slow_period', 1, 100)
        strategy.macd_signal_period = trial.suggest_int('macd_signal_period', 1, 100)
    else:
        strategy.macd_fast_period = None
        strategy.macd_slow_period = None
        strategy.macd_signal_period = None

    if use_mfi:
        strategy.mfi_period = trial.suggest_int('mfi_period', 1, 100)
    else:
        strategy.mfi_period = None

    if use_obv:
        strategy.obv_ma_period = trial.suggest_int('obv_ma_period', 1, 100)
    else:
        strategy.obv_ma_period = None

    # Run strategy
    strategy.reset_state()
    strategy.run_strategy(data)

    performance = strategy.get_strategy_performance(data)
    pnl = performance['total_pnl']
    max_drawdown = performance['max_drawdown']
    win_rate = performance['win_rate']
    num_trades = performance['num_trades']

    print(f"\nPnL: {pnl:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")
    print(f"Win Rate: {win_rate:.2f}")
    print(f"Number of Trades: {num_trades}")

    # Restrictions for the strategy
    if pnl <= 100 or max_drawdown >= 70 or num_trades < 10:
        return 0

    normalized_pnl = np.log1p(max(0, pnl)) / np.log1p(10000)
    normalized_mdd = max(0, 1 - (max_drawdown / 100))
    normalized_wr = win_rate / 100

    combined_metric = (
        0.6 * normalized_pnl +
        0.3 * normalized_mdd +
        0.05 * normalized_wr +
        0.04 * (num_trades / 1000)
    )

    return combined_metric

# Run multistrategy optimization
def run_multistrategy_optimization(backtest_start, backtest_end, filename='data/symbol_data.csv'):
    # Initialize API and fetch data
    client = Client(API_KEY, API_SECRET)
    
    print("\nFetching data...\n")
    data = fetch_and_store_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, backtest_start, filename, backtest_end)
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

    """ manager.add_strategy(
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
    ) """

    """ manager.add_strategy(
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
    ) """
    
    """ manager.add_strategy(
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
    ) """
    
    """ manager.add_strategy(
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
    ) """

    """ manager.add_strategy(
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
    ) """
    
    """ manager.add_strategy(
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
    ) """

    """ manager.add_strategy(
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
    ) """

    """ manager.add_strategy(
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
    ) """

    """ manager.add_strategy(
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
    ) """

    """ manager.add_strategy(
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
    ) """

    print("Strategies added\n")

    # Run strategies
    print("Calculating metrics...")
    manager.run_strategies(data)
    manager.calculate_metrics()
    manager.plot_results(data)
    print("\nMetrics calculated\n")

# In-sample permutation test
def permutation_test(strategy_func, data, n_permutations=1000, **kwargs):
    """Run a permutation test by shuffling data and re-running the strategy."""
    original_performance = strategy_func(data, **kwargs)
    perm_better_than_orig = 0
    permuted_performances = []

    for _ in tqdm(range(n_permutations)):
        permuted_data = permute_returns_and_rebuild(data, random_state=None) # Use None to get a different random seed each time
        permuted_data.index = data.index  # Keep original index if needed
        permuted_data.columns = data.columns
        perf = strategy_func(permuted_data, **kwargs)
        if perf > original_performance:
            perm_better_than_orig += 1
        permuted_performances.append(perf)
    
    return original_performance, permuted_performances, perm_better_than_orig / n_permutations

def run_in_sample_permutation_test(backtest_start, backtest_end, filename='data/symbol_data.csv', n_permutations=1000):
    client = Client(API_KEY, API_SECRET)

    print("\nFetching data...\n")
    data = fetch_and_store_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, backtest_start, filename, backtest_end)
    print("Data fetched\n")

    strategy = Optimize_parameters()

    def strategy_func(d, **kwargs):
        strategy.reset_state()
        strategy.run_strategy(d)
        return strategy.get_strategy_performance(d)['profit_factor']
    
    print("Running permutation test...\n")
    orig, perms, better = permutation_test(strategy_func, data, n_permutations=n_permutations)
    print(
        f"\nOrig PF: {orig}\n"
        f"Perm mean: {np.mean(perms)}\n"
        f"Std: {np.std(perms)}\n"
        f"Better than orig: {better:.2%}\n"
    )
    
    # Plot results
    plot_permutation_results(orig, perms, data)

# Walk-forward permutation test
def run_walk_forward_permutation_test(backtest_start, backtest_end, filename='data/symbol_data.csv', n_permutations=100):
    # Fetch data for walk-forward period
    client = Client(API_KEY, API_SECRET)

    print("\nFetching data...\n")
    data = fetch_and_store_data(client, SYMBOL, Client.KLINE_INTERVAL_1HOUR, backtest_start, filename, backtest_end)
    print("Data fetched\n")

    strategy = Optimize_parameters()
    strategy.run_walk_forward_optimization(data)
    original_performance = strategy.get_strategy_performance(data)['profit_factor']
    perm_better_than_orig = 0
    permuted_performances = []

    print("Running walk-forward permutation test...\n")
    for _ in tqdm(range(n_permutations)):
        permuted_data = permute_returns_and_rebuild(data, random_state=None) # Use None to get a different random seed each time
        permuted_data.index = data.index  # Keep original index if needed
        permuted_data.columns = data.columns
        strategy.reset_state()
        strategy.run_walk_forward_optimization(permuted_data)
        perf = strategy.get_strategy_performance(data)['profit_factor']
        if perf > original_performance:
            perm_better_than_orig += 1
        permuted_performances.append(perf)
    
    better = perm_better_than_orig / n_permutations
    print(
        f"\nOrig PF: {original_performance}\n"
        f"Perm mean: {np.mean(permuted_performances)}\n"
        f"Std: {np.std(permuted_performances)}\n"
        f"Better than orig: {better:.2%}\n"
    )

    # Plot results
    plot_permutation_results(original_performance, permuted_performances, data)

# Methods for permutation testing
def permute_returns_and_rebuild(data, random_state=None):
    """Permute returns, reconstruct price series, keep first and last price the same."""
    closes = data['close'].values
    opens = data['open'].values

    # Calculate log returns
    log_returns = np.log(closes[1:] / closes[:-1])

    # Permute the returns except the first and last
    if len(log_returns) > 2:
        middle_returns = log_returns[1:-1].copy()
        rng = np.random.default_rng(random_state)
        rng.shuffle(middle_returns)
        permuted_returns = np.concatenate([[log_returns[0]], middle_returns, [log_returns[-1]]])
    else:
        permuted_returns = log_returns

    # Rebuild close prices
    permuted_closes = [closes[0]]
    for r in permuted_returns:
        permuted_closes.append(permuted_closes[-1] * np.exp(r))
    permuted_closes = np.array(permuted_closes)

    # Set open = previous close
    permuted_opens = np.roll(permuted_closes, 1)
    permuted_opens[0] = opens[0]

    permuted_data = data.copy()
    permuted_data['open'] = permuted_opens
    permuted_data['close'] = permuted_closes

    # Add random wicks for more realistic highs/lows
    wick_upper_cap = np.random.uniform(0.0005, 0.02)
    high_wick = permuted_data['close'] * (1 + np.random.uniform(0, wick_upper_cap, len(permuted_data)))
    permuted_data['high'] = np.maximum(permuted_data['open'], permuted_data['close'])
    permuted_data['high'] = np.maximum(permuted_data['high'], high_wick)
    wick_upper_cap = np.random.uniform(0.0005, 0.02)
    low_wick = permuted_data['close'] * (1 - np.random.uniform(0, wick_upper_cap, len(permuted_data)))
    permuted_data['low'] = np.minimum(permuted_data['open'], permuted_data['close'])
    permuted_data['low'] = np.minimum(permuted_data['low'], low_wick)

    return permuted_data

def plot_permutation_results(orig, perms, data):
    """Plot the results of the permutation test."""
    plt.figure(figsize=(8, 4))
    plt.hist(perms, bins=30, alpha=0.7, label='Permuted')
    plt.axvline(orig, color='red', linestyle='dashed', linewidth=2, label='Original')
    plt.xlabel('Profit Factor')
    plt.ylabel('Frequency')
    plt.title('Permutation Test: Profit Factor')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/permutation_dist.png')

    # Plot original vs one permuted price series
    plt.figure(figsize=(12, 4))
    plt.plot(data['close'].values, label='Original Close')
    # Create one permuted series for visualization
    permuted_data = permute_returns_and_rebuild(data, random_state=None)
    plt.plot(permuted_data['close'].values, label='Permuted Close', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.title('Original vs Permuted Close Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/permutation_price.png')


if __name__ == '__main__':
    # Choose optimization to run
    optimization = 4    # 1: Optimize a particular strategy, 2: Find the best strategy, 
                        # 3: Combine and run multiple strategies together, 4: In-sample permutation test,
                        # 5: Walk forward optimization

    # Choose mode for optimization 1
    mode = 2            # 1: Run parameter optimization, 2: Print metrics and plot results 

    # Choose how many trials to run for optimization 2
    trials = 5000

    # Choose how many permutations to run for optimization 4
    n_permutations = 1000

    # Choose which time period to run the optimization on
    time_period = 1     # 1: 2016-2019, 2: 2020, 3: 2021, 4: 2022, 
                        # 5: 2023, 6: 2024, 7: Custom period (set BACKTEST_START and BACKTEST_END in config.py)

    match time_period:
        case 1:
            print("Running optimization on 2016-2019 data...\n")
            backtest_start = "1 Jan, 2016"
            backtest_end = "31 Dec, 2019"
            filename = 'data/symbol_data_2016_2019.csv'
        case 2:
            print("Running optimization on 2020 data...\n")
            backtest_start = "1 Jan, 2020"
            backtest_end = "31 Dec, 2020"
            filename = 'data/symbol_data_2020.csv'
        case 2:
            print("Running optimization on 2021 data...\n")
            backtest_start = "1 Jan, 2021"
            backtest_end = "31 Dec, 2021"
            filename = 'data/symbol_data_2021.csv'
        case 3:
            print("Running optimization on 2022 data...\n")
            backtest_start = "1 Jan, 2022"
            backtest_end = "31 Dec, 2022"
            filename = 'data/symbol_data_2022.csv'
        case 4:
            print("Running optimization on 2023 data...\n")
            backtest_start = "1 Jan, 2023"
            backtest_end = "31 Dec, 2023"
            filename = 'data/symbol_data_2023.csv'
        case 5:
            print("Running optimization on 2024 data...\n")
            backtest_start = "1 Jan, 2024"
            backtest_end = "31 Dec, 2024"
            filename = 'data/symbol_data_2024.csv'
        case 6:
            print("Running optimization on 2025 data...\n")
            backtest_start = "1 Jan, 2025"
            backtest_end = "31 Dec, 2025"
            filename = 'data/symbol_data_2025.csv'
        case 7:
            print("Running optimization on data from BACKTEST_START to BACKTEST_END...\n")
            backtest_start = BACKTEST_START
            backtest_end = BACKTEST_END
            filename = 'data/symbol_data_custom.csv'

    match optimization:
        case 1:
            run_parameter_optimization(mode, backtest_start, backtest_end, filename)                     # Optimize a particular strategys parameters
        case 2:
            run_automated_optimization(trials, backtest_start, backtest_end, filename)                   # Find the best strategy
        case 3:
            run_multistrategy_optimization(backtest_start, backtest_end, filename)                       # Combine and run multiple strategies together
        case 4:
            run_in_sample_permutation_test(backtest_start, backtest_end, filename, n_permutations)       # In-sample permutation test
        case 5:
            run_walk_forward_permutation_test(backtest_start, backtest_end, filename, n_permutations)    # Walk forward optimization
