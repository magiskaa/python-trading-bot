import optuna
from binance.client import Client
from strategy.optimize_parameters import Optimize_parameters
from strategy.optimize_time_frame import Optimize_time_frame
from strategy.strategy_utils import fetch_data, fetch_multi_timeframe_data, find_best_timeframe
from config.config import SYMBOL, BACKTEST_START, API_KEY, API_SECRET, DEFAULT_PARAMS
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
        'leverage': 10,
        'macd_fast_period': 12,
        'macd_slow_period': 26,
        'macd_signal_period': 9,
        'mfi_period': 14,
        'obv_ma_period': 14,
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

    storage = 'sqlite:///data/parameter_optimization.db'
    study_name = 'parameter_optimization'

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize', 
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=200)

    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)

def objective(trial):
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

    # Conditional parameter definitions
    if use_bb:
        bb_period = trial.suggest_int('bb_period', 1, 30)
        bb_std = trial.suggest_float('bb_std', 0.2, 3.0)
        strategy.bb_period = bb_period
        strategy.bb_std = bb_std
    else:
        strategy.bb_period = None
        strategy.bb_std = None
    
    if use_adx:
        adx_period = trial.suggest_int('adx_period', 1, 60)
        adx_threshold = trial.suggest_int('adx_threshold', 1, 50)
        strategy.adx_period = adx_period
        strategy.adx_threshold = adx_threshold
    else:  
        strategy.adx_period = None
        strategy.adx_threshold = None

    if use_rsi:
        rsi_period = trial.suggest_int('rsi_period', 1, 50)
        rsi_overbought = trial.suggest_int('rsi_overbought', 55, 95)
        rsi_oversold = trial.suggest_int('rsi_oversold', 5, 45)
        strategy.rsi_period = rsi_period
        strategy.rsi_overbought = rsi_overbought
        strategy.rsi_oversold = rsi_oversold
    else:
        strategy.rsi_period = None
        strategy.rsi_overbought = None
        strategy.rsi_oversold = None

    if use_keltner:
        keltner_period = trial.suggest_int('keltner_period', 1, 35)
        keltner_atr_factor = trial.suggest_float('keltner_atr_factor', 0.5, 3.0)
        strategy.keltner_period = keltner_period
        strategy.keltner_atr_factor = keltner_atr_factor
    else:
        strategy.keltner_period = None
        strategy.keltner_atr_factor = None

    if use_hma:
        hma_period = trial.suggest_int('hma_period', 1, 35)
        strategy.hma_period = hma_period
    else:
        strategy.hma_period = None

    if use_vwap:
        vwap_std = trial.suggest_float('vwap_std', 0.2, 3.0)
        strategy.vwap_std = vwap_std
    else:
        strategy.vwap_std = None

    if use_macd:
        macd_fast_period = trial.suggest_int('macd_fast_period', 1, 40)
        macd_slow_period = trial.suggest_int('macd_slow_period', 1, 40)
        macd_signal_period = trial.suggest_int('macd_signal_period', 1, 40)
        strategy.macd_fast_period = macd_fast_period
        strategy.macd_slow_period = macd_slow_period
        strategy.macd_signal_period = macd_signal_period
    else:
        strategy.macd_fast_period = None
        strategy.macd_slow_period = None
        strategy.macd_signal_period = None

    if use_mfi:
        mfi_period = trial.suggest_int('mfi_period', 1, 35)
        strategy.mfi_period = mfi_period
    else:
        strategy.mfi_period = None

    if use_obv:
        obv_ma_period = trial.suggest_int('obv_ma_period', 1, 40)
        strategy.obv_ma_period = obv_ma_period
    else:
        strategy.obv_ma_period = None
    
    strategy.reset_state()
    strategy.run_strategy(data, True)

    performance = strategy.get_strategy_performance(data)
    pnl = performance['total_pnl']
    max_drawdown = performance['max_drawdown']

    print("\nPnL: ", pnl)
    print("Max Drawdown: ", max_drawdown)

    weight_pnl = 0.5
    weight_mdd = 0.5

    normalized_pnl = min(1.0, pnl / 1000) if pnl > 0 else max(-1.0, pnl / 1000)
    normalized_mdd = 1 - max_drawdown / 100

    combined_metric = (
        weight_pnl * normalized_pnl +
        weight_mdd * normalized_mdd
    )

    return combined_metric


if __name__ == '__main__':
    #run_parameter_optimization_strategy()
    #run_timeframe_optimization_strategy()
    automated_optimization()