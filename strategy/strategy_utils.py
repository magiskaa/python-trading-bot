import numpy as np
import pandas as pd

def fetch_multi_timeframe_data(client, symbol, start_date, timeframes):
    """Fetch data for different timeframe intervals."""
    timeframe_data = {}
    for tf in timeframes:
        klines = client.get_historical_klines(symbol, tf, start_date)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        timeframe_data[tf] = df
    return timeframe_data

def fetch_data(client, symbol, interval, start_str):
    klines = client.get_historical_klines(symbol, interval, start_str)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                        'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def find_best_timeframe(strategy, timeframe_data, best_params):
    """Test the optimized parameters on different timeframes."""
    timeframe_results = {}

    for param, value in best_params.items():
        setattr(strategy, param, value)

    print("\nTesting timeframes:")
    for timeframe, data in timeframe_data.items():
        strategy.reset_state()
        strategy.run_strategy(data)

        trades = strategy.trades
        wins = len([t for t in trades if t['pnl'] > 0])
        win_rate = (wins / len(trades) * 100) if trades else 0

        results = {
            'pnl': sum(strategy.pnl),
            'max_drawdown': strategy.max_drawdown * 100,
            'num_trades': len(trades),
            'win_rate': win_rate
        }

        timeframe_results[timeframe] = results
        print(f"\n{timeframe}:")
        print(f"PnL: ${results['pnl']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Trades: {results['num_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")

    return timeframe_results