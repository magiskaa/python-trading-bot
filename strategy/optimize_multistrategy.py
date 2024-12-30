from strategy.strategy_base import Strategy_base
from config.config import DEFAULT_PARAMS

class Optimize_multistrategy(Strategy_base):
    def __init__(self, 
                 bb_period, bb_std, 
                 adx_period, adx_threshold, 
                 rsi_period, rsi_overbought, rsi_oversold, 
                 stop_loss_pct, take_profit_pct, 
                 atr_period, atr_multiplier, 
                 keltner_period, keltner_atr_factor, 
                 hma_period, 
                 vwap_std, 
                 macd_fast_period, macd_slow_period, macd_signal_period, 
                 mfi_period, 
                 obv_ma_period):
        
        # Initialize the base class with common parameters
        super().__init__(DEFAULT_PARAMS['starting_balance'], DEFAULT_PARAMS['leverage'])

        # Set strategy parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.keltner_period = keltner_period
        self.keltner_atr_factor = keltner_atr_factor
        self.hma_period = hma_period
        self.vwap_std = vwap_std
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.mfi_period = mfi_period
        self.obv_ma_period = obv_ma_period
        
        # Initialize additional attributes
        self.current_position = 0
        self.entry_price = 0.0
        self.current_balance = self.starting_balance
        self.balance_history = []
        self.last_df = None
