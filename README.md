# Python Trading Bot

A cryptocurrency trading bot using technical indicators to trade on Binance futures market.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Configuration](#configuration)
- [LICENSE](#LICENSE)
- [Contact](#contact)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/magiskaa/python-trading-bot.git
cd python_trading_bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a config file:
```bash
cp config/config.example.py config/config.py
```

5. Add your Binance API keys to `config/config.py`

## Features

- Technical analysis using Bollinger Bands, RSI, ADX, Keltner channels, HMA, VWAP, MACD, OBV and MFI
- Trading strategy finder using optuna library
- Strategy parameter optimization with parallel processing
- Multiple strategy performance testing
- Performance visualization
- Risk management with take-profit and dynamic stop-loss
- Futures trading with real money

## Configuration

Edit `config/config.py` to set your:
- API keys
- Trading pairs
- Default parameters
- Risk management settings
- Parameters for multistrategy testing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Valtteri Antikainen, vantikaine@gmail.com