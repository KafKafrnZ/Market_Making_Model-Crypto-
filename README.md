# Crypto Market Maker Bot

A sophisticated cryptocurrency market-making bot built with Python, leveraging machine learning (LSTM), technical analysis, and the Binance API to provide liquidity by placing buy and sell orders around a dynamic spread. The bot includes risk management, backtesting, and real-time visualization using Dash.

## Features
- **Price Prediction**: Uses an LSTM neural network optimized with Optuna for hyperparameter tuning to predict future prices.
- **Dynamic Spread Calculation**: Adjusts spreads based on ATR, RSI, MACD volatility, VWAP, and order book depth.
- **Order Management**: Places ladder buy/sell orders with configurable quantities and spread ranges.
- **Risk Management**: Enforces maximum loss per trade and drawdown limits.
- **Backtesting**: Simulates trading strategies on historical data.
- **Visualization**: Real-time price and prediction plotting using Dash.
- **Rate Limiting**: Handles Binance API weight limits with a custom rate limiter.
- **Test Mode**: Supports simulated trading without real funds.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **Binance API Keys**: Obtain an API key and secret from Binance (stored in a `.env` file).
- **Dependencies**: Listed in `requirements.txt`.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd crypto-market-maker