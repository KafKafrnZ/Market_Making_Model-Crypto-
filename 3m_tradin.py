import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import tensorflow as tf
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sklearn.preprocessing import MinMaxScaler
import time
import random
import threading
import logging
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import requests
from datetime import datetime
import json
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error
import optuna
import talib
from collections import deque

load_dotenv()

# Constants for minimum data requirements
MIN_HISTORICAL_POINTS = 50
MIN_PREDICTION_POINTS = 20

# Logger setup
class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger("MarketMaker")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)

# Binance client setup
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    logger.error("API keys not found.")
    raise ValueError("API keys not found")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)

class WeightBasedRateLimiter:
    def __init__(self, weight_limit_per_minute=1200, reset_interval=60):
        self.weight_limit = weight_limit_per_minute
        self.reset_interval = reset_interval
        self.current_weight = 0
        self.last_reset = time.time()
        self.lock = threading.Lock()

    def wait_if_needed(self, request_weight):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_reset
            if elapsed >= self.reset_interval:
                self.current_weight = 0
                self.last_reset = current_time
            if self.current_weight + request_weight > self.weight_limit:
                sleep_time = self.reset_interval - elapsed
                logger.info(f"Rate limit approaching, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.current_weight = 0
                self.last_reset = time.time() + sleep_time
            self.current_weight += request_weight
            logger.debug(f"Current API weight: {self.current_weight}/{self.weight_limit}")

class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, dense_units):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim, dense_units)
        self.output_layer = nn.Linear(dense_units, 1)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = torch.relu(self.dense(out))
        out = self.output_layer(out)
        return out

# Backtester Class
class Backtester:
    def __init__(self, historical_data, initial_balance=10000):
        self.historical_data = historical_data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trades = []

    def run_backtest(self, strategy):
        for i in range(len(self.historical_data) - 1):
            current_price = self.historical_data[i]
            next_price = self.historical_data[i + 1]
            action = strategy(current_price)
            if action == "BUY" and self.balance >= current_price:
                self.balance -= current_price
                self.positions.append(current_price)
                self.trades.append(("BUY", current_price))
            elif action == "SELL" and self.positions:
                purchase_price = self.positions.pop(0)
                profit = next_price - purchase_price
                self.balance += next_price
                self.trades.append(("SELL", next_price, profit))
        return self.balance, self.trades

# Risk Management Class
class RiskManager:
    def __init__(self, max_loss_per_trade=0.02, max_drawdown=0.1):
        self.max_loss_per_trade = max_loss_per_trade
        self.max_drawdown = max_drawdown
        self.initial_balance = None
        self.current_balance = None

    def set_balance(self, balance):
        self.initial_balance = balance
        self.current_balance = balance

    def check_risk(self, trade_profit):
        if self.current_balance is None or self.initial_balance is None:
            raise ValueError("Balance not set")
        self.current_balance += trade_profit
        drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
        if abs(trade_profit) / self.current_balance > self.max_loss_per_trade or drawdown > self.max_drawdown:
            return False
        return True

def calculate_indicators(historical_data):
    if len(historical_data) < 26:
        logger.error("Not enough data for technical indicators")
        return np.zeros((len(historical_data), 8))  # Updated to 8 columns with MACD volatility
    close_prices = historical_data[:, 3]
    high_prices = historical_data[:, 1]
    low_prices = historical_data[:, 2]
    volumes = historical_data[:, 4]
    
    logger.debug(f"Calculating indicators with {len(historical_data)} points")
    logger.debug(f"Volumes: {volumes[-5:]}")
    
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
    bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    rsi = talib.RSI(close_prices, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Calculate MACD volatility (standard deviation of MACD over a 14-period window)
    macd_volatility = pd.Series(macd).rolling(window=14, min_periods=1).std().to_numpy()
    macd_volatility = np.nan_to_num(macd_volatility, nan=0.0)
    
    cum_vol = np.cumsum(volumes)
    cum_vol_price = np.cumsum(volumes * close_prices)
    vwap = np.where(cum_vol == 0, close_prices, cum_vol_price / np.where(cum_vol == 0, 1e-10, cum_vol))
    
    indicators = np.column_stack((atr, bollinger_upper, bollinger_middle, bollinger_lower, rsi, macd, vwap, macd_volatility))
    indicators = np.nan_to_num(indicators, nan=0.0, posinf=0.0, neginf=0.0)
    return indicators

def calculate_dynamic_spread(mid_price, indicators, order_book_depth):
    atr = indicators[-1, 0]  # ATR at index 0
    rsi = indicators[-1, 4]  # RSI at index 4
    vwap = indicators[-1, 6]  # VWAP at index 6
    macd_volatility = indicators[-1, 7]  # MACD volatility at index 7
    base_spread = 0.001
    volatility_adjustment = atr / mid_price  # ATR-based volatility
    macd_vol_adjustment = macd_volatility / mid_price  # MACD volatility adjustment
    momentum_adjustment = abs(rsi - 50) / 50  # RSI-based momentum
    liquidity_adjustment = 1 / order_book_depth  # Order book depth
    spread = (base_spread + volatility_adjustment + macd_vol_adjustment + momentum_adjustment + liquidity_adjustment) / 2  # Average adjustments
    return min(max(spread, 0.0005), 0.01)

def objective(trial, historical_data):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    dense_units = trial.suggest_int('dense_units', 16, 128)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    sequence_length = trial.suggest_int('sequence_length', 20, min(100, len(historical_data) - 1), step=10)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    epochs = trial.suggest_int('epochs', 10, 50, step=10)

    if len(historical_data) < sequence_length + 1:
        return float('inf')

    scaler = MinMaxScaler()
    indicators = calculate_indicators(historical_data)
    augmented_data = np.hstack((historical_data, indicators))
    scaled_data = scaler.fit_transform(augmented_data)
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, 3])
    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = PriceLSTM(input_dim=augmented_data.shape[1], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, dense_units=dense_units)
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            if torch.isnan(loss):
                logger.error(f"NaN loss in trial with lr={lr}, epoch={epoch}")
                return float('inf')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        val_output = model(torch.FloatTensor(X_val))
        val_loss = criterion(val_output, torch.FloatTensor(y_val).unsqueeze(1))
        if torch.isnan(val_loss):
            logger.error("NaN validation loss in trial")
            return float('inf')
    return val_loss.item()

def train_lstm_model(historical_data, sequence_length, batch_size, epochs, n_trials=20):
    logger.info(f"Historical data points: {len(historical_data)}")
    if len(historical_data) < MIN_HISTORICAL_POINTS:
        logger.warning(f"Less than optimal historical data. Required: {MIN_HISTORICAL_POINTS}, Available: {len(historical_data)}. Proceeding with caution.")
    if len(historical_data) < 10:
        logger.error(f"Too few data points to train model: {len(historical_data)}")
        return None, None, sequence_length, batch_size

    indicators = calculate_indicators(historical_data)
    if np.any(np.isnan(indicators)) or np.any(np.isinf(indicators)):
        logger.error("Indicators contain nan or inf values")
        indicators = np.nan_to_num(indicators, nan=0.0, posinf=0.0, neginf=0.0)
    historical_data = np.hstack((historical_data, indicators))
    if np.any(np.isnan(historical_data)) or np.any(np.isinf(historical_data)):
        logger.error("Combined historical data contains nan or inf values")
        historical_data = np.nan_to_num(historical_data, nan=0.0, posinf=0.0, neginf=0.0)
    close_prices = historical_data[:, 3]
    mean, std = np.mean(close_prices), np.std(close_prices)
    mask = (close_prices > mean - 3 * std) & (close_prices < mean + 3 * std)
    historical_data = historical_data[mask]
    logger.info(f"Filtered historical data shape: {historical_data.shape}")

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, historical_data), n_trials=n_trials)
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(historical_data)
    if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
        logger.error("Scaled data contains nan or inf values")
        scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

    sequence_length = min(best_params['sequence_length'], len(historical_data) - 1)
    batch_size = best_params['batch_size']
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, 3])
    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)

    model = PriceLSTM(
        input_dim=historical_data.shape[1],
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        dense_units=best_params['dense_units']
    )
    if best_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    elif best_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=best_params['lr'], momentum=best_params['momentum'])
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience, patience_counter = 10, 0

    for epoch in range(best_params['epochs']):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            if torch.isnan(loss):
                logger.error(f"NaN loss detected in epoch {epoch+1}")
                return None, None, sequence_length, batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_mae = mean_absolute_error(y_val.numpy(), val_output.numpy())
        if torch.isnan(val_loss):
            logger.error(f"NaN validation loss in epoch {epoch+1}")
            return None, None, sequence_length, batch_size
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
        logger.info(f"Epoch {epoch+1}/{best_params['epochs']}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}, Val MAE: {val_mae}")
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    logger.info("Price prediction model trained successfully")
    return model, scaler, sequence_length, batch_size

def predict_future_price(model, scaler, historical_data, sequence_length):
    logger.info(f"Predicting with {len(historical_data)} data points, sequence_length={sequence_length}")
    if len(historical_data) < MIN_PREDICTION_POINTS:
        logger.error(f"Not enough data for prediction. Required: {MIN_PREDICTION_POINTS}, Available: {len(historical_data)}")
        return historical_data[-1, 3] if len(historical_data) > 0 else 0.0
    indicators = calculate_indicators(historical_data)
    augmented_data = np.hstack((historical_data, indicators))
    recent_data = augmented_data[-sequence_length:]
    scaled_data = scaler.transform(recent_data)
    X = torch.FloatTensor(scaled_data).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(X)
    dummy_output = np.zeros((1, augmented_data.shape[1]))
    dummy_output[0, 3] = prediction.numpy()[0, 0]
    prediction_transformed = scaler.inverse_transform(dummy_output)
    return prediction_transformed[0, 3]

def run_dash(shared_data):
    app = dash.Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id='price-graph'),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
    ])
    @app.callback(
        Output('price-graph', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_graph(n):
        with shared_data['lock']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=shared_data['timestamps'], y=shared_data['historical_prices'], name='Historical'))
            fig.add_trace(go.Scatter(x=shared_data['timestamps'], y=shared_data['predicted_prices'], name='Predicted'))
            fig.update_layout(title='Price History and Predictions', xaxis_title='Time', yaxis_title='Price')
        return fig
    app.run_server(debug=False, port=8050)

class CryptoMarketMaker:
    def __init__(self, symbol, order_quantity, max_orders=10, min_spread=0.001, max_spread=0.005, order_refresh_time=60, test_mode=True, historical_data_days=90, retrain_interval=3600):
        self.symbol = symbol
        self.order_quantity = order_quantity
        self.max_orders = max_orders
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.order_refresh_time = order_refresh_time
        self.test_mode = test_mode
        self.historical_data_days = historical_data_days
        self.retrain_interval = retrain_interval
        self.rate_limiter = WeightBasedRateLimiter()
        self.active_orders = {'buy': [], 'sell': []}
        self.symbol_info = self.get_symbol_info()
        self.data_lock = threading.Lock()
        self.recent_data = deque(maxlen=100)

        self.shared_data = {
            'lock': self.data_lock,
            'historical_prices': [], 'predicted_prices': [], 'market_prices': [], 'timestamps': [],
            'active_orders': self.active_orders, 'balances': self.get_account_balances() or {'base': {'free': 0}, 'quote': {'free': 0}}
        }

        logger.info(f"Fetching historical data for {self.symbol}...")
        historical_data = self.get_historical_prices()
        if len(historical_data) < MIN_HISTORICAL_POINTS:
            logger.warning(f"Insufficient historical data ({len(historical_data)} points), trying extended range...")
            historical_data = self.get_historical_prices_extended()
        if len(historical_data) < MIN_HISTORICAL_POINTS:
            logger.warning(f"Still insufficient data ({len(historical_data)} points), switching to 15-minute interval...")
            historical_data = self.get_historical_prices_short_interval()
        if len(historical_data) < MIN_HISTORICAL_POINTS:
            logger.error(f"Final attempt failed: only {len(historical_data)} points available. Proceeding with limited data.")
        
        self.shared_data['historical_prices'] = [row[3] for row in historical_data[-100:]]
        self.shared_data['timestamps'] = [int(time.time() * 1000) - (900000 * (len(self.shared_data['historical_prices']) - i)) 
                                          for i in range(len(self.shared_data['historical_prices']))]
        self.shared_data['market_prices'] = self.shared_data['historical_prices'].copy()
        self.shared_data['predicted_prices'] = self.shared_data['historical_prices'].copy()

        for row in historical_data[-min(100, len(historical_data)):]:
            self.recent_data.append(row)

        self.dash_thread = threading.Thread(target=run_dash, args=(self.shared_data,))
        self.dash_thread.daemon = True
        self.dash_thread.start()

        self.sequence_length = min(50, len(historical_data))
        self.batch_size = 32
        self.model = None
        self.scaler = None
        self.model_ready = False
        self.last_retrain_time = 0

        self.training_thread = threading.Thread(target=self.initialize_model, args=(historical_data,))
        self.training_thread.daemon = True
        self.training_thread.start()

        # Initialize RiskManager
        self.risk_manager = RiskManager()
        self.risk_manager.set_balance(self.shared_data['balances']['quote']['free'])

    def initialize_model(self, historical_data):
        try:
            self.model, self.scaler, self.sequence_length, self.batch_size = train_lstm_model(historical_data, self.sequence_length, self.batch_size, epochs=50)
            if self.model is None:
                logger.error("Model training failed due to insufficient data or other issues")
                return
            self.model_ready = True
            self.last_retrain_time = time.time()
        except Exception as e:
            logger.error(f"Error training model: {e}")

    def retrain_model(self):
        try:
            logger.info("Retraining model with new data...")
            historical_data = self.get_historical_prices()
            if len(historical_data) < MIN_HISTORICAL_POINTS:
                historical_data = self.get_historical_prices_extended()
            if len(historical_data) < MIN_HISTORICAL_POINTS:
                historical_data = self.get_historical_prices_short_interval()
            self.model, self.scaler, self.sequence_length, self.batch_size = train_lstm_model(historical_data, self.sequence_length, self.batch_size, epochs=50)
            if self.model is None:
                logger.error("Model retraining failed")
                return
            self.last_retrain_time = time.time()
            logger.info("Model retraining complete")
        except Exception as e:
            logger.error(f"Error retraining model: {e}")

    def get_symbol_info(self):
        try:
            self.rate_limiter.wait_if_needed(1)
            exchange_info = client.get_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == self.symbol:
                    price_precision = 0
                    quantity_precision = 0
                    for filter_item in symbol_info['filters']:
                        if filter_item['filterType'] == 'PRICE_FILTER':
                            price_precision = self.get_precision_from_step(filter_item['tickSize'])
                        elif filter_item['filterType'] == 'LOT_SIZE':
                            quantity_precision = self.get_precision_from_step(filter_item['stepSize'])
                    return {
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset'],
                        'pricePrecision': price_precision,
                        'quantityPrecision': quantity_precision,
                        'minNotional': next((filter_item['minNotional'] for filter_item in symbol_info['filters'] 
                                           if filter_item['filterType'] == 'MIN_NOTIONAL'), 0)
                    }
            logger.error(f"Symbol {self.symbol} not found in exchange info")
            return None
        except BinanceAPIException as e:
            logger.error(f"Binance API Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None

    def get_precision_from_step(self, step_size):
        step_str = "{:0.8f}".format(float(step_size))
        return len(step_str.rstrip('0').split('.')[1])

    def format_price(self, price):
        if not self.symbol_info:
            return float(f"{price:.2f}")
        precision = self.symbol_info.get('pricePrecision', 2)
        return float(f"{price:.{precision}f}")

    def format_quantity(self, quantity):
        if not self.symbol_info:
            return float(f"{quantity:.6f}")
        precision = self.symbol_info.get('quantityPrecision', 6)
        return float(f"{quantity:.{precision}f}")

    def get_historical_prices(self):
        try:
            self.rate_limiter.wait_if_needed(1)
            klines = client.get_historical_klines(
                self.symbol, 
                Client.KLINE_INTERVAL_1HOUR, 
                f"{self.historical_data_days} days ago UTC"
            )
            data = np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines])
            logger.info(f"Fetched {len(data)} historical data points for {self.historical_data_days} days")
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.error("Historical data contains nan or inf values")
                return np.array([])
            return data
        except BinanceAPIException as e:
            logger.error(f"Binance API Error in get_historical_prices: {e}")
            return np.array([])
        except Exception as e:
            logger.error(f"Error in get_historical_prices: {e}")
            return np.array([])

    def get_historical_prices_extended(self):
        try:
            extended_days = 180
            self.rate_limiter.wait_if_needed(1)
            klines = client.get_historical_klines(
                self.symbol, 
                Client.KLINE_INTERVAL_1HOUR, 
                f"{extended_days} days ago UTC"
            )
            data = np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines])
            logger.info(f"Extended fetch: {len(data)} historical data points for {extended_days} days")
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.error("Extended historical data contains nan or inf values")
                return np.array([])
            return data
        except BinanceAPIException as e:
            logger.error(f"Binance API Error in get_historical_prices_extended: {e}")
            return np.array([])
        except Exception as e:
            logger.error(f"Error in get_historical_prices_extended: {e}")
            return np.array([])

    def get_historical_prices_short_interval(self):
        try:
            self.rate_limiter.wait_if_needed(1)
            klines = client.get_historical_klines(
                self.symbol, 
                Client.KLINE_INTERVAL_15MINUTE, 
                f"{self.historical_data_days} days ago UTC"
            )
            data = np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines])
            logger.info(f"Short interval fetch: {len(data)} historical data points for {self.historical_data_days} days (15-minute interval)")
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.error("Short interval historical data contains nan or inf values")
                return np.array([])
            return data
        except BinanceAPIException as e:
            logger.error(f"Binance API Error in get_historical_prices_short_interval: {e}")
            return np.array([])
        except Exception as e:
            logger.error(f"Error in get_historical_prices_short_interval: {e}")
            return np.array([])

    def get_current_price(self):
        try:
            self.rate_limiter.wait_if_needed(1)
            order_book = client.get_order_book(symbol=self.symbol, limit=5)
            bid_price = float(order_book['bids'][0][0])
            ask_price = float(order_book['asks'][0][0])
            mid_price = (bid_price + ask_price) / 2
            return {
                'bid': bid_price,
                'ask': ask_price,
                'mid': mid_price,
                'spread': ask_price - bid_price,
                'spread_pct': (ask_price - bid_price) / bid_price,
                'depth': len(order_book['bids']) + len(order_book['asks'])
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None

    def get_account_balances(self):
        try:
            if self.test_mode:
                if not self.symbol_info:
                    base_asset = self.symbol[:-4]
                    quote_asset = self.symbol[-4:]
                else:
                    base_asset = self.symbol_info['baseAsset']
                    quote_asset = self.symbol_info['quoteAsset']
                return {
                    'base': {'asset': base_asset, 'free': 10.0},
                    'quote': {'asset': quote_asset, 'free': 10000.0}
                }
            self.rate_limiter.wait_if_needed(10)
            account = client.get_account()
            if not self.symbol_info:
                base_asset = self.symbol[:-4]
                quote_asset = self.symbol[-4:]
            else:
                base_asset = self.symbol_info['baseAsset']
                quote_asset = self.symbol_info['quoteAsset']
            base_balance = 0
            quote_balance = 0
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    base_balance = float(balance['free'])
                elif balance['asset'] == quote_asset:
                    quote_balance = float(balance['free'])
            return {
                'base': {'asset': base_asset, 'free': base_balance},
                'quote': {'asset': quote_asset, 'free': quote_balance}
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting account balances: {e}")
            return None

    def cancel_all_orders(self):
        try:
            if self.test_mode:
                logger.info(f"[TEST MODE] Cancelling all orders for {self.symbol}")
                self.active_orders = {'buy': [], 'sell': []}
                return True
            self.rate_limiter.wait_if_needed(1)
            open_orders = client.get_open_orders(symbol=self.symbol)
            for order in open_orders:
                self.rate_limiter.wait_if_needed(1)
                client.cancel_order(symbol=self.symbol, orderId=order['orderId'])
                logger.info(f"Cancelled order {order['orderId']}")
            self.active_orders = {'buy': [], 'sell': []}
            return True
        except BinanceAPIException as e:
            logger.error(f"Binance API Error in cancel_all_orders: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in cancel_all_orders: {e}")
            return False

    def place_order(self, side, price, quantity):
        try:
            formatted_price = self.format_price(price)
            formatted_quantity = self.format_quantity(quantity)
            if self.symbol_info:
                min_notional = float(self.symbol_info.get('minNotional', 0))
                if formatted_price * formatted_quantity < min_notional:
                    logger.warning(f"Order value {formatted_price * formatted_quantity} below minimum notional {min_notional}")
                    return None
            if self.test_mode:
                logger.info(f"[TEST MODE] Placing {side} order: {formatted_quantity} {self.symbol} @ {formatted_price}")
                order_id = f"test_{side}_{int(time.time() * 1000)}"
                order = {
                    'orderId': order_id,
                    'price': formatted_price,
                    'origQty': formatted_quantity,
                    'executedQty': 0,
                    'status': 'NEW',
                    'side': side.lower()
                }
                self.active_orders[side.lower()].append(order)
                return order_id
            self.rate_limiter.wait_if_needed(1)
            order = client.create_order(
                symbol=self.symbol,
                side=side.upper(),
                type=Client.ORDER_TYPE_LIMIT,
                timeInForce=Client.TIME_IN_FORCE_GTC,
                quantity=formatted_quantity,
                price=str(formatted_price)
            )
            logger.info(f"Placed {side} order: {formatted_quantity} {self.symbol} @ {formatted_price}")
            self.active_orders[side.lower()].append(order)
            return order['orderId']
        except BinanceAPIException as e:
            logger.error(f"Binance API Error in place_order: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in place_order: {e}")
            return None

    def create_ladder_orders(self, predicted_price):
        market = self.get_current_price()
        if not market:
            logger.error("Failed to get market data, skipping order creation")
            return False
        balances = self.get_account_balances()
        if not balances:
            logger.error("Failed to get account balances, skipping order creation")
            return False
        historical_data = np.array(self.recent_data)
        if len(historical_data) == 0:
            logger.error("No recent data available for spread calculation")
            return False
        if not self.cancel_all_orders():
            logger.error("Failed to cancel existing orders, skipping order creation")
            return False
        
        self.shared_data['balances'] = balances
        mid_price = market['mid']
        if self.model_ready and predicted_price:
            mid_price = (0.7 * mid_price) + (0.3 * predicted_price)
            logger.info(f"Adjusted mid price from {market['mid']} to {mid_price} based on prediction {predicted_price}")
        
        indicators = calculate_indicators(historical_data)
        order_book_depth = market['depth']
        dynamic_spread = calculate_dynamic_spread(mid_price, indicators, order_book_depth)
        dynamic_spread = max(self.min_spread, min(self.max_spread, dynamic_spread))
        logger.info(f"Dynamic spread calculated: {dynamic_spread:.6f}")

        spread_range = self.max_spread - dynamic_spread
        spread_step = spread_range / (self.max_orders - 1) if self.max_orders > 1 else 0

        for i in range(self.max_orders):
            current_spread = dynamic_spread + (i * spread_step)
            bid_price = mid_price * (1 - current_spread)
            quantity_multiplier = 1 + (i / self.max_orders)
            order_quantity = self.order_quantity * quantity_multiplier
            bid_price = self.format_price(bid_price)
            order_quantity = self.format_quantity(order_quantity)
            cost = bid_price * order_quantity
            if self.test_mode or cost <= balances['quote']['free']:
                order_id = self.place_order('BUY', bid_price, order_quantity)
                if order_id and not self.test_mode:
                    profit = 0  # Placeholder; real profit needs execution data
                    if not self.risk_manager.check_risk(-cost):
                        logger.warning(f"Risk limit exceeded for BUY order {order_id}, cancelling...")
                        self.cancel_all_orders()
                        return False
            else:
                logger.warning(f"Insufficient {balances['quote']['asset']} for buy order: {cost}")

        for i in range(self.max_orders):
            current_spread = dynamic_spread + (i * spread_step)
            ask_price = mid_price * (1 + current_spread)
            quantity_multiplier = 1 + (i / self.max_orders)
            order_quantity = self.order_quantity * quantity_multiplier
            ask_price = self.format_price(ask_price)
            order_quantity = self.format_quantity(order_quantity)
            if self.test_mode or order_quantity <= balances['base']['free']:
                order_id = self.place_order('SELL', ask_price, order_quantity)
                if order_id and not self.test_mode:
                    profit = 0  # Placeholder
                    if not self.risk_manager.check_risk(0):
                        logger.warning(f"Risk limit exceeded for SELL order {order_id}, cancelling...")
                        self.cancel_all_orders()
                        return False
            else:
                logger.warning(f"Insufficient {balances['base']['asset']} for sell order: {order_quantity}")

        self.shared_data['active_orders'] = self.active_orders
        return True

    def check_and_update_fills(self):
        if self.test_mode:
            for side in ['buy', 'sell']:
                i = 0
                while i < len(self.active_orders[side]):
                    order = self.active_orders[side][i]
                    if order['executedQty'] == 0 and random.random() < 0.05:
                        fill_percent = random.uniform(0.1, 1.0)
                        order['executedQty'] = order['origQty'] * fill_percent
                        order['status'] = 'PARTIALLY_FILLED'
                        logger.info(f"[TEST MODE] Order {order['orderId']} partially filled: {order['executedQty']}/{order['origQty']}")
                        trade_profit = (order['price'] * order['executedQty']) if side == 'sell' else -(order['price'] * order['executedQty'])
                        if not self.risk_manager.check_risk(trade_profit):
                            logger.warning(f"Risk limit exceeded in test mode for {side} order {order['orderId']}")
                            self.cancel_all_orders()
                            return
                        i += 1
                    elif order['executedQty'] < order['origQty'] and random.random() < 0.02:
                        remaining_qty = order['origQty'] - order['executedQty']
                        order['executedQty'] = order['origQty']
                        order['status'] = 'FILLED'
                        logger.info(f"[TEST MODE] Order {order['orderId']} filled completely")
                        trade_profit = (order['price'] * remaining_qty) if side == 'sell' else -(order['price'] * remaining_qty)
                        if not self.risk_manager.check_risk(trade_profit):
                            logger.warning(f"Risk limit exceeded in test mode for {side} order {order['orderId']}")
                            self.cancel_all_orders()
                            return
                        self.active_orders[side].pop(i)
                    else:
                        i += 1
            return
        try:
            self.rate_limiter.wait_if_needed(1)
            open_orders = client.get_open_orders(symbol=self.symbol)
            current_order_ids = set(order['orderId'] for order in open_orders)
            for side in ['buy', 'sell']:
                for i in range(len(self.active_orders[side]) - 1, -1, -1):
                    order = self.active_orders[side][i]
                    if order['orderId'] not in current_order_ids:
                        executed_qty = float(order['origQty'])
                        trade_profit = (order['price'] * executed_qty) if side == 'sell' else -(order['price'] * executed_qty)
                        if not self.risk_manager.check_risk(trade_profit):
                            logger.warning(f"Risk limit exceeded for {side} order {order['orderId']}")
                            self.cancel_all_orders()
                            return
                        self.active_orders[side].pop(i)
            updated_balances = self.get_account_balances()
            if updated_balances:
                self.shared_data['balances'] = updated_balances
                self.risk_manager.set_balance(self.shared_data['balances']['quote']['free'])
        except BinanceAPIException as e:
            logger.error(f"Binance API Error in check_and_update_fills: {e}")
        except Exception as e:
            logger.error(f"Error in check_and_update_fills: {e}")

    def predict_price(self):
        if not self.model_ready:
            logger.info("Model not ready yet, using last price as prediction")
            current_price = self.get_current_price()
            return current_price['mid'] if current_price else None
        historical_data = np.array(self.recent_data)
        if len(historical_data) < self.sequence_length:
            logger.error(f"Not enough buffered data for prediction. Required: {self.sequence_length}, Available: {len(historical_data)}")
            current_price = self.get_current_price()
            return current_price['mid'] if current_price else None
        return predict_future_price(self.model, self.scaler, historical_data, self.sequence_length)

    def save_state(self, filename='market_maker_state.json'):
        try:
            state = {
                'symbol': self.symbol,
                'active_orders': self.active_orders,
                'historical_prices': self.shared_data['historical_prices'],
                'timestamps': self.shared_data['timestamps'],
                'last_update': int(time.time())
            }
            with open(filename, 'w') as f:
                json.dump(state, f)
            logger.info(f"State saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self, filename='market_maker_state.json'):
        try:
            if not os.path.exists(filename):
                logger.info(f"No state file found at {filename}")
                return False
            with open(filename, 'r') as f:
                state = json.load(f)
            if state.get('symbol') != self.symbol:
                logger.warning(f"State file is for symbol {state.get('symbol')}, but current symbol is {self.symbol}")
                return False
            if 'historical_prices' in state and len(state['historical_prices']) > 0:
                self.shared_data['historical_prices'] = state['historical_prices']
            if 'timestamps' in state and len(state['timestamps']) > 0:
                self.shared_data['timestamps'] = state['timestamps']
            logger.info(f"State loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

    def run(self):
        try:
            self.load_state()
            start_time = time.time()
            while not self.model_ready and (time.time() - start_time) < 300:
                logger.info("Waiting for model to train...")
                time.sleep(5)
            if not self.model_ready:
                logger.warning("Model training timeout or failed, proceeding with basic strategy")

            last_save_time = time.time()
            last_order_update = 0
            max_retries = 5
            retry_count = 0

            while retry_count < max_retries:
                try:
                    current_time = time.time()

                    if current_time - self.last_retrain_time >= self.retrain_interval and self.model_ready:
                        retrain_thread = threading.Thread(target=self.retrain_model)
                        retrain_thread.daemon = True
                        retrain_thread.start()

                    current_market = self.get_current_price()
                    if not current_market:
                        logger.error("Failed to get current market price")
                        retry_count += 1
                        time.sleep(5)
                        continue

                    market_price = current_market['mid']
                    avg_volume = np.mean([row[4] for row in list(self.recent_data) if row[4] > 0]) or 1.0
                    new_data_point = np.array([market_price, market_price, market_price, market_price, avg_volume])
                    self.recent_data.append(new_data_point)

                    with self.data_lock:
                        self.shared_data['historical_prices'].append(market_price)
                        self.shared_data['market_prices'].append(market_price)
                        self.shared_data['timestamps'].append(int(current_time * 1000))
                        if len(self.shared_data['historical_prices']) > 100:
                            self.shared_data['historical_prices'] = self.shared_data['historical_prices'][-100:]
                            self.shared_data['market_prices'] = self.shared_data['market_prices'][-100:]
                            self.shared_data['timestamps'] = self.shared_data['timestamps'][-100:]

                    predicted_price = self.predict_price()
                    with self.data_lock:
                        self.shared_data['predicted_prices'].append(predicted_price)
                        if len(self.shared_data['predicted_prices']) > 100:
                            self.shared_data['predicted_prices'] = self.shared_data['predicted_prices'][-100:]

                    self.check_and_update_fills()

                    if current_time - last_order_update >= self.order_refresh_time:
                        logger.info(f"Updating orders: Market Price={market_price:.2f}, Predicted={predicted_price:.2f}")
                        if self.create_ladder_orders(predicted_price):
                            last_order_update = current_time
                        else:
                            logger.error("Failed to create ladder orders")
                            retry_count += 1
                            time.sleep(5)
                            continue

                    if current_time - last_save_time >= 300:
                        self.save_state()
                        last_save_time = current_time

                    retry_count = 0
                    time.sleep(1)

                except Exception as inner_e:
                    logger.error(f"Error in main loop iteration: {inner_e}")
                    retry_count += 1
                    time.sleep(5)

            logger.error(f"Max retries ({max_retries}) exceeded, shutting down")
            self.cancel_all_orders()
            self.save_state()

        except KeyboardInterrupt:
            logger.info("Shutting down market maker...")
            self.cancel_all_orders()
            self.save_state()
            logger.info("Shutdown complete")
        except Exception as e:
            logger.error(f"Fatal error in run: {e}")
            self.cancel_all_orders()
            self.save_state()

if __name__ == "__main__":
    try:
        client.get_system_status()
        logger.info("Successfully connected to Binance API")
    except Exception as e:
        logger.error(f"Failed to connect to Binance API: {e}")
        exit(1)

    market_maker = CryptoMarketMaker(
        symbol="BTCUSDT",
        order_quantity=0.001,
        max_orders=5,
        min_spread=0.001,
        max_spread=0.005,
        order_refresh_time=60,
        test_mode=True,
        historical_data_days=30,
        retrain_interval=3600
    )

    # Run backtesting before starting the market maker
    historical_data = market_maker.get_historical_prices()
    if len(historical_data) > 0:
        backtester = Backtester(historical_data[:, 3], initial_balance=10000)
        final_balance, trades = backtester.run_backtest(lambda price: "BUY" if random.random() > 0.5 else "SELL")
        logger.info(f"Backtesting complete. Final balance: {final_balance}, Trades: {len(trades)}")
    else:
        logger.error("No historical data available for backtesting")

    market_maker.run()