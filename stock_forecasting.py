#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stock Forecasting Module
=======================

This module implements PyTorch-based forecasting models for stock price prediction.

Features:
- LSTM, GRU, and Transformer models for time series forecasting
- Ensemble methods combining multiple models
- Hyperparameter optimization
- Confidence intervals for predictions
- Integration with user preferences
- Visualization of forecasting results
"""

import os
import sys
import random
import pandas as pd
import numpy as np
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reproducibility and cudnn tuning
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """
    
    def __init__(self, X, y):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        """
        Get the number of samples.
        
        Returns:
        --------
        int
            Number of samples
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Parameters:
        -----------
        idx : int
            Index of the sample
            
        Returns:
        --------
        tuple
            Tuple of (X, y) for the sample
        """
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of hidden units
        num_layers : int
            Number of LSTM layers
        output_size : int
            Number of output features
        dropout : float, optional
            Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Linear layer
        out = self.fc(out)
        
        return out

class GRUModel(nn.Module):
    """
    GRU model for time series forecasting.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the GRU model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of hidden units
        num_layers : int
            Number of GRU layers
        output_size : int
            Number of output features
        dropout : float, optional
            Dropout rate
        """
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Linear layer
        out = self.fc(out)
        
        return out

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the positional encoding.
        
        Parameters:
        -----------
        d_model : int
            Dimension of the model
        dropout : float, optional
            Dropout rate
        max_len : int, optional
            Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    Transformer model for time series forecasting.
    """
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.2):
        """
        Initialize the Transformer model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        d_model : int
            Dimension of the model
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        output_size : int
            Number of output features
        dropout : float, optional
            Dropout rate
        """
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, output_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # Reshape input for transformer: (seq_len, batch_size, input_size)
        x = x.permute(1, 0, 2)
        
        # Embed input
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Get the output from the last time step
        x = x[-1, :, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Linear layer
        x = self.fc(x)
        
        return x

class StockForecaster:
    """
    Stock price forecaster using PyTorch models.
    """
    
    def __init__(self, preferences_manager=None):
        """
        Initialize the stock forecaster.
        
        Parameters:
        -----------
        preferences_manager : UserPreferencesManager, optional
            Manager for user preferences
        """
        self.preferences_manager = preferences_manager
        
        # Set default forecasting parameters if no preferences manager
        if preferences_manager is None:
            self.forecasting_params = {
                'horizon': 30,  # Days to forecast
                'models': ['lstm', 'gru', 'ensemble'],
                'use_sentiment_analysis': False,
                'confidence_interval': 0.95,
                'monte_carlo_simulations': 1000,
                'sequence_length': 60,  # Days of historical data to use for each prediction
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 10,
                'learning_rate': 0.001,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        else:
            self.forecasting_params = preferences_manager.get_forecast_params()
            # Add default values for parameters not in preferences
            default_params = {
                'sequence_length': 60,
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 10,
                'learning_rate': 0.001,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            for key, value in default_params.items():
                if key not in self.forecasting_params:
                    self.forecasting_params[key] = value
        
        # Set device
        self.device = torch.device(self.forecasting_params['device'])
        
        logger.info(f"Stock forecaster initialized (using {self.device})")
    
    def prepare_data(self, prices_df, ticker):
        """
        Prepare data for forecasting.
        
        Parameters:
        -----------
        prices_df : pandas.DataFrame
            DataFrame with prices for all tickers
        ticker : str
            Ticker symbol to forecast
            
        Returns:
        --------
        tuple
            Tuple of (X_train, X_val, y_train, y_val, scaler)
        """
        logger.info(f"Preparing data for {ticker}")
        
        # Extract price series for the ticker
        price_series = prices_df[ticker].dropna()
        
        # Check if we have enough data
        min_len = self.forecasting_params['sequence_length'] + self.forecasting_params['horizon']
        if len(price_series) < min_len:
            logger.error(
                f"Not enough data for {ticker} "
                f"(need at least {min_len} days)"
            )
            return None
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(price_series.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        seq_len = self.forecasting_params['sequence_length']
        horizon = self.forecasting_params['horizon']
        for i in range(len(scaled_data) - seq_len - horizon + 1):
            X.append(scaled_data[i : i + seq_len])
            y.append(scaled_data[i + seq_len : i + seq_len + horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data into train (85%) and validation (15%)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.15,     # 15% validation
            shuffle=False       # preserve time ordering
        )
        
        logger.info(
            f"Data prepared for {ticker}: "
            f"{X_train.shape[0]} train, {X_val.shape[0]} validation samples"
        )
        
        return X_train, X_val, y_train, y_val, scaler

    
    def train_model(self, X_train, y_train, X_val, y_val, model_type='lstm'):
        """
        Train a forecasting model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training input features
        y_train : numpy.ndarray
            Training target values
        X_val : numpy.ndarray
            Validation input features
        y_val : numpy.ndarray
            Validation target values
        model_type : str, optional
            Type of model to train ('lstm', 'gru', 'transformer', or 'xgboost')
            
        Returns:
        --------
        tuple
            Tuple of (model, training_history)
        """
        logger.info(f"Training {model_type.upper()} model")
        
        # --- XGBoost Training ---
        if model_type == 'xgboost':
            # Reshape data for XGBoost (samples, features)
            n_samples, n_steps, n_features = X_train.shape
            X_train_reshaped = X_train.reshape((n_samples, n_steps * n_features))
            y_train_reshaped = y_train.squeeze(axis=-1)

            n_samples_val, _, _ = X_val.shape
            X_val_reshaped = X_val.reshape((n_samples_val, n_steps * n_features))
            y_val_reshaped = y_val.squeeze(axis=-1)
            
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.05,
                early_stopping_rounds=10,
                n_jobs=-1
            )
            
            logger.info("--- XGBoost Training Log ---")
            model.fit(
                X_train_reshaped,
                y_train_reshaped,
                eval_set=[(X_val_reshaped, y_val_reshaped)],
                verbose=False
            )
            
            logger.info("XGBoost model trained successfully.")
            return model, {}  # XGBoost fit does not return a history object

        # --- PyTorch Model Training ---
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset   = TimeSeriesDataset(X_val,   y_val)
        
        # DataLoader performance settings
        pin_memory = (self.device.type == 'cuda')
        num_workers = min(4, os.cpu_count() or 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.forecasting_params['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.forecasting_params['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Instantiate model
        input_size  = X_train.shape[2]
        output_size = y_train.shape[1]
        
        if model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.forecasting_params['hidden_size'],
                num_layers=self.forecasting_params['num_layers'],
                output_size=output_size,
                dropout=self.forecasting_params['dropout']
            )
        elif model_type == 'gru':
            model = GRUModel(
                input_size=input_size,
                hidden_size=self.forecasting_params['hidden_size'],
                num_layers=self.forecasting_params['num_layers'],
                output_size=output_size,
                dropout=self.forecasting_params['dropout']
            )
        elif model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=self.forecasting_params['hidden_size'],
                nhead=4,
                num_layers=self.forecasting_params['num_layers'],
                output_size=output_size,
                dropout=self.forecasting_params['dropout']
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None, None
        
        model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.forecasting_params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        scaler_amp = torch.cuda.amp.GradScaler() if pin_memory else None
        
        epochs   = self.forecasting_params['epochs']
        patience = self.forecasting_params['early_stopping_patience']
        
        best_val_loss   = float('inf')
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"--- {model_type.upper()} Training Log ---")
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=pin_memory):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.squeeze(-1))
                if scaler_amp:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch.squeeze(-1)).item()
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        logger.info(f"{model_type.upper()} model trained successfully")
        return model, history

    def forecast(self, prices_df, ticker):
        """
        Forecast future prices for a ticker.
        """
        logger.info(f"Forecasting prices for {ticker}")
        
        # Prepare data for training and evaluation
        data = self.prepare_data(prices_df, ticker)
        if data is None:
            logger.error(f"Failed to prepare data for {ticker}")
            return None
        
        # Unpack only the train/val splits (no test set)
        X_train, X_val, y_train, y_val, scaler = data
        
        # --- training and prediction logic unchanged below ---
        models    = {}
        histories = {}
        for model_type in self.forecasting_params['models']:
            if model_type != 'ensemble':
                model, history = self.train_model(X_train, y_train, X_val, y_val, model_type)
                if model is not None:
                    models[model_type]    = model
                    histories[model_type] = history
        
        if not models:
            logger.error(f"All models failed to train for {ticker}.")
            return None

        # Prepare last sequence for real forecast
        price_series = prices_df[ticker].dropna()
        full_scaled_data = scaler.transform(price_series.values.reshape(-1, 1))
        last_sequence_for_forecast = full_scaled_data[-self.forecasting_params['sequence_length']:]
        
        # Make predictions
        predictions = {}
        for model_type, model in models.items():
            last_seq = last_sequence_for_forecast.reshape(1, self.forecasting_params['sequence_length'], 1)
            if model_type == 'xgboost':
                n_s, n_t, n_f = last_seq.shape
                xgb_input = last_seq.reshape((n_s, n_t * n_f))
                pred_raw = model.predict(xgb_input)[0]
                pred = np.array([pred_raw])
            else:
                model.eval()
                t_in = torch.tensor(last_seq, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    out = model(t_in).cpu().numpy().flatten()
                pred = out
            predictions[model_type] = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        
        # Ensemble prediction
        if 'ensemble' in self.forecasting_params['models'] and len(predictions) > 1:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            predictions['ensemble'] = ensemble_pred
        
        # Confidence intervals
        ci = {}
        for model_type, model in models.items():
            last_seq = last_sequence_for_forecast.reshape(1, self.forecasting_params['sequence_length'], 1)
            if model_type == 'xgboost':
                base = predictions[model_type]
                ci[model_type] = {
                    'lower': base * 0.95,
                    'upper': base * 1.05
                }
            else:
                model.train()  # enable dropout
                sims = self.forecasting_params['monte_carlo_simulations']
                mc_preds = []
                t_in = torch.tensor(last_seq, dtype=torch.float32).to(self.device)
                for _ in range(sims):
                    with torch.no_grad():
                        out = model(t_in).cpu().numpy().flatten()
                    mc_preds.append(scaler.inverse_transform(out.reshape(-1,1)).flatten())
                mc_arr = np.array(mc_preds)
                alpha = self.forecasting_params['confidence_interval']
                lower = np.percentile(mc_arr, (1-alpha)*50, axis=0)
                upper = np.percentile(mc_arr, 100-(1-alpha)*50, axis=0)
                ci[model_type] = {'lower': lower, 'upper': upper}
        
        # Ensemble CI
        if 'ensemble' in predictions:
            lowers = [ci[m]['lower'] for m in ci if m!='xgboost']
            uppers = [ci[m]['upper'] for m in ci if m!='xgboost']
            if lowers:
                ci['ensemble'] = {
                    'lower': np.mean(lowers, axis=0),
                    'upper': np.mean(uppers, axis=0)
                }
            else:
                ci['ensemble'] = ci.get('xgboost', {
                    'lower': predictions['ensemble'],
                    'upper': predictions['ensemble']
                })
        
        # Assemble results
        last_date = prices_df[ticker].index[-1]
        forecast_dates = [
            (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            for i in range(self.forecasting_params['horizon'])
        ]
        
        result = {
            'ticker': ticker,
            'last_price': prices_df[ticker].iloc[-1],
            'last_date': last_date.strftime('%Y-%m-%d'),
            'forecast_dates': forecast_dates,
            'forecasts': {}
        }
        for model_type in self.forecasting_params['models']:
            if model_type in predictions and model_type in ci:
                result['forecasts'][model_type] = {
                    'mean': predictions[model_type].tolist(),
                    'lower': ci[model_type]['lower'].tolist(),
                    'upper': ci[model_type]['upper'].tolist()
                }
        
        logger.info(f"Forecasting completed for {ticker}")
        return result

    def forecast_multiple(self, prices_df: pd.DataFrame, tickers: list) -> dict:
        """
        Forecast future prices for multiple tickers.

        Parameters:
        -----------
        prices_df : pandas.DataFrame
            DataFrame with prices for all tickers (columns are tickers)
        tickers : list
            List of ticker symbols to forecast

        Returns:
        --------
        dict
            Dictionary with forecasting results for each ticker
        """
        logger.info(f"Attempting to forecast prices for {len(tickers)} tickers")
        results = {}

        for ticker in tickers:
            # 1) Ensure ticker exists
            if ticker not in prices_df.columns:
                logger.warning(f"Skipping forecast for '{ticker}': no price series found.")
                continue

            # 2) Ensure there's at least some non-NaN data
            if prices_df[ticker].isnull().all():
                logger.warning(f"Skipping forecast for '{ticker}': price series is all NaN.")
                continue

            # 3) Safe to pull history
            historical = prices_df[ticker].dropna()

            try:
                # 4) Run your single‐ticker forecast
                result = self.forecast(prices_df, ticker)
                if result is None:
                    continue

                # 5) Attach the historical series for plotting
                result['historical_prices'] = historical

                # 6) Store the completed result
                results[ticker] = result

            except Exception as e:
                logger.error(f"An unexpected error occurred during forecast for {ticker}: {e}")

        logger.info(
            f"Forecasting completed for {len(results)} out of {len(tickers)} requested tickers."
        )
        return results
