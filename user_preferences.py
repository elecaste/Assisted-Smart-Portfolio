#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
User Preferences Manager Module
==============================

This module handles user preferences collection and management through
an interactive questionnaire. It stores and provides access to user
preferences for other modules.

Features:
- Interactive questionnaire for user preferences
- Risk profile assessment
- Investment horizon determination
- Sector and asset preferences
- Optimization parameter customization
"""

import os
import sys
import torch
import json
import random
import logging
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import financedatabase as fd
import pickle
import yfinance as yf


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_valid_ticker(ticker, period="1y"):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info or "shortName" not in info:
            return False
        hist = t.history(period=period)
        return not hist.empty
    except Exception as e:
        logger.warning(f"Validation failed for ticker {ticker}: {e}")
        return False


class UserPreferencesManager:
    """
    Manages user preferences for portfolio scanning and optimization.
    """
    
    def __init__(self):
        """
        Initialize the user preferences manager.
        """
        # Default preferences
        self.preferences = {
            'risk_profile': 'moderate',  # 'conservative', 'moderate', 'aggressive'
            'investment_horizon': 'medium',  # 'short', 'medium', 'long'
            'preferred_sectors': [],  # List of preferred sectors
            'excluded_sectors': [],  # List of excluded sectors
            'preferred_assets': [],  # List of preferred assets
            'excluded_assets': [],  # List of excluded assets
            'benchmark': '^GSPC',  # Default benchmark (S&P 500)
            'risk_free_rate': 0.02,  # Annual risk-free rate
            'optimization_method': 'efficient_frontier',  # 'efficient_frontier', 'risk_parity', 'min_volatility', 'max_sharpe'
            'min_weight': 0.01,  # Minimum weight per asset
            'max_weight': 0.4,  # Maximum weight per asset
            'target_return': None,  # Target return for efficient frontier
            'target_volatility': None,  # Target volatility for efficient frontier
            'sector_constraints': {},  # Sector constraints
            'asset_constraints': {},  # Asset-specific constraints
            'data_sources': ['yahoo', 'alphavantage', 'fmp'],  # Data sources in order of preference
            'api_keys': {  # API keys
                'alphavantage': None,
                'fmp': None,
                'finnhub': None
            },
            'forecast_horizon': 30,  # Forecast horizon in days
            'forecast_models': ['lstm', 'gru'],  # Forecast models to use
            'num_forecast_tickers': 5, # Number of tickers to forecast
            'report_formats': ['html'],  # Report formats to generate
            'selected_tickers': []  # Tickers selected by the user
        }
        
        # Questionnaire responses
        self.responses = {}
        
        logger.info("User preferences manager initialized")
    
    async def collect_preferences(self):
        """
        Collect user preferences through an interactive questionnaire.
        
        Returns:
        --------
        dict
            Dictionary with user preferences
        """
        logger.info("Collecting user preferences")
        
        # Welcome message
        print("\n" + "=" * 80)
        print("Welcome to Portfolio Scanner Advanced")
        print("Please answer the following questions to customize your portfolio analysis")
        print("=" * 80 + "\n")
        
        # Risk profile
        await self._ask_risk_profile()
        
        # Investment horizon
        await self._ask_investment_horizon()
        
        # Ticker selection (new dynamic method)
        await self._ask_dynamic_ticker_selection()
        
        # Benchmark
        await self._ask_benchmarks()
        
        # Optimization method
        await self._ask_optimization_method()
        
        # Data sources
        await self._ask_data_sources()
        
        # API keys
        # await self._ask_api_keys()
        
        # Forecast preferences
        await self._ask_forecast_preferences()
        
        # Thank you message
        print("\n" + "=" * 80)
        print("Thank you for your responses!")
        print("Your preferences have been saved and will be used for portfolio analysis")
        print("=" * 80 + "\n")
        
        logger.info("User preferences collected")
        
        return self.preferences
    
    async def _ask_risk_profile(self):
        """
        Ask user about risk profile.
        """
        print("\n--- Risk Profile ---")
        print("1. Conservative: Lower risk, lower potential returns")
        print("2. Moderate: Balanced risk and potential returns")
        print("3. Aggressive: Higher risk, higher potential returns")
        
        while True:
            try:
                response = input("Select your risk profile (1-3): ").strip()
                
                if response == '1':
                    self.preferences['risk_profile'] = 'conservative'
                    break
                elif response == '2':
                    self.preferences['risk_profile'] = 'moderate'
                    break
                elif response == '3':
                    self.preferences['risk_profile'] = 'aggressive'
                    break
                else:
                    print("Invalid response. Please enter a number between 1 and 3.")
            
            except Exception as e:
                logger.warning(f"Error asking risk profile: {e}")
                print("An error occurred. Please try again.")
        
        self.responses['risk_profile'] = self.preferences['risk_profile']
        
        # Update optimization parameters based on risk profile
        if self.preferences['risk_profile'] == 'conservative':
            self.preferences['min_weight'] = 0.02
            self.preferences['max_weight'] = 0.2
            self.preferences['optimization_method'] = 'min_volatility'
        elif self.preferences['risk_profile'] == 'moderate':
            self.preferences['min_weight'] = 0.01
            self.preferences['max_weight'] = 0.3
            self.preferences['optimization_method'] = 'efficient_frontier'
        elif self.preferences['risk_profile'] == 'aggressive':
            self.preferences['min_weight'] = 0.0
            self.preferences['max_weight'] = 0.4
            self.preferences['optimization_method'] = 'max_sharpe'
    
    async def _ask_investment_horizon(self):
        """
        Ask user about investment horizon.
        """
        print("\n--- Investment Horizon ---")
        print("1. Short-term: Less than 1 year")
        print("2. Medium-term: 1-5 years")
        print("3. Long-term: More than 5 years")
        
        while True:
            try:
                response = input("Select your investment horizon (1-3): ").strip()
                
                if response == '1':
                    self.preferences['investment_horizon'] = 'short'
                    break
                elif response == '2':
                    self.preferences['investment_horizon'] = 'medium'
                    break
                elif response == '3':
                    self.preferences['investment_horizon'] = 'long'
                    break
                else:
                    print("Invalid response. Please enter a number between 1 and 3.")
            
            except Exception as e:
                logger.warning(f"Error asking investment horizon: {e}")
                print("An error occurred. Please try again.")
        
        self.responses['investment_horizon'] = self.preferences['investment_horizon']
        
        # Update data parameters based on investment horizon
        if self.preferences['investment_horizon'] == 'short':
            self.preferences['period'] = '1y'
            self.preferences['forecast_horizon'] = 30
        elif self.preferences['investment_horizon'] == 'medium':
            self.preferences['period'] = '3y'
            self.preferences['forecast_horizon'] = 90
        elif self.preferences['investment_horizon'] == 'long':
            self.preferences['period'] = '5y'
            self.preferences['forecast_horizon'] = 180
    
    async def _ask_dynamic_ticker_selection(self):
        """
        Ask user whether to enter their own comma-separated tickers or generate one via filters.
        Validates any manually entered tickers against yfinance data.
        """
        print("\n--- Ticker Selection ---")
        # Let user choose manual entry or filter-based generation; default to filter (2)
        choice = input(
            "Would you like to (1) enter your own tickers or (2) generate via filters? [1/2, default 2]: "
        ).strip()
        if not choice:
            choice = '2'
        while choice not in ('1', '2'):
            print("Invalid choice. Enter 1 or 2.")
            choice = input(
                "Would you like to (1) enter your own tickers or (2) generate via filters? [1/2, default 2]: "
            ).strip()
            if not choice:
                choice = '2'

        if choice == '1':
            # Manual entry path
            while True:
                raw = input("Enter comma-separated tickers: ").strip()
                tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]
                valid = []
                print("\nValidating each ticker via yfinance:")
                for t in tickers:
                    print(f" - Checking {t}...", end=' ')
                    if is_valid_ticker(t, period=self.preferences.get('period', '1y')):
                        print("valid")
                        valid.append(t)
                    else:
                        print("invalid")
                if valid:
                    print(f"\nValidated tickers: {', '.join(valid)}")
                    self.preferences['selected_tickers'] = valid
                    break
                print("\nNone of those tickers returned data. Please try again.")
        else:
            # Filter-based generation path
            print("Please select filters to generate a list of stocks for your portfolio.")

            # Define cache path for the equities database
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            equities_cache_path = os.path.join(cache_dir, 'equities_database.pkl')

            equities = None
            # Check if a cached version of the database exists
            if os.path.exists(equities_cache_path):
                response = input("Found a local cache of the equities database. Use it? (Y/n): ").strip().lower()
                if response in ('', 'y', 'yes'):
                    try:
                        print("Loading equities database from cache...")
                        with open(equities_cache_path, 'rb') as f:
                            equities = pickle.load(f)
                        print("Loaded successfully.")
                    except Exception as e:
                        print(f"Could not load cache due to an error: {e}. Re-downloading...")
                        equities = None

            # If cache is not used or doesn't exist, download the database
            if equities is None:
                print("\nFetching equities database. This may take several minutes on the first run...")
                try:
                    equities = fd.Equities()
                    with open(equities_cache_path, 'wb') as f:
                        pickle.dump(equities, f)
                    print("Equities database downloaded and cached.")
                except Exception as e:
                    logger.error(f"Failed to download from financedatabase: {e}")
                    print("Could not download the equities database. Please check your internet connection.")
                    print("Falling back to manual ticker entry.")
                    await self._ask_dynamic_ticker_selection()
                    return

            try:
                options = fd.show_options("equities")

                def get_selections(item_type, item_list):
                    print(f"\nAvailable {item_type}s:")
                    for i, item in enumerate(item_list, 1):
                        print(f"{i}. {item}")
                    while True:
                        resp = input(f"Select one or more {item_type}s (e.g., 1,5) or press Enter for ALL: ").strip()
                        if not resp:
                            return item_list
                        try:
                            picks = [item_list[int(i) - 1] for i in resp.split(',')]
                            return picks
                        except Exception:
                            print("Invalid input. Please enter comma-separated numbers.")

                # 1. Country
                countries = sorted(options['country'])
                selected_countries = get_selections("country", countries)
                # 2. Currency
                currencies = sorted(options['currency'])
                selected_currencies = get_selections("currency", currencies)
                # 3. Industry Group
                industry_groups = sorted(options['industry_group'])
                selected_industry_groups = get_selections("industry group", industry_groups)

                # 4. Market Cap Filter (multi-select)
                mc_options = ['All market caps', 'Large Cap', 'Mid Cap', 'Small & Micro Cap']
                selected_mc = get_selections("market cap", mc_options)

                print("\nFiltering for stocks with selected criteria...")
                filtered_equities = equities.select(
                    country=selected_countries,
                    currency=selected_currencies,
                    industry_group=selected_industry_groups
                )

                # Apply multi-select market_cap filter
                if 'market_cap' in filtered_equities.columns:
                    mc_series = filtered_equities['market_cap'].astype(str).str.lower()
                    if 'All market caps' in selected_mc:
                        mask = pd.Series(True, index=filtered_equities.index)
                    else:
                        mask = pd.Series(False, index=filtered_equities.index)
                        if 'Large Cap' in selected_mc:
                            mask |= mc_series == 'large cap'
                        if 'Mid Cap' in selected_mc:
                            mask |= mc_series == 'mid cap'
                        if 'Small & Micro Cap' in selected_mc:
                            mask |= mc_series.isin(['small cap', 'micro cap'])
                    filtered_equities = filtered_equities[mask]

                ticker_list = filtered_equities.index.to_list()
                if not ticker_list:
                    print("No tickers found for the selected criteria. Please try different filters.")
                    await self._ask_dynamic_ticker_selection()
                    return

                print(f"Found {len(ticker_list)} tickers.")

                # Determine how many to select
                max_tickers = len(ticker_list)
                default_n = min(50, max_tickers)
                while True:
                    resp = input(f"Enter the number of tickers to select (1-{max_tickers}, default {default_n}): ").strip()
                    if not resp:
                        n = default_n
                        break
                    try:
                        n = int(resp)
                        if 1 <= n <= max_tickers:
                            break
                    except ValueError:
                        pass
                    print(f"Please enter a valid number between 1 and {max_tickers}.")

                # YFinance validation, one by one
                valid_tickers = []
                print("\nValidating tickers via yfinance:")
                if n < max_tickers:
                    validations = 0
                    shuffled = ticker_list.copy()
                    random.shuffle(shuffled)
                    for ticker in shuffled:
                        if len(valid_tickers) >= n:
                            break
                        validations += 1
                        print(f" - Checking {ticker}...", end=' ')
                        if is_valid_ticker(ticker, period=self.preferences['period']):
                            print("valid")
                            valid_tickers.append(ticker)
                        else:
                            print("invalid")
                    print(f"\nSelected {len(valid_tickers)} valid tickers out of requested {n}.")
                    print(f"Validated {validations} tickers in the process.")
                else:
                    total_validations = 0
                    for ticker in ticker_list:
                        total_validations += 1
                        print(f" - Checking {ticker}...", end=' ')
                        if is_valid_ticker(ticker, period=self.preferences['period']):
                            print("valid")
                            valid_tickers.append(ticker)
                        else:
                            print("invalid")
                    print(f"\nValidated all tickers. Found {len(valid_tickers)} valid out of {total_validations}.")

                if not valid_tickers:
                    print("No valid tickers found. Restarting the selection process...\n")
                    await self._ask_dynamic_ticker_selection()
                    return

                self.preferences['selected_tickers'] = valid_tickers

                # Print first 10 valid tickers
                print("\nFirst 10 valid tickers:")
                for t in valid_tickers[:10]:
                    print(f" - {t}")
                print("...")

            except Exception as e:
                logger.error(f"An error occurred during dynamic ticker selection: {e}")
                print("An error occurred during the filtering process.")
                print("Falling back to manual ticker entry.")
                await self._ask_dynamic_ticker_selection()
                return

        # Record in responses
        self.responses['selected_tickers'] = self.preferences['selected_tickers']



    async def _ask_benchmarks(self):
        """
        Prompt the user to select one or more benchmark tickers by number.
        If no input is given, defaults to all standard benchmarks.
        """
        # Define available default benchmarks and their friendly names
        defaults = [
            ('^GSPC', 'S&P 500'),
            ('^IXIC', 'Nasdaq Composite'),
            ('^DJI', 'Dow Jones Industrial Average'),
            ('^RUT', 'Russell 2000')
        ]

        # Display options to the user
        print("Available default benchmarks:")
        for idx, (ticker, name) in enumerate(defaults, start=1):
            print(f"  {idx}) {name} ({ticker})")
        all_idx = len(defaults) + 1
        custom_idx = len(defaults) + 2
        print(f"  {all_idx}) All of the above")
        print(f"  {custom_idx}) Custom ticker entry")

        # Prompt user for selection
        raw = input(
            f"Enter choices by number separated by commas (e.g. 1,3),\n"
            f"or press Enter to select all defaults [{all_idx}]: "
        ).strip()

        if not raw:
            # Default: all benchmarks
            selected = list(range(1, all_idx))
        else:
            # Parse selected numbers
            try:
                nums = [int(x.strip()) for x in raw.split(',') if x.strip()]
            except ValueError:
                print("Invalid input. Defaulting to all benchmarks.")
                nums = list(range(1, all_idx))
            selected = nums

        benchmarks = []
        for num in selected:
            if 1 <= num <= len(defaults):
                benchmarks.append(defaults[num-1][0])
            elif num == all_idx:
                # all defaults
                benchmarks.extend([t for t, _ in defaults])
            elif num == custom_idx:
                # custom tickers
                custom_raw = input(
                    "Enter custom benchmark tickers separated by commas: "
                ).strip().upper()
                custom_list = [b.strip() for b in custom_raw.split(',') if b.strip()]
                benchmarks.extend(custom_list)
            else:
                print(f"Ignoring invalid choice: {num}")

        # Remove duplicates while preserving order
        seen = set()
        benchmarks = [x for x in benchmarks if not (x in seen or seen.add(x))]

        # Save preferences and responses
        self.preferences['benchmarks'] = benchmarks
        self.responses['benchmarks'] = benchmarks

        # Echo selection with friendly names
        print("\nSelected benchmarks:")
        for ticker in benchmarks:
            # Find friendly name or mark as custom
            friendly = next((name for t, name in defaults if t == ticker), None)
            name = friendly or 'Custom'
            print(f"  {ticker}: {name}")


    async def _ask_optimization_method(self):
        """
        Ask user about optimization method.
        """
        print("\n--- Optimization Method ---")
        print("1. Efficient Frontier (balanced risk/return)")
        print("2. Maximum Sharpe Ratio (optimal risk-adjusted return)")
        print("3. Minimum Volatility (lowest risk)")
        print("4. Risk Parity (equal risk contribution)")
        
        while True:
            try:
                response = input("Select an optimization method (1-4): ").strip()
                
                if response == '1':
                    self.preferences['optimization_method'] = 'efficient_frontier'
                    break
                elif response == '2':
                    self.preferences['optimization_method'] = 'max_sharpe'
                    break
                elif response == '3':
                    self.preferences['optimization_method'] = 'min_volatility'
                    break
                elif response == '4':
                    self.preferences['optimization_method'] = 'risk_parity'
                    break
                else:
                    print("Invalid response. Please enter a number between 1 and 4.")
            
            except Exception as e:
                logger.warning(f"Error asking optimization method: {e}")
                print("An error occurred. Please try again.")
        
        self.responses['optimization_method'] = self.preferences['optimization_method']
        
        # Ask for risk-free rate
        print("\nEnter the annual risk-free rate (as a decimal, e.g., 0.02 for 2%):")
        
        while True:
            try:
                response = input("Risk-free rate (default: 0.02): ").strip()
                
                if not response:
                    self.preferences['risk_free_rate'] = 0.02
                    break
                
                risk_free_rate = float(response)
                
                if 0 <= risk_free_rate <= 0.2:
                    self.preferences['risk_free_rate'] = risk_free_rate
                    break
                else:
                    print("Invalid response. Please enter a number between 0 and 0.2.")
            
            except Exception as e:
                logger.warning(f"Error asking risk-free rate: {e}")
                print("An error occurred. Please try again.")
        
        self.responses['risk_free_rate'] = self.preferences['risk_free_rate']
    
    async def _ask_data_sources(self):
        """
        Ask user about data sources.
        """
        print("\n--- Data Sources ---")
        print("Used data sources:")
        print("Yahoo Finance")
    
        self.preferences['data_sources'] = ['yahoo']  # Default to Yahoo Finance

        self.responses['data_sources'] = self.preferences['data_sources']
    
    
    async def _ask_forecast_preferences(self):
        """
        Ask user about forecast preferences.
        """
        print("\n--- Forecast Preferences ---")
        
        # Forecast horizon
        print("\nEnter the forecast horizon in days:")
        
        while True:
            try:
                response = input(f"Forecast horizon (default: {self.preferences['forecast_horizon']}): ").strip()
                
                if not response:
                    break
                
                forecast_horizon = int(response)
                
                if 1 <= forecast_horizon <= 365:
                    self.preferences['forecast_horizon'] = forecast_horizon
                    break
                else:
                    print("Invalid response. Please enter a number between 1 and 365.")
            
            except Exception as e:
                logger.warning(f"Error asking forecast horizon: {e}")
                print("An error occurred. Please try again.")
        
        self.responses['forecast_horizon'] = self.preferences['forecast_horizon']
        
        # Forecast models
        print("\nSelect forecast models to use:")
        print("1. LSTM (Long Short-Term Memory)")
        print("2. GRU (Gated Recurrent Unit)")
        print("3. Transformer")
        print("4. XGBoost")
        print("5. All of the above")
        
        while True:
            try:
                response = input("Select models (comma-separated, default: 4): ").strip()
                
                if not response:
                    self.preferences['forecast_models'] = ['xgboost']
                    break
                
                if response == '5':
                    self.preferences['forecast_models'] = ['lstm', 'gru', 'transformer', 'xgboost']
                    break
                
                model_indices = [int(idx.strip()) for idx in response.split(',') if idx.strip()]
                
                if all(1 <= idx <= 4 for idx in model_indices):
                    model_map = {1: 'lstm', 2: 'gru', 3: 'transformer', 4: 'xgboost'}
                    self.preferences['forecast_models'] = [model_map[idx] for idx in model_indices]
                    break
                else:
                    print("Invalid response. Please enter numbers between 1 and 5.")
            
            except Exception as e:
                logger.warning(f"Error asking forecast models: {e}")
                print("An error occurred. Please try again.")
        
        self.responses['forecast_models'] = self.preferences['forecast_models']
        
        # Number of tickers to forecast
        print("\nEnter the number of top tickers to forecast:")
        while True:
            try:
                max_tickers = len(self.preferences.get('selected_tickers', []))
                if max_tickers == 0:
                    print("No tickers selected yet. Cannot set number of forecast tickers.")
                    self.preferences['num_forecast_tickers'] = 0
                    break
                    
                default_num = min(5, max_tickers)
                response = input(f"Number of tickers to forecast (1-{max_tickers}, default: {default_num}): ").strip()
                
                if not response:
                    self.preferences['num_forecast_tickers'] = default_num
                    break
                
                num_tickers = int(response)
                
                if 1 <= num_tickers <= max_tickers:
                    self.preferences['num_forecast_tickers'] = num_tickers
                    break
                else:
                    print(f"Invalid response. Please enter a number between 1 and {max_tickers}.")
            
            except Exception as e:
                logger.warning(f"Error asking number of forecast tickers: {e}")
                print("An error occurred. Please try again.")

        self.responses['num_forecast_tickers'] = self.preferences['num_forecast_tickers']


    
    async def _ask_ticker_selection(self):
        """
        Ask user about ticker selection.
        """
        print("\n--- Ticker Selection (Manual Fallback) ---")
        print("Enter ticker symbols for your portfolio (comma-separated):")
        
        while True:
            try:
                response = input("Tickers: ").strip()
                
                if response:
                    tickers = [ticker.strip().upper() for ticker in response.split(',') if ticker.strip()]
                    
                    if tickers:
                        self.preferences['selected_tickers'] = tickers
                        break
                    else:
                        print("Invalid response. Please enter at least one ticker.")
                else:
                    print("Invalid response. Please enter at least one ticker.")
            
            except Exception as e:
                logger.warning(f"Error asking ticker selection: {e}")
                print("An error occurred. Please try again.")
        
        self.responses['selected_tickers'] = self.preferences['selected_tickers']
        
        # Add preferred assets to selected tickers if not already included
        for asset in self.preferences['preferred_assets']:
            if asset not in self.preferences['selected_tickers']:
                self.preferences['selected_tickers'].append(asset)
    
    def get_risk_profile(self):
        """
        Get user's risk profile.
        
        Returns:
        --------
        str
            Risk profile ('conservative', 'moderate', 'aggressive')
        """
        return self.preferences.get('risk_profile', 'moderate')
    
    def get_investment_horizon(self):
        """
        Get user's investment horizon.
        
        Returns:
        --------
        str
            Investment horizon ('short', 'medium', 'long')
        """
        return self.preferences.get('investment_horizon', 'medium')
    
    def get_preferred_sectors(self):
        """
        Get user's preferred sectors.
        
        Returns:
        --------
        list
            List of preferred sectors
        """
        return self.preferences.get('preferred_sectors', [])
    
    def get_excluded_sectors(self):
        """
        Get user's excluded sectors.
        
        Returns:
        --------
        list
            List of excluded sectors
        """
        return self.preferences.get('excluded_sectors', [])
    
    def get_preferred_assets(self):
        """
        Get user's preferred assets.
        
        Returns:
        --------
        list
            List of preferred assets
        """
        return self.preferences.get('preferred_assets', [])

    def get_benchmarks(self):
        return self.preferences.get('benchmarks', ['^GSPC'])

    def get_excluded_assets(self):
        """
        Get user's excluded assets.
        
        Returns:
        --------
        list
            List of excluded assets
        """
        return self.preferences.get('excluded_assets', [])
    
    def get_benchmark(self):
        """
        Get user's benchmark.
        
        Returns:
        --------
        str
            Benchmark ticker symbol
        """
        return self.preferences.get('benchmark', '^GSPC')
    
    def get_risk_free_rate(self):
        """
        Get user's risk-free rate.
        
        Returns:
        --------
        float
            Risk-free rate
        """
        return self.preferences.get('risk_free_rate', 0.02)
    
    def get_optimization_method(self):
        """
        Get user's optimization method.
        
        Returns:
        --------
        str
            Optimization method
        """
        return self.preferences.get('optimization_method', 'efficient_frontier')
    
    def get_optimization_params(self):
        """
        Get optimization parameters.
        
        Returns:
        --------
        dict
            Dictionary with optimization parameters
        """
        return {
            'method': self.preferences.get('optimization_method', 'efficient_frontier'),
            'risk_free_rate': self.preferences.get('risk_free_rate', 0.02),
            'min_weight': self.preferences.get('min_weight', 0.01),
            'max_weight': self.preferences.get('max_weight', 0.4),
            'target_return': self.preferences.get('target_return'),
            'target_volatility': self.preferences.get('target_volatility'),
            'sector_constraints': self.preferences.get('sector_constraints', {}),
            'asset_constraints': self.preferences.get('asset_constraints', {}),
            'frequency': 252,
            'efficient_frontier_points': 50
        }
    
    def get_data_params(self):
        """
        Get data parameters.
        
        Returns:
        --------
        dict
            Dictionary with data parameters
        """
        # Set period based on investment horizon
        investment_horizon = self.preferences.get('investment_horizon', 'medium')
        
        if investment_horizon == 'short':
            period = '1y'
        elif investment_horizon == 'medium':
            period = '3y'
        else:  # long
            period = '5y'
        
        return {
            'period': self.preferences.get('period', period),
            'interval': self.preferences.get('interval', '1d'),
            'sources': self.preferences.get('data_sources', ['yahoo', 'alphavantage', 'fmp']),
            'benchmark': self.preferences.get('benchmark', '^GSPC'),
            'min_history': 252,
            'max_retries': 3,
            'retry_delay': 1,
            'timeout': 30,
            'use_cache': True,
            'cache_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache'),
            'cache_expiry': 24
        }
    
    def get_api_keys(self):
        """
        Get API keys.
        
        Returns:
        --------
        dict
            Dictionary with API keys
        """
        return self.preferences.get('api_keys', {
            'alphavantage': None,
            'fmp': None,
            'finnhub': None
        })
    
    def get_forecast_params(self):
        """
        Get forecast parameters.
        
        Returns:
        --------
        dict
            Dictionary with forecast parameters
        """
        return {
            'horizon': self.preferences.get('forecast_horizon', 30),
            'models': self.preferences.get('forecast_models', ['lstm', 'gru']),
            'use_sentiment_analysis': False,
            'confidence_interval': 0.95,
            'monte_carlo_simulations': 1000,
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
    
    def get_report_params(self):
        """
        Get report parameters.
        
        Returns:
        --------
        dict
            Dictionary with report parameters
        """
        return {
            'formats': self.preferences.get('report_formats', ['html']),
        }
    
    def get_selected_tickers(self):
        """
        Get selected tickers.
        
        Returns:
        --------
        list
            List of selected tickers
        """
        return self.preferences.get('selected_tickers', [])
    
    def get_forecast_tickers(self):
        """
        Get tickers for forecasting.
        
        Returns:
        --------
        list
            List of tickers for forecasting
        """
        # Use preferred assets for forecasting if available
        preferred_assets = self.preferences.get('preferred_assets', [])
        if preferred_assets:
            return preferred_assets
        
        selected_tickers = self.preferences.get('selected_tickers', [])
        num_to_forecast = self.preferences.get('num_forecast_tickers', 5)
        
        # Ensure we don't try to forecast more tickers than available
        num_to_forecast = min(num_to_forecast, len(selected_tickers))
        
        return selected_tickers[:num_to_forecast]

    
    def get_asset_sectors(self):
        """
        Get asset sectors.
        
        Returns:
        --------
        dict
            Dictionary with asset sectors
        """
        # This is a placeholder method
        # In a real implementation, you would get this information from a data source
        return {}
    
    def save_preferences(self, file_path):
        """
        Save preferences to file.
        
        Parameters:
        -----------
        file_path : str
            Path to save preferences
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.preferences, f, indent=4)
            
            logger.info(f"Preferences saved to {file_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
            return False
    
    def load_preferences(self, file_path):
        """
        Load preferences from file.
        
        Parameters:
        -----------
        file_path : str
            Path to load preferences from
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                self.preferences = json.load(f)
            
            logger.info(f"Preferences loaded from {file_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
            return False
