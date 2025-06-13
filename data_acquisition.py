#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Acquisition Module
=======================

This module handles data acquisition from various financial data sources.
It prioritizes real-time data and implements robust fallback mechanisms
based on user preferences.

Features:
- Multi-source data acquisition (Yahoo Finance, Alpha Vantage, Financial Modeling Prep, SEC EDGAR)
- Asynchronous data fetching for improved performance
- Robust error handling and fallback mechanisms
- Comprehensive financial data retrieval (prices, returns, fundamentals)
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import aiohttp
import asyncio
import yfinance as yf
from datetime import datetime, timedelta
import json
# import time # Not used directly, random also not used.
# import random # Not used
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAcquisition:
    """
    Handles data acquisition from various financial data sources.
    """
    
    def __init__(self, preferences_manager=None):
        """
        Initialize the data acquisition module.
        
        Parameters:
        -----------
        preferences_manager : UserPreferencesManager, optional
            Manager for user preferences
        """
        self.preferences_manager = preferences_manager
        self.session: Optional[aiohttp.ClientSession] = None
        self.cik_map: Optional[Dict[str, str]] = None # For SEC CIK lookup

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default data parameters if no preferences manager
        if preferences_manager is None:
            self.data_params = {
                'period': '2y',
                'interval': '1d',
                'sources': ['yahoo'], # Default for prices
                'fundamental_sources': ['edgar', 'yahoo'], # Default for fundamentals
                'benchmark': '^GSPC',
                'min_history': 252,
                'max_retries': 3,
                'retry_delay': 1,
                'timeout': 30,
                'use_cache': True,
                'cache_dir': os.path.join(project_root, 'cache'),
                'cache_expiry': 24 
            }
        else:
            self.data_params = preferences_manager.get_data_params()
            # Ensure fundamental_sources is present, provide default if not
            if 'fundamental_sources' not in self.data_params:
                self.data_params['fundamental_sources'] = ['edgar', 'yahoo']
            if 'cache_dir' not in self.data_params: # Ensure cache_dir if get_data_params doesn't provide it
                 self.data_params['cache_dir'] = os.path.join(project_root, 'cache')


        # Create cache directory if it doesn't exist and caching is enabled
        if self.data_params['use_cache'] and not os.path.exists(self.data_params['cache_dir']):
            try:
                os.makedirs(self.data_params['cache_dir'])
            except OSError as e:
                logger.error(f"Error creating cache directory {self.data_params['cache_dir']}: {e}")
                self.data_params['use_cache'] = False # Disable cache if directory creation fails
        
        logger.info("Data acquisition module initialized")
    
    async def _ensure_session(self):
        """
        Ensure that an aiohttp session exists.
        """
        if self.session is None or self.session.closed:
            # Define a common timeout for all requests unless overridden
            timeout = aiohttp.ClientTimeout(total=self.data_params.get('timeout', 30))
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """
        Close the aiohttp session.
        """
        if self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.info("Aiohttp session closed.")
    
    
    def _get_cache_path(self, identifier, data_type, source):
        if not self.data_params.get('use_cache', False):
            return None
        
        cache_dir = self.data_params['cache_dir']
        # Sanitize identifier to be filename-friendly
        safe_identifier = "".join([c if c.isalnum() else "_" for c in identifier])
        return os.path.join(cache_dir, f"{safe_identifier}_{data_type}_{source}.json")
    
    def _is_cache_valid(self, cache_path):
        if not self.data_params.get('use_cache', False) or cache_path is None or not os.path.exists(cache_path):
            return False
        
        cache_expiry = self.data_params.get('cache_expiry', 24)
        if cache_expiry <= 0: # Cache never expires if 0 or negative
            return True
        
        try:
            mtime = os.path.getmtime(cache_path)
            cache_time = datetime.fromtimestamp(mtime)
            expiry_time = cache_time + timedelta(hours=cache_expiry)
            return datetime.now() < expiry_time
        except FileNotFoundError:
            return False
            
    def _save_to_cache(self, data, cache_path):
        if not self.data_params.get('use_cache', False) or cache_path is None:
            return False
        try:
            # Ensure data is JSON serializable (e.g. convert DataFrames)
            if isinstance(data, pd.DataFrame):
                data_to_save = data.to_json(orient='split', date_format='iso')
            elif isinstance(data, pd.Series):
                 data_to_save = data.to_json(orient='split', date_format='iso')
            else: # Assuming dict or other JSON-serializable types
                data_to_save = data

            with open(cache_path, 'w') as f:
                # If already a string (from to_json), don't dump again
                if isinstance(data_to_save, str):
                    f.write(data_to_save)
                else:
                    json.dump(data_to_save, f)
            return True
        except Exception as e:
            logger.warning(f"Error saving to cache {cache_path}: {e}")
            return False

    def _load_from_cache(self, cache_path):
        if not self._is_cache_valid(cache_path):
            return None
        try:
            with open(cache_path, 'r') as f:
                # Try to load as JSON, might need adjustment if DataFrames were saved differently
                loaded_data = json.load(f)
                # If it was a DataFrame saved as json string by _save_to_cache
                if isinstance(loaded_data, str) :
                     try: # for pandas json
                         return pd.read_json(loaded_data, orient='split', convert_dates=True)
                     except ValueError: # if it was just a plain string not from pandas
                         return loaded_data
                # if it was a dict that could represent a DataFrame (e.g. from older cache)
                if isinstance(loaded_data, dict) and 'columns' in loaded_data and 'data' in loaded_data and 'index' in loaded_data:
                    return pd.DataFrame(loaded_data['data'], index=pd.Index(loaded_data['index'], name='Date'), columns=loaded_data['columns'])

                return loaded_data # For non-DataFrame data like CIK map or simple dicts
        except Exception as e:
            logger.warning(f"Error loading from cache {cache_path}: {e}")
            return None

    async def get_historical_data(self, tickers: List[str], period: Optional[str]=None, interval: Optional[str]=None, source: Optional[str]=None) -> Dict[str, Optional[pd.DataFrame]]:
        if not tickers:
            logger.info("No tickers provided for historical data fetching.")
            return {}
            
        logger.info(f"Getting historical data for {len(tickers)} tickers")
        
        period = period or self.data_params['period']
        interval = interval or self.data_params['interval']
        sources = [source] if source is not None else self.data_params['sources']
        
        results: Dict[str, Optional[pd.DataFrame]] = {}
        
        for ticker in tickers:
            if not ticker or not isinstance(ticker, str):
                logger.warning(f"Invalid ticker skipped: {ticker}")
                continue
            
            data_found_for_ticker = False
            for src in sources:
                try:
                    if src == 'yahoo':
                        data = await self._get_historical_data_yahoo(ticker, period, interval)
                    else:
                        logger.warning(f"Unknown data source for historical prices: {src}")
                        continue
                    
                    if data is not None and not data.empty:
                        # Ensure index is datetime
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)
                        results[ticker] = data
                        data_found_for_ticker = True
                        logger.info(f"Successfully fetched historical data for {ticker} from {src}")
                        break 
                except Exception as e:
                    logger.warning(f"Error getting historical data for {ticker} from {src}: {e}")
            
            if not data_found_for_ticker:
                logger.warning(f"No historical data found for {ticker} from any configured source.")
                results[ticker] = None
        
        successful_fetches = sum(1 for data in results.values() if data is not None)
        logger.info(f"Got historical data for {successful_fetches} out of {len(tickers)} tickers")
        
        return results

    async def _get_historical_data_yahoo(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        logger.info(f"Attempting to get historical data for {ticker} from Yahoo Finance (period: {period}, interval: {interval})")
        
        cache_path = self._get_cache_path(ticker, f"historical_{period}_{interval}", "yahoo")
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if isinstance(cached_data, pd.DataFrame): # Make sure it's a DataFrame
                logger.info(f"Loaded historical data for {ticker} from Yahoo Finance cache.")
                # Ensure index is datetime after loading from cache
                if not isinstance(cached_data.index, pd.DatetimeIndex):
                     cached_data.index = pd.to_datetime(cached_data.index)
                return cached_data
            elif cached_data is not None: # Cached but not DataFrame, indicates an issue
                 logger.warning(f"Invalid cache format for {ticker} from Yahoo Finance. Refetching.")
        
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=True, auto_adjust=True) # auto_adjust=True is often preferred
            
            if data is None or data.empty:
                logger.warning(f"No historical data found for {ticker} from Yahoo Finance.")
                return None
            
            # yfinance usually returns index as DatetimeIndex
            if self._save_to_cache(data, cache_path):
                 logger.info(f"Saved historical data for {ticker} from Yahoo Finance to cache.")
            return data
        except Exception as e: # Catch a broader range of yfinance issues
            error_msg = str(e)
            if 'SQLite' in error_msg or 'ImproperlyConfigured' in error_msg:
                logger.error(f"Yahoo Finance download for {ticker} failed due to a configuration or database issue (possibly SQLite driver): {error_msg}. Please check your Python/Conda environment for SQLite3 support.")
            elif "No data found, symbol may be delisted" in error_msg or "Failed to get ticker" in error_msg :
                 logger.warning(f"No data found for {ticker} on Yahoo Finance, it might be delisted or an invalid ticker: {error_msg}")
            else:
                logger.warning(f"Generic error getting historical data for {ticker} from Yahoo Finance: {error_msg}")
            return None


    async def get_prices_dataframe(self, tickers: List[str], period: Optional[str]=None, interval: Optional[str]=None, source: Optional[str]=None) -> Optional[pd.DataFrame]:
        if not tickers:
            logger.info("No tickers provided for prices dataframe.")
            return pd.DataFrame()

        logger.info(f"Getting prices dataframe for {len(tickers)} tickers")
        historical_data_dict = await self.get_historical_data(tickers, period, interval, source)
        
        if not historical_data_dict:
            logger.warning("No historical data available to create prices dataframe.")
            return pd.DataFrame()

        prices_list = []
        valid_tickers_in_order = []

        for ticker in tickers:
            data = historical_data_dict.get(ticker)
            if data is not None and not data.empty:
                price_series = None
                # Check for MultiIndex columns (likely from a cached multi-ticker download)
                if isinstance(data.columns, pd.MultiIndex):
                    # Try to find the 'Close' column for the specific ticker
                    if ('Close', ticker) in data.columns:
                        price_series = data[('Close', ticker)]
                    elif ('Adj Close', ticker) in data.columns:  # Fallback
                        price_series = data[('Adj Close', ticker)]
                # Handle simple columns (from a fresh single-ticker download)
                else:
                    if 'Close' in data.columns:
                        price_series = data['Close']
                    elif 'Adj Close' in data.columns:  # Fallback
                        price_series = data['Adj Close']

                if price_series is not None:
                    prices_list.append(price_series.rename(ticker))
                    valid_tickers_in_order.append(ticker)
                else:
                    logger.warning(f"Could not find a usable price column for ticker {ticker}.")
        
        if not prices_list:
            logger.warning("No price data found for any ticker to create prices dataframe.")
            return pd.DataFrame()

        prices_df = pd.concat(prices_list, axis=1)
        # Reorder columns to match the input, dropping any tickers for which data wasn't found
        prices_df = prices_df.reindex(columns=valid_tickers_in_order) 

        if prices_df.empty:
            logger.warning("Prices dataframe is empty after processing.")
            return pd.DataFrame()
        
        # Drop rows where all tickers are NaN (often at the beginning)
        prices_df.dropna(how='all', inplace=True)
        
        # Forward-fill and then back-fill missing values
        prices_df.ffill(inplace=True)
        prices_df.bfill(inplace=True)
        
        logger.info(f"Successfully created prices dataframe with {len(prices_df.columns)} tickers: {list(prices_df.columns)}")
        return prices_df

    async def get_returns_dataframe(self, tickers: List[str], period: Optional[str]=None, interval: Optional[str]=None, source: Optional[str]=None) -> Optional[pd.DataFrame]:
        if not tickers:
            logger.info("No tickers provided for returns dataframe.")
            return pd.DataFrame()

        logger.info(f"Getting returns dataframe for {len(tickers)} tickers")
        prices_df = await self.get_prices_dataframe(tickers, period, interval, source)
        
        if prices_df is None or prices_df.empty:
            logger.warning("Price data is not available or empty; cannot calculate returns.")
            return pd.DataFrame()
            
        returns_df = prices_df.pct_change().dropna(how='all') 
        
        if returns_df.empty:
            logger.warning("Returns dataframe is empty after pct_change and dropna.")
            return pd.DataFrame()
            
        # The first row of pct_change is always NaN, so drop it.
        returns_df = returns_df.iloc[1:]

        logger.info(f"Successfully created returns dataframe with {len(returns_df.columns)} tickers.")
        return returns_df

    async def get_benchmark_returns(self, benchmarks: Optional[List[str]]=None, period: Optional[str]=None, interval: Optional[str]=None, source: Optional[str]=None) -> Optional[pd.DataFrame]:
        logger.info("Getting benchmark returns")
        tickers = benchmarks or self.data_params.get('benchmarks', ['^GSPC'])
        returns_df = await self.get_returns_dataframe(tickers, period, interval, source)
        if returns_df is None or returns_df.empty:
            logger.warning(f"No benchmark returns available for {tickers}.")
            return None
        # returns_df: columns are tickers
        return returns_df

    # --- SEC EDGAR Integration ---
    async def _load_cik_map(self) -> Dict[str, str]:
        if self.cik_map is not None: # Check if already loaded
             return self.cik_map

        logger.info("Loading SEC Ticker to CIK mapping...")
        cache_path = self._get_cache_path("ALL_TICKERS", "cik_map", "sec")
        
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if isinstance(cached_data, dict):
                self.cik_map = cached_data
                logger.info(f"CIK map loaded from cache. {len(self.cik_map)} entries.")
                return self.cik_map

        self.cik_map = {}
        # SEC's official ticker.txt URL
        url = "https://www.sec.gov/include/ticker.txt"
        try:
            await self._ensure_session()
            async with self.session.get(url) as response:
                response.raise_for_status() # Check for HTTP errors
                text_content = await response.text(encoding='latin-1') # SEC file uses latin-1
                
                for line in text_content.splitlines():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        ticker_symbol, cik_number = parts[0].lower(), parts[1]
                        self.cik_map[ticker_symbol] = cik_number.zfill(10) # CIKs are 10 digits, zero-padded
                
                if self._save_to_cache(self.cik_map, cache_path):
                    logger.info(f"Successfully loaded and cached CIK map for {len(self.cik_map)} tickers.")
                else:
                    logger.warning("Failed to save CIK map to cache.")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error downloading ticker.txt from SEC: {e}")
        except Exception as e:
            logger.error(f"Error processing CIK map: {e}")
        
        if not self.cik_map:
            logger.warning("CIK map is empty after attempting to load.")
        return self.cik_map

    async def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        ticker_lower = ticker.lower()
        cik_map_data = await self._load_cik_map() # Ensures map is loaded
        cik = cik_map_data.get(ticker_lower)
        if not cik:
            logger.warning(f"CIK not found for ticker: {ticker}")
        return cik

    async def _get_financial_data_edgar(self, ticker: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        logger.info(f"Getting financial data for {ticker} from SEC EDGAR")
        cik = await self._get_cik_from_ticker(ticker)
        if not cik:
            return None

        cache_path = self._get_cache_path(ticker, "financial_facts", "edgar")
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if isinstance(cached_data, dict) : # Expecting dict
                logger.info(f"Loaded financial facts for {ticker} (CIK: {cik}) from EDGAR cache.")
                return cached_data

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        extracted_facts_data: Dict[str, List[Dict[str, Any]]] = {}

        try:
            await self._ensure_session()
            # Add a specific user-agent for SEC requests
            headers = {'User-Agent': 'MyCompany/MyAppName myemail@example.com'} # Replace with your info
            logger.debug(f"Requesting EDGAR data from: {url}")

            async with self.session.get(url, headers=headers) as response:
                # SEC API might return 403 if User-Agent is not set or too many requests
                if response.status == 403:
                     logger.error(f"Access denied (403) for EDGAR data for {ticker} (CIK: {cik}). Ensure User-Agent is set and respectful of rate limits.")
                     return None
                response.raise_for_status()
                company_facts_json = await response.json()

            if not company_facts_json or 'facts' not in company_facts_json or 'us-gaap' not in company_facts_json['facts']:
                logger.warning(f"No 'us-gaap' facts found in EDGAR data for {ticker} (CIK: {cik}).")
                return None

            facts_to_extract = { # XBRL Tag: Desired Key Name
                "Revenues": "Revenues",
                "NetIncomeLoss": "NetIncomeLoss",
                "Assets": "Assets",
                "Liabilities": "Liabilities",
                "StockholdersEquity": "StockholdersEquity", # Common stock equity
                "EarningsPerShareBasic": "EPSBasic",
                "CashAndCashEquivalentsAtCarryingValue": "CashAndCashEquivalents"
            }

            us_gaap_facts = company_facts_json['facts']['us-gaap']

            for xbrl_tag, data_key_name in facts_to_extract.items():
                if xbrl_tag in us_gaap_facts and 'units' in us_gaap_facts[xbrl_tag]:
                    # Typically, financial data is in USD. You might need to handle other currencies if relevant.
                    if 'USD' in us_gaap_facts[xbrl_tag]['units']:
                        fact_data_list = us_gaap_facts[xbrl_tag]['units']['USD']
                        # We are interested in annual (10-K) and quarterly (10-Q) filings.
                        # The 'form' field indicates this. 'fy' is fiscal year, 'fp' is fiscal period (e.g., Q1, Q2, Q3, Q4, FY).
                        # 'end' is the date the period ended. 'val' is the value.
                        # Let's simplify and take all reported values for now.
                        # A more advanced parser would filter by form type, fiscal year/period, and pick unique end dates.
                        
                        # Filter for unique end dates, preferring 10-K over 10-Q if same end date.
                        # This is a simplified approach. Real XBRL parsing is more robust.
                        processed_fact_data = []
                        seen_end_dates = {} # To handle duplicates, prefer 10-K

                        for item in sorted(fact_data_list, key=lambda x: (x['end'], x['form'] != '10-K')): # Sort to process 10-K first for a given date
                            end_date = item['end']
                            if end_date not in seen_end_dates or item['form'] == '10-K':
                                processed_fact_data.append({
                                    "end_date": item['end'],
                                    "value": item['val'],
                                    "form_type": item['form'],
                                    "fiscal_year": item.get('fy'),
                                    "fiscal_period": item.get('fp'),
                                    "filed_date": item['filed']
                                })
                                seen_end_dates[end_date] = item['form']
                        
                        extracted_facts_data[data_key_name] = processed_fact_data
                    else:
                        logger.debug(f"No USD units for fact '{xbrl_tag}' for {ticker}. Available units: {list(us_gaap_facts[xbrl_tag]['units'].keys())}")
                else:
                    logger.debug(f"Fact '{xbrl_tag}' not found or has no units for {ticker} in us-gaap data.")
            
            if not extracted_facts_data:
                 logger.warning(f"No specified financial facts extracted from EDGAR for {ticker} (CIK: {cik}).")
                 return None


            if self._save_to_cache(extracted_facts_data, cache_path):
                logger.info(f"Saved financial facts for {ticker} (CIK: {cik}) from EDGAR to cache.")
            return extracted_facts_data

        except aiohttp.ClientError as e:
            logger.warning(f"HTTP error getting EDGAR data for {ticker} (CIK: {cik}): {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error for EDGAR data for {ticker} (CIK: {cik}): {e}")
        except Exception as e:
            logger.warning(f"Generic error getting EDGAR data for {ticker} (CIK: {cik}): {e}")
        return None

    async def get_financial_data(self, ticker: str, source: Optional[str]=None) -> Optional[Dict]: # Return type is now just Dict
        logger.info(f"Getting financial data for {ticker}")
        
        # Use specific sources for financials, defaulting to fundamental_sources
        sources_to_try = [source] if source is not None else self.data_params.get('fundamental_sources', ['edgar'])
        
        for src in sources_to_try:
            data: Optional[Dict] = None # Ensure data is Optional[Dict]
            try:
                if src == 'edgar':
                    data = await self._get_financial_data_edgar(ticker)
                else:
                    logger.warning(f"Unknown data source for financials: {src}")
                    continue
                
                if data is not None and data: # Check if data is not None and not empty
                    logger.info(f"Successfully fetched financial data for {ticker} from {src}.")
                    return data
            except Exception as e:
                logger.warning(f"Error getting financial data for {ticker} from {src}: {e}")
        
        logger.warning(f"No financial data found for {ticker} from any configured source.")
        return None

    # --- Company Info Methods (Unchanged from original, ensure they align with any new caching/error handling) ---
    async def get_company_info(self, ticker: str, source: Optional[str]=None) -> Optional[Dict]:
        logger.info(f"Getting company information for {ticker}")
        sources_to_try = [source] if source is not None else self.data_params.get('sources', ['yahoo'])
        
        for src in sources_to_try:
            info: Optional[Dict] = None
            try:
                if src == 'yahoo':
                    info = await self._get_company_info_yahoo(ticker)
                else:
                    logger.warning(f"Unknown data source for company info: {src}")
                    continue
                
                if info is not None and info:
                    logger.info(f"Successfully fetched company info for {ticker} from {src}.")
                    return info
            except Exception as e:
                logger.warning(f"Error getting company information for {ticker} from {src}: {e}")
        
        logger.warning(f"No company information found for {ticker} from any configured source.")
        return None

    async def _get_company_info_yahoo(self, ticker: str) -> Optional[Dict]:
        logger.info(f"Getting company information for {ticker} from Yahoo Finance")
        cache_path = self._get_cache_path(ticker, "company_info", "yahoo")
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if isinstance(cached_data, dict):
                 logger.info(f"Loaded company info for {ticker} from Yahoo cache.")
                 return cached_data
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            if not info: # yf.Ticker.info can return an empty dict
                logger.warning(f"No company information found for {ticker} from Yahoo Finance (empty info dict).")
                return None
            if self._save_to_cache(info, cache_path):
                 logger.info(f"Saved company info for {ticker} from Yahoo to cache.")
            return info
        except Exception as e:
            error_msg = str(e)
            if 'SQLite' in error_msg or 'ImproperlyConfigured' in error_msg :
                 logger.error(f"Yahoo Finance Ticker.info for {ticker} failed due to configuration/DB issue: {error_msg}")
            else:
                 logger.warning(f"Error getting company information for {ticker} from Yahoo Finance: {error_msg}")
            return None

    # --- Search and Sector Ticker Methods (Largely unchanged, ensure they use new caching/error handling patterns if modified) ---
    async def search_tickers(self, query: str, source: Optional[str]=None) -> List[str]:
        logger.info(f"Searching for tickers with query: {query}")
        sources_to_try = [source] if source is not None else self.data_params.get('sources', ['yahoo'])
        
        for src in sources_to_try:
            tickers_found: Optional[List[str]] = None
            try:
                if src == 'yahoo': tickers_found = await self._search_tickers_yahoo(query)
                else:
                    logger.warning(f"Unknown data source for ticker search: {src}")
                    continue
                
                if tickers_found is not None and len(tickers_found) > 0: # Check for non-empty list
                    logger.info(f"Found {len(tickers_found)} tickers for query '{query}' from {src}.")
                    return tickers_found
            except Exception as e:
                logger.warning(f"Error searching for tickers with query '{query}' from {src}: {e}")
        
        logger.warning(f"No tickers found for query: {query} from any configured source.")
        return []

    async def _search_tickers_yahoo(self, query: str) -> Optional[List[str]]:
        logger.info(f"Searching Yahoo Finance for tickers with query: {query}")
        cache_path = self._get_cache_path(query, "search", "yahoo")
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if isinstance(cached_data, list):
                logger.info(f"Loaded ticker search results for '{query}' from Yahoo cache.")
                return cached_data
        try:
            # yf.Tickers might not be the best for search.
            # A direct query to Yahoo's search/lookup endpoint is usually better for actual search.
            # yfinance doesn't have a direct search function. This is a known limitation.
            # For this example, we'll simulate that if yf.Tickers(query).tickers works, it's a "search".
            # This is not a robust search.
            ticker_objects = yf.Tickers(query).tickers # This will try to get info for 'query' as if it's a list of tickers.
            symbols = list(ticker_objects.keys()) # If 'query' was 'AAPL MSFT', symbols will be ['AAPL', 'MSFT']
                                               # If 'query' was 'Apple', this might fail or return empty.
            if not symbols:
                 logger.warning(f"yf.Tickers('{query}') did not yield symbols for Yahoo Finance search.")
                 return None # No symbols found
            if self._save_to_cache(symbols, cache_path):
                logger.info(f"Cached ticker search results for '{query}' from Yahoo.")
            return symbols
        except Exception as e:
            logger.warning(f"Error using yf.Tickers for search query '{query}' on Yahoo Finance: {e}")
            return None


    async def get_sector_tickers(self, sector: str, source: Optional[str]=None) -> List[str]:
        logger.info(f"Getting tickers for sector: {sector}")
        # yahoo is often good for this. Yahoo less so via direct, simple APIs.
        sources_to_try = [source] if source is not None else self.data_params.get('sources', ['yahoo']) # Prioritize yahoo for sector
        
        for src in sources_to_try:
            tickers_found: Optional[List[str]] = None
            try:
                # Placeholder for Yahoo as they don't have straightforward sector ticker APIs
                if src == 'yahoo': tickers_found = await self._get_sector_tickers_yahoo(sector)
                else:
                    logger.warning(f"Unknown data source for sector tickers: {src}")
                    continue
                
                if tickers_found is not None and len(tickers_found) > 0:
                    logger.info(f"Found {len(tickers_found)} tickers for sector '{sector}' from {src}.")
                    return tickers_found
            except Exception as e:
                logger.warning(f"Error getting tickers for sector '{sector}' from {src}: {e}")
        
        logger.warning(f"No tickers found for sector: {sector} from any configured source.")
        return []

    async def _get_sector_tickers_yahoo(self, sector: str) -> Optional[List[str]]:
        logger.warning(f"Yahoo Finance does not provide a direct API for sector tickers for '{sector}'. This method is a placeholder.")
        # In a real scenario, this might involve scraping screeners.
        return None

