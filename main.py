#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Portfolio Scanner Advanced - Main Module
=======================================

This is the main entry point for the Portfolio Scanner Advanced application.
It orchestrates the entire workflow from data acquisition to report generation.

Features:
- User preference collection through questionnaire
- Real-time financial data acquisition from multiple sources
- Portfolio optimization with efficient frontier visualization
- Advanced risk analysis with multiple metrics
- PyTorch-based stock price forecasting
- Comprehensive report generation in multiple formats
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
import json
import argparse

# Import application modules
from src.user_preferences import UserPreferencesManager
from src.data_acquisition import DataAcquisition
from src.portfolio_optimization import PortfolioOptimizer
from src.stock_forecasting import StockForecaster
from src.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioScannerAdvanced:
    """
    Main class for the Portfolio Scanner Advanced application.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the Portfolio Scanner Advanced application.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory for output files
        """
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize modules
        self.preferences_manager = UserPreferencesManager()
        self.data_acquisition = None
        self.optimizer = None
        self.forecaster = None
        self.report_generator = None
        self.visualization_manager = None
        
        logger.info("Portfolio Scanner Advanced initialized")
    
    async def run(self, skip_questionnaire=False, preferences_file=None):
        """
        Run the Portfolio Scanner Advanced application.
        
        Parameters:
        -----------
        skip_questionnaire : bool, optional
            Whether to skip the questionnaire and use default preferences
        preferences_file : str, optional
            Path to preferences file to load
            
        Returns:
        --------
        dict
            Dictionary with results
        """
        logger.info("Starting Portfolio Scanner Advanced")
        
        try:
            # Step 1: Collect user preferences
            if not skip_questionnaire:
                logger.info("Collecting user preferences")
                await self.preferences_manager.collect_preferences()
            elif preferences_file is not None:
                logger.info(f"Loading preferences from {preferences_file}")
                self.preferences_manager.load_preferences(preferences_file)
            else:
                logger.info("Using default preferences")
            
            # Step 2: Initialize modules with preferences
            self.data_acquisition = DataAcquisition(self.preferences_manager)
            self.optimizer = PortfolioOptimizer(self.preferences_manager)
            self.forecaster = StockForecaster(self.preferences_manager)
            self.report_generator = ReportGenerator(self.preferences_manager, self.output_dir)
            # Step 3: Get tickers from preferences
            tickers = self.preferences_manager.get_selected_tickers()
            logger.info(f"Selected tickers: {tickers}")
            
            # Step 4: Get historical data
            logger.info("Fetching Prices data")
            # Step 5: Get prices dataframe
            prices_df = await self.data_acquisition.get_prices_dataframe(tickers)
            
            # Step 6: Get returns dataframe
            returns_df = await self.data_acquisition.get_returns_dataframe(tickers)
            
            # Step 7: Get benchmark returns
            benchmarks = self.preferences_manager.get_benchmarks()
            logger.info("Fetching Benchmark data")

            benchmark_returns = await self.data_acquisition.get_benchmark_returns(benchmarks)
            
            # Step 8: Optimize portfolio
            logger.info("Optimizing portfolio")
            optimization_result = self.optimizer.optimize_portfolio(returns_df, prices_df)
            
            # **FIX START**: Calculate portfolio returns and add to results
            if optimization_result and 'weights' in optimization_result:
                # Align returns_df columns with weights
                aligned_tickers = list(optimization_result['weights'].keys())
                aligned_returns_df = returns_df[aligned_tickers]
                
                # Calculate the daily returns of the optimized portfolio
                portfolio_daily_returns = aligned_returns_df.dot(pd.Series(optimization_result['weights']))
                optimization_result['portfolio_returns'] = portfolio_daily_returns
            # **FIX END**

            # Add benchmark returns to optimization result
            if benchmark_returns is not None and optimization_result is not None:
                optimization_result['benchmark_returns'] = benchmark_returns
            
            # Step 9: Perform risk analysis
            logger.info("Performing risk analysis")
            risk_analysis = self._perform_risk_analysis(returns_df, benchmark_returns)
            
            # Step 10: Forecast prices
            logger.info("Forecasting prices")
            forecast_tickers = self.preferences_manager.get_forecast_tickers()
            if not forecast_tickers:
                # Use top weighted tickers if no forecast tickers specified
                if optimization_result and 'weights' in optimization_result:
                    weights = optimization_result.get('weights', {})
                    forecast_tickers = sorted(weights.keys(), key=lambda x: weights[x], reverse=True)[:5]
            
            # This is the corrected line: removed 'await'
            forecasting_results = self.forecaster.forecast_multiple(prices_df, forecast_tickers)
            
            # Step 11: Generate reports
            logger.info("Generating reports")
            self.report_generator.add_optimization_results(optimization_result)
            self.report_generator.add_risk_analysis(risk_analysis)
            self.report_generator.add_forecasting_results(forecasting_results)
            
            report_paths = self.report_generator.generate_all_reports()
            
            # Step 12: Create result dictionary
            result = {
                'optimization_result': optimization_result,
                'risk_analysis': risk_analysis,
                'forecasting_results': forecasting_results,
                'report_paths': report_paths
            }
            
            logger.info("Portfolio Scanner Advanced completed successfully")
            
            return result
        
        finally:
            # Close data acquisition session
            if self.data_acquisition is not None:
                await self.data_acquisition.close()
    
    def _perform_risk_analysis(self, returns_df, benchmark_returns=None):
        """
        Perform risk analysis on returns data.
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            DataFrame with asset returns
        benchmark_returns : pandas.Series, optional
            Series with benchmark returns
            
        Returns:
        --------
        dict
            Dictionary with risk analysis results
        """
        logger.info("Performing risk analysis")

        if returns_df is None or returns_df.empty:
            logger.warning("Returns DataFrame is empty. Skipping risk analysis.")
            return {'risk_metrics': {}}
        
        # Initialize risk analysis dictionary
        risk_analysis = {
            'risk_metrics': {}
        }
        
        # Calculate risk metrics for each ticker
        for ticker in returns_df.columns:
            ticker_returns = returns_df[ticker].dropna()
            
            # Skip if not enough data
            if len(ticker_returns) < 30:
                continue
            
            # Calculate VaR and CVaR
            var_95 = np.percentile(ticker_returns, 5)
            cvar_95 = ticker_returns[ticker_returns <= var_95].mean()
            
            # Calculate maximum drawdown
            cum_returns = (1 + ticker_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Calculate volatility
            volatility = ticker_returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate Sharpe ratio
            risk_free_rate = self.preferences_manager.get_risk_free_rate() if self.preferences_manager else 0.02
            mean_return = ticker_returns.mean() * 252  # Annualized
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate beta if benchmark returns available
            beta = None
            if benchmark_returns is not None:
                # Align dates
                aligned_returns = pd.concat([ticker_returns, benchmark_returns], axis=1).dropna()
                if not aligned_returns.empty and aligned_returns.shape[1] == 2:
                    cov = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])[0, 1]
                    var = np.var(aligned_returns.iloc[:, 1])
                    beta = cov / var if var > 0 else None
            
            # Calculate Sortino ratio
            downside_returns = ticker_returns[ticker_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Store risk metrics
            risk_analysis['risk_metrics'][ticker] = {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'beta': beta
            }
        
        # Calculate portfolio risk metrics if optimization result available
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            # This would be calculated based on the optimized portfolio weights
            # but is omitted here for brevity
            pass
        
        return risk_analysis
    
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
        if self.preferences_manager is not None:
            return self.preferences_manager.save_preferences(file_path)
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
        if self.preferences_manager is not None:
            return self.preferences_manager.load_preferences(file_path)
        return False


async def main():
    """
    Main function for command-line execution.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Portfolio Scanner Advanced')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--skip-questionnaire', action='store_true', help='Skip questionnaire')
    parser.add_argument('--preferences-file', type=str, help='Path to preferences file')
    args = parser.parse_args()
    
    # Create and run Portfolio Scanner Advanced
    scanner = PortfolioScannerAdvanced(output_dir=args.output_dir)
    result = await scanner.run(
        skip_questionnaire=args.skip_questionnaire,
        preferences_file=args.preferences_file
    )
    
    # Print report paths
    if result and 'report_paths' in result:
        print("\nGenerated Reports:")
        for report_type, path in result['report_paths'].items():
            print(f"{report_type.upper()}: {path}")
    

    
    return result


if __name__ == "__main__":
    asyncio.run(main())