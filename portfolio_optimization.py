#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Portfolio Optimization Module
==========================

This module handles portfolio optimization using various methods including
Mean-Variance Optimization, Efficient Frontier, and Risk Parity.

Features:
- Multiple optimization methods
- Efficient frontier calculation and visualization
- Risk-adjusted return optimization
- Custom constraints support
- Integration with user preferences
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
import pypfopt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt import DiscreteAllocation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Handles portfolio optimization using various methods.
    """
    
    def __init__(self, preferences_manager=None):
        """
        Initialize the portfolio optimizer.
        
        Parameters:
        -----------
        preferences_manager : UserPreferencesManager, optional
            Manager for user preferences
        """
        self.preferences_manager = preferences_manager
        
        # Set default optimization parameters if no preferences manager
        if preferences_manager is None:
            self.optimization_params = {
                'method': 'efficient_frontier',  # 'efficient_frontier', 'risk_parity', 'min_volatility', 'max_sharpe'
                'risk_free_rate': 0.02,  # Annual risk-free rate
                'frequency': 252,  # Daily data frequency
                'target_return': None,  # Target return for efficient frontier
                'target_volatility': None,  # Target volatility for efficient frontier
                'min_weight': 0.01,  # Minimum weight per asset
                'max_weight': 0.4,  # Maximum weight per asset
                'weight_sum': 1.0,  # Sum of weights
                'sector_constraints': {},  # Sector constraints
                'asset_constraints': {},  # Asset-specific constraints
                'efficient_frontier_points': 50,  # Number of points on efficient frontier
                'use_market_caps': False,  # Use market caps for weight bounds
                'use_black_litterman': False,  # Use Black-Litterman model
                'views': {},  # Views for Black-Litterman model
                'confidence': 0.5  # Confidence in views for Black-Litterman model
            }
        else:
            self.optimization_params = preferences_manager.get_optimization_params()
        
        logger.info("Portfolio optimizer initialized")
    
    def optimize_portfolio(self, returns_df, prices_df=None, market_caps=None):
        """
        Optimize portfolio based on returns data.
        
        Parameters:
        -----------
        returns_df : pandas.DataFrame
            DataFrame with asset returns
        prices_df : pandas.DataFrame, optional
            DataFrame with asset prices
        market_caps : dict, optional
            Dictionary with market caps for each asset
            
        Returns:
        --------
        dict
            Dictionary with optimization results
        """
        logger.info("Optimizing portfolio")
        
        # Check if returns data is available
        if returns_df is None or returns_df.empty or len(returns_df.columns) < 2:
            logger.error("Returns data is not available or has fewer than 2 assets for optimization.")
            return None
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(
            prices=prices_df if prices_df is not None else returns_df,
            returns_data=prices_df is None,
            frequency=self.optimization_params['frequency']
        )
        
        S = risk_models.sample_cov(
            prices=prices_df if prices_df is not None else returns_df,
            returns_data=prices_df is None,
            frequency=self.optimization_params['frequency']
        )
        
        # --- NEW: Filter out tickers with NaN returns or zero variance ---
        variances = np.diag(S)
        valid_tickers_mask = (variances > 1e-9) & (~mu.isna())
        valid_tickers = mu.index[valid_tickers_mask].tolist()

        dropped_tickers = [t for t in mu.index if t not in valid_tickers]
        if dropped_tickers:
            logger.warning(f"Dropping tickers with invalid data (NaN or zero variance): {dropped_tickers}")

        # Filter all inputs to only include valid tickers
        mu = mu[valid_tickers]
        S = S.loc[valid_tickers, valid_tickers]
        returns_df = returns_df[valid_tickers]
        if prices_df is not None:
            prices_df = prices_df[valid_tickers]
        
        # Check if there are enough assets left to optimize
        if len(valid_tickers) < 2:
            logger.error("Not enough valid tickers remaining to perform portfolio optimization.")
            return None
        # --- END OF NEW LOGIC ---

        # Create efficient frontier object
        ef = EfficientFrontier(
            expected_returns=mu,
            cov_matrix=S,
            weight_bounds=(self.optimization_params['min_weight'], self.optimization_params['max_weight'])
        )
        
        # Add sector constraints if specified
        sector_constraints = self.optimization_params.get('sector_constraints', {})
        if sector_constraints and hasattr(self.preferences_manager, 'get_asset_sectors'):
            asset_sectors = self.preferences_manager.get_asset_sectors()
            
            for sector, (min_weight, max_weight) in sector_constraints.items():
                # Get assets in sector
                sector_assets = [asset for asset, s in asset_sectors.items() if s == sector]
                
                if sector_assets:
                    # Add constraint
                    ef.add_sector_constraint(sector_assets, min_weight, max_weight)
        
        # Add asset-specific constraints if specified
        asset_constraints = self.optimization_params.get('asset_constraints', {})
        for asset, (min_weight, max_weight) in asset_constraints.items():
            if asset in mu.index:
                ef.add_constraint(lambda w: w[mu.index.get_loc(asset)] >= min_weight)
                ef.add_constraint(lambda w: w[mu.index.get_loc(asset)] <= max_weight)
        
        # Optimize portfolio based on method
        try:
            method = self.optimization_params.get('method', 'max_sharpe')
            if method == 'max_sharpe':
                weights = ef.max_sharpe(risk_free_rate=self.optimization_params['risk_free_rate'])
            elif method == 'min_volatility':
                weights = ef.min_volatility()
            elif method == 'risk_parity':
                # PyPortfolioOpt does not have a direct risk parity in the main class.
                # It requires a different approach or a custom objective.
                # Falling back to a robust method like max_sharpe for now.
                logger.warning("Risk Parity is not directly supported in this workflow, falling back to max_sharpe.")
                weights = ef.max_sharpe(risk_free_rate=self.optimization_params['risk_free_rate'])
            else: # Default to max_sharpe
                weights = ef.max_sharpe(risk_free_rate=self.optimization_params['risk_free_rate'])

            cleaned_weights = ef.clean_weights()
            
            expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
                risk_free_rate=self.optimization_params['risk_free_rate'],
                verbose=False
            )
        except Exception as e:
            logger.error(f"An error occurred during portfolio optimization: {e}")
            return None

        # Create result dictionary
        result = {
            'weights': cleaned_weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_method': method,
            'risk_free_rate': self.optimization_params['risk_free_rate']
        }
        
        # Calculate efficient frontier
        ef_result = self.calculate_efficient_frontier(mu, S)
        if ef_result: result['efficient_frontier'] = ef_result
        
        # Calculate asset-specific metrics
        asset_returns = {}
        asset_volatility = {}
        
        for asset in mu.index:
            asset_returns[asset] = mu[asset]
            asset_volatility[asset] = np.sqrt(S.loc[asset, asset])
        
        result['asset_returns'] = asset_returns
        result['asset_volatility'] = asset_volatility
        
        # Add optimal portfolio to result
        result['optimal_portfolio'] = {
            'expected_return': result['expected_return'],
            'volatility': result['volatility'],
            'sharpe_ratio': result['sharpe_ratio']
        }
        
        logger.info(f"Portfolio optimized using {result['optimization_method']}")
        
        return result
    
    def calculate_efficient_frontier(self, mu, S):
        """
        Calculate efficient frontier.
        """
        logger.info("Calculating efficient frontier")
        num_points = self.optimization_params.get('efficient_frontier_points', 50)
        
        try:
            ef = EfficientFrontier(mu, S)
            risk_range = np.linspace(ef.min_volatility()[1], ef.max_sharpe()[1] * 1.2, num=num_points)
            returns, volatilities, sharpe_ratios = [], [], []

            for vol in risk_range:
                try:
                    ef_point = EfficientFrontier(mu, S)
                    ef_point.efficient_risk(target_volatility=vol)
                    ret, _, sharpe = ef_point.portfolio_performance(verbose=False)
                    returns.append(ret)
                    volatilities.append(vol)
                    sharpe_ratios.append(sharpe)
                except Exception:
                    continue # Skip points that can't be calculated

            if not volatilities: return None
            
            return {'volatility': volatilities, 'returns': returns, 'sharpe_ratios': sharpe_ratios}
            
        except Exception as e:
            logger.error(f"Could not calculate efficient frontier: {e}")
            return None
    
    def allocate_portfolio(self, weights, latest_prices, total_portfolio_value):
        """
        Allocate portfolio to discrete units of assets.
        """
        logger.info("Allocating portfolio")
        
        da = DiscreteAllocation(
            weights=weights,
            latest_prices=latest_prices,
            total_portfolio_value=total_portfolio_value
        )
        
        allocation, leftover = da.greedy_portfolio()
        
        allocation_value = {asset: units * latest_prices.get(asset, 0) for asset, units in allocation.items()}
        
        return {
            'allocation': allocation,
            'allocation_value': allocation_value,
            'leftover': leftover,
            'total_value': total_portfolio_value
        }
    
    def plot_efficient_frontier(self, optimization_result, save_path=None):
        """
        Plot efficient frontier.
        
        Parameters:
        -----------
        optimization_result : dict
            Dictionary with optimization results
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        str or None
            Path to the saved plot if save_path is provided, None otherwise
        """
        logger.info("Plotting efficient frontier")
        
        # Check if efficient frontier data is available
        if 'efficient_frontier' not in optimization_result:
            logger.error("Efficient frontier data not available")
            return None
        
        # Get efficient frontier data
        ef_data = optimization_result['efficient_frontier']
        
        # Create figure
        fig = go.Figure()
        
        # Add efficient frontier
        fig.add_trace(
            go.Scatter(
                x=ef_data['volatility'],
                y=ef_data['returns'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add optimal portfolio
        if 'optimal_portfolio' in optimization_result:
            optimal_portfolio = optimization_result['optimal_portfolio']
            
            fig.add_trace(
                go.Scatter(
                    x=[optimal_portfolio['volatility']],
                    y=[optimal_portfolio['expected_return']],
                    mode='markers',
                    name='Optimal Portfolio',
                    marker=dict(
                        color='green',
                        size=12,
                        symbol='star'
                    )
                )
            )
        
        # Add individual assets
        if 'asset_returns' in optimization_result and 'asset_volatility' in optimization_result:
            asset_returns = optimization_result['asset_returns']
            asset_volatility = optimization_result['asset_volatility']
            
            fig.add_trace(
                go.Scatter(
                    x=list(asset_volatility.values()),
                    y=list(asset_returns.values()),
                    mode='markers+text',
                    name='Individual Assets',
                    text=list(asset_returns.keys()),
                    textposition='top center',
                    marker=dict(
                        color='purple',
                        size=8,
                        symbol='circle'
                    )
                )
            )
        
        # Add capital market line
        if 'risk_free_rate' in optimization_result:
            risk_free_rate = optimization_result['risk_free_rate']
            
            # Get maximum Sharpe ratio point
            max_sharpe_idx = np.argmax(ef_data['sharpe_ratios']) if 'sharpe_ratios' in ef_data else None
            
            if max_sharpe_idx is not None:
                max_sharpe_return = ef_data['returns'][max_sharpe_idx]
                max_sharpe_vol = ef_data['volatility'][max_sharpe_idx]
                
                # Calculate capital market line
                cml_vols = [0, max_sharpe_vol * 1.5]
                cml_returns = [risk_free_rate, risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_vol * cml_vols[1]]
                
                fig.add_trace(
                    go.Scatter(
                        x=cml_vols,
                        y=cml_returns,
                        mode='lines',
                        name='Capital Market Line',
                        line=dict(color='red', width=2, dash='dash')
                    )
                )
        
        # Update layout
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            showlegend=True,
            width=800,
            height=600
        )
        
        # Save plot if save_path is provided
        if save_path is not None:
            fig.write_html(save_path)
            logger.info(f"Efficient frontier plot saved to {save_path}")
            return save_path
        
        # Show plot
        fig.show()
        
        return None
    
    def plot_portfolio_weights(self, weights, save_path=None):
        """
        Plot portfolio weights.
        
        Parameters:
        -----------
        weights : dict
            Dictionary with asset weights
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        str or None
            Path to the saved plot if save_path is provided, None otherwise
        """
        logger.info("Plotting portfolio weights")
        
        # Create figure
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(weights.keys()),
                    values=list(weights.values()),
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hole=0.3
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Composition',
            showlegend=True,
            width=800,
            height=600
        )
        
        # Save plot if save_path is provided
        if save_path is not None:
            fig.write_html(save_path)
            logger.info(f"Portfolio weights plot saved to {save_path}")
            return save_path
        
        # Show plot
        fig.show()
        
        return None


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Get historical data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    data = yf.download(tickers, period='2y')
    
    # Calculate returns
    prices_df = data['Adj Close']
    returns_df = prices_df.pct_change().dropna()
    
    # Create optimizer
    optimizer = PortfolioOptimizer()
    
    # Optimize portfolio
    result = optimizer.optimize_portfolio(returns_df, prices_df)
    
    # Print results
    print("\nOptimization Results:")
    print(f"Method: {result['optimization_method']}")
    print(f"Expected Return: {result['expected_return'] * 100:.2f}%")
    print(f"Volatility: {result['volatility'] * 100:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    
    print("\nPortfolio Weights:")
    for asset, weight in result['weights'].items():
        print(f"{asset}: {weight * 100:.2f}%")
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier(result, 'efficient_frontier.html')
    
    # Plot portfolio weights
    optimizer.plot_portfolio_weights(result['weights'], 'portfolio_weights.html')
