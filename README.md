# Portfolio Scanner Advanced

A comprehensive financial portfolio analysis tool with real-time data acquisition, optimization, forecasting, and visualization capabilities.

## Features

- **User Preference Collection**: Interactive questionnaire to customize analysis based on risk profile, investment horizon, and sector/asset preferences
- **Real-Time Data Acquisition**: Multi-source data retrieval from Yahoo Finance, Alpha Vantage, and Financial Modeling Prep
- **Portfolio Optimization**: Multiple optimization methods including Efficient Frontier, Maximum Sharpe Ratio, Minimum Volatility, and Risk Parity
- **Risk Analysis**: Comprehensive risk metrics including VaR, CVaR, Maximum Drawdown, and Beta
- **PyTorch-Based Forecasting**: Advanced price prediction using LSTM, GRU, and Transformer models
- **Interactive Visualizations**: Rich visualizations for prices, returns, efficient frontier, portfolio weights, and forecasts
- **Detailed Reports**: Customizable HTML and Excel reports with integrated analysis and forecasting results

## Installation

1. Clone the repository or extract the zip file
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script:

```bash
python src/main.py
```

This will start the interactive questionnaire to collect your preferences and then perform the analysis.

### Advanced Usage

You can skip the questionnaire and use saved preferences:

```bash
python src/main.py --skip-questionnaire --preferences-file preferences.json
```

You can also specify a custom output directory:

```bash
python src/main.py --output-dir /path/to/output
```

## Module Structure

- `main.py`: Main entry point and orchestration
- `user_preferences.py`: User preference collection and management
- `data_acquisition.py`: Real-time financial data acquisition
- `portfolio_optimization.py`: Portfolio optimization and efficient frontier calculation
- `stock_forecasting.py`: PyTorch-based price forecasting
- `visualization.py`: Data visualization and chart generation
- `report_generator.py`: HTML and Excel report generation

## API Keys

For optimal performance, it's recommended to use API keys for Alpha Vantage and Financial Modeling Prep. These can be provided during the questionnaire or set as environment variables:

```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
export FMP_API_KEY=your_key_here
```

## Example Output

The tool generates various outputs in the specified output directory:

- HTML reports with interactive visualizations
- Excel reports with detailed analysis
- Visualization files for efficient frontier, portfolio weights, etc.
- Forecast charts for selected assets

## Requirements

See `requirements.txt` for a complete list of dependencies.


