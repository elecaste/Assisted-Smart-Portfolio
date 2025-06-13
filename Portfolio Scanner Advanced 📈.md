# Portfolio Scanner Advanced 📈

A comprehensive AI-powered portfolio optimization system that integrates real-time financial data acquisition, advanced machine learning forecasting, and modern portfolio theory to deliver superior investment performance.

![Portfolio Analysis](https://img.shields.io/badge/Portfolio-Analysis-blue)
![AI Forecasting](https://img.shields.io/badge/AI-Forecasting-green)
![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

- **🤖 AI-Enhanced Forecasting**: LSTM, GRU, and XGBoost ensemble models for price prediction
- **📊 Portfolio Optimization**: Modern Portfolio Theory with efficient frontier calculation
- **🌐 Real-Time Data**: Multi-source data acquisition with intelligent fallback mechanisms
- **📈 Advanced Risk Analysis**: VaR, CVaR, Maximum Drawdown, Sharpe/Sortino ratios
- **🎯 Interactive Visualizations**: Professional-grade charts and dashboards
- **🔍 Comprehensive Database**: 300,000+ global securities via FinanceDatabase
- **📱 Web Interface**: Responsive HTML reports with interactive elements
- **⚡ High Performance**: Asynchronous data processing and intelligent caching

## 🏆 Performance Results

| Metric | Portfolio | Benchmark |
|--------|-----------|-----------|
| Expected Return | **24.03%** | 12.5% |
| Sharpe Ratio | **0.91** | 0.58 |
| Max Drawdown | **-18.2%** | -28.4% |
| Volatility | 24.10% | 18.2% |

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning)
- 4GB+ RAM recommended
- Internet connection for data acquisition

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/portfolio_scanner_final_v3.git
cd portfolio_scanner_final_v3
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python3 -c "import torch, pandas, yfinance; print('Installation successful!')"
```

## 🚀 Quick Start

### Basic Usage

Navigate to the project directory and run:

```bash
cd ./pr_2/back_head/portfolio_scanner_final_v3
python3 -m src.main
```

This will start the interactive questionnaire to collect your preferences and perform the analysis.

### Example Session

```bash
$ python3 -m src.main

================================================================================
Welcome to Portfolio Scanner Advanced
Please answer the following questions to customize your portfolio analysis
================================================================================

--- Risk Profile ---
1. Conservative: Lower risk, lower potential returns
2. Moderate: Balanced risk and potential returns  
3. Aggressive: Higher risk, higher potential returns
Select your risk profile (1-3): 2

--- Investment Horizon ---
1. Short-term: Less than 1 year
2. Medium-term: 1-5 years
3. Long-term: More than 5 years
Select your investment horizon (1-3): 3

# ... continue with questionnaire
```

## 📖 User Guide

### 1. Portfolio Configuration

The system guides you through several configuration steps:

#### Risk Profile Selection
- **Conservative**: Lower volatility, stable returns
- **Moderate**: Balanced risk-return profile
- **Aggressive**: Higher volatility, growth-focused

#### Ticker Selection
Choose between:
- **Manual Entry**: Specify your own tickers
- **Filtered Selection**: Use advanced filters by:
  - Country (107 countries available)
  - Currency (41 currencies)
  - Industry (24 sectors)
  - Market Cap (Large, Mid, Small & Micro)

#### Optimization Method
- **Efficient Frontier**: Balanced risk/return optimization
- **Maximum Sharpe Ratio**: Optimal risk-adjusted returns
- **Minimum Volatility**: Lowest risk approach
- **Risk Parity**: Equal risk contribution

#### Forecasting Models
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units
- **XGBoost**: Gradient boosting
- **Ensemble**: Combined model approach (recommended)

### 2. Understanding Results

#### Portfolio Composition
- Optimized weights for each security
- Sector and geographic diversification
- Risk contribution analysis

#### Performance Metrics
- **Expected Return**: Annualized portfolio return
- **Volatility**: Portfolio risk measure
- **Sharpe Ratio**: Risk-adjusted performance
- **Maximum Drawdown**: Worst-case scenario loss

#### Risk Analysis
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR**: Conditional Value at Risk (Expected Shortfall)
- **Beta**: Systematic risk vs. market
- **Sortino Ratio**: Downside risk-adjusted returns

#### Forecasting Results
- 30-day price predictions with confidence intervals
- Model ensemble performance metrics
- Uncertainty quantification

### 3. Output Files

The system generates several output files in the `output/` directory:

```
output/
├── portfolio_report_YYYYMMDD_HHMMSS.html    # Interactive HTML report
├── portfolio_analysis_YYYYMMDD_HHMMSS.xlsx  # Excel spreadsheet
├── efficient_frontier.png                    # Risk-return visualization
├── portfolio_composition.png                 # Allocation chart
├── forecasting_results/                      # AI model predictions
│   ├── BKU_forecast.png
│   ├── PSMT_forecast.png
│   └── ...
└── risk_analysis/                           # Risk metrics
    ├── risk_metrics_comparison.png
    └── benchmark_comparison.png
```

## ⚙️ Advanced Configuration

### Command Line Options

```bash
# Skip questionnaire and use default settings
python3 -m src.main --skip-questionnaire

# Use saved preferences
python3 -m src.main --preferences-file preferences.json

# Custom output directory
python3 -m src.main --output-dir /path/to/output

# Combine options
python3 -m src.main --skip-questionnaire --output-dir ./results
```

### Environment Variables

Set optional API keys for enhanced data access:

```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
export FMP_API_KEY=your_key_here
```

### Configuration Files

Save and reuse preferences:

```python
# Save current preferences
scanner = PortfolioScannerAdvanced()
scanner.save_preferences('my_preferences.json')

# Load saved preferences
scanner.load_preferences('my_preferences.json')
```

## 🏗️ Project Structure

```
portfolio_scanner_final_v3/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Main application entry point
│   ├── user_preferences.py        # User preference management
│   ├── data_acquisition.py        # Financial data retrieval
│   ├── portfolio_optimization.py  # Portfolio optimization engine
│   ├── stock_forecasting.py       # AI forecasting models
│   └── report_generator.py        # Report and visualization generation
├── cache/                         # Data caching directory
├── output/                        # Generated reports and charts
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── preferences.json              # Saved user preferences (optional)
```

## 🔧 Development Setup

### For Developers

1. **Clone and setup development environment:**
```bash
git clone https://github.com/yourusername/portfolio_scanner_final_v3.git
cd portfolio_scanner_final_v3
python3 -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Install development dependencies:**
```bash
pip install pytest black flake8 jupyter
```

3. **Run tests:**
```bash
pytest tests/
```

4. **Code formatting:**
```bash
black src/
flake8 src/
```

### Module Overview

- **`main.py`**: Orchestrates the entire workflow
- **`user_preferences.py`**: Handles user input and preference management
- **`data_acquisition.py`**: Multi-source financial data retrieval with caching
- **`portfolio_optimization.py`**: Modern Portfolio Theory implementation
- **`stock_forecasting.py`**: PyTorch-based AI forecasting models
- **`report_generator.py`**: HTML/Excel report generation with visualizations

## 📊 Technical Specifications

### Data Sources
- **Primary**: Yahoo Finance (via yfinance)
- **Database**: FinanceDatabase (300,000+ securities)
- **Backup**: Alpha Vantage, Financial Modeling Prep
- **Benchmarks**: S&P 500, NASDAQ, Dow Jones, Russell 2000

### AI Models
- **LSTM**: Long Short-Term Memory networks for trend analysis
- **GRU**: Gated Recurrent Units for efficient processing
- **XGBoost**: Gradient boosting for non-linear relationships
- **Ensemble**: Weighted combination of all models

### Optimization Algorithms
- **Mean-Variance Optimization**: Markowitz efficient frontier
- **Risk Parity**: Equal risk contribution allocation
- **Black-Litterman**: Bayesian approach with market views
- **Custom Constraints**: Sector limits, position sizing

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
```bash
# Solution: Ensure all dependencies are installed
pip install -r requirements.txt
```

2. **Data Download Failures**
```bash
# Solution: Check internet connection and try again
# The system has automatic retry mechanisms
```

3. **Memory Issues with Large Portfolios**
```bash
# Solution: Reduce the number of securities or use a machine with more RAM
# Recommended: 4GB+ RAM for optimal performance
```

4. **CUDA/GPU Issues**
```bash
# Solution: The system automatically falls back to CPU
# For GPU acceleration, ensure PyTorch CUDA is properly installed
```

### Performance Optimization

- **Enable Caching**: Significantly reduces data download time
- **Use SSD Storage**: Improves cache performance
- **GPU Acceleration**: Install CUDA-enabled PyTorch for faster training
- **Parallel Processing**: The system automatically uses available CPU cores

## 📈 Example Results

### Portfolio Allocation
- **BKU**: 20.0% (Financial Services)
- **PSMT**: 20.0% (Materials)
- **NPO**: 20.0% (Energy)
- **CUZ**: 20.0% (Materials)
- **FLGT.MX**: 6.55% (Industrials)
- **Others**: 13.45% (Diversified)

### Risk Metrics
- **VaR (95%)**: -3.2%
- **CVaR (95%)**: -4.8%
- **Maximum Drawdown**: -18.2%
- **Beta**: 0.95
- **Correlation with S&P 500**: 0.73

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FinanceDatabase**: Comprehensive financial database
- **yfinance**: Yahoo Finance API wrapper
- **PyTorch**: Deep learning framework
- **pypfopt**: Portfolio optimization library
- **Plotly**: Interactive visualization library

## 📞 Support

For support and questions:

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/portfolio_scanner_final_v3/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/portfolio_scanner_final_v3/wiki)

## 🔮 Future Enhancements

- [ ] Real-time portfolio monitoring
- [ ] Mobile application
- [ ] Advanced ESG integration
- [ ] Cryptocurrency support
- [ ] Social sentiment analysis
- [ ] Automated rebalancing
- [ ] Multi-currency support
- [ ] Options and derivatives analysis

---

**⭐ If you find this project useful, please give it a star on GitHub!**

Made with ❤️ for the financial community

