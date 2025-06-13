#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report Generator Module
====================

This module handles the generation of comprehensive reports for portfolio analysis,
in HTML format with integrated forecasting results.

Features:
- Multiple report formats (HTML)
- Customizable report sections
- Integration of visualizations
- Inclusion of forecasting results
- Detailed portfolio analysis
- Efficient frontier visualization
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import textwrap
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import jinja2
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Configure logging
txt = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=txt)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates reports for portfolio analysis with integrated forecasting.
    """

    def __init__(
        self,
        preferences_manager: Optional[Any] = None,
        output_dir: Optional[str] = None
    ):
        # Manage preferences and output
        self.preferences_manager = preferences_manager
        self.report_params = (
            preferences_manager.get_report_params()
            if preferences_manager
            else {
                'formats': ['html'],
                'include_summary': True,
                'include_portfolio_details': True,
                'include_risk_analysis': True,
                'include_optimization_details': True,
                'include_forecasting': True,
                'chart_width': 800,
                'chart_height': 500,
                'language': 'en'
            }
        )
        
        self.chart_width = self.report_params.get('chart_width', 800)
        self.chart_height = self.report_params.get('chart_height', 500)

        # Setup output directory
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path(__file__).parent.parent / 'reports'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data
        self.optimization_results: Optional[Dict[str, Any]] = None
        self.risk_analysis_results: Optional[Dict[str, Any]] = None
        self.forecasting_results: Optional[Dict[str, Any]] = None

        # Templates dir
        self.templates_dir = Path(__file__).parent / 'templates'
        self.templates_dir.mkdir(exist_ok=True)
        self._create_default_html_template()

        logger.info("Report generator initialized")

    def add_optimization_results(self, optimization_results: Dict[str, Any]) -> None:
        """Store optimization results for reporting."""
        self.optimization_results = optimization_results
        logger.info("Optimization results added")

    def add_risk_analysis(self, risk_analysis_results: Dict[str, Any]) -> None:
        """Store risk analysis results for reporting."""
        self.risk_analysis_results = risk_analysis_results
        logger.info("Risk analysis results added")

    def add_forecasting_results(self, forecasting_results: Dict[str, Any]) -> None:
        """Store forecasting results for reporting."""
        self.forecasting_results = forecasting_results
        logger.info("Forecasting results added")

    def generate_all_reports(self) -> Dict[str, Path]:
        """Generate reports based on report_params and return paths."""
        if not (self.optimization_results and self.risk_analysis_results):
            logger.error("Missing optimization or risk analysis data")
            return {}

        report_paths: Dict[str, Path] = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if 'html' in self.report_params.get('formats', []):
            html_path = self.output_dir / f"portfolio_report_{timestamp}.html"
            self.generate_html_report(html_path)
            report_paths['html'] = html_path

        logger.info(f"Generated {len(report_paths)} report(s)")
        return report_paths

    def generate_html_report(self, output_path: Path) -> Path:
        """Render the HTML report and open it in a browser."""
        template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir))
        )
        template = template_env.get_template('report_template.html')

        chart_paths = self._create_report_charts()
        data = self._prepare_template_data(chart_paths)
        html = template.render(**data)

        output_path.write_text(html, encoding='utf-8')
        logger.info(f"HTML report saved to {output_path}")

        try:
            webbrowser.open(f"file://{output_path.resolve()}")
            logger.info("Opened report in browser")
        except Exception:
            logger.exception("Could not open report in browser")

        return output_path

    def _create_report_charts(self) -> Dict[str, Path]:
        """Generate all chart HTML files and return their paths."""
        charts_dir = self.output_dir / 'charts'
        charts_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_paths: Dict[str, Path] = {}

        # Portfolio Composition\  
        try:
            if self.optimization_results and 'weights' in self.optimization_results:
                weights = self.optimization_results['weights']
                fig = go.Figure(
                    data=[go.Pie(
                        labels=list(weights.keys()),
                        values=list(weights.values()),
                        textinfo='label+percent',
                        hole=0.3
                    )]
                )
                fig.update_layout(
                    title_text='Portfolio Composition',
                    width=self.chart_width,
                    height=self.chart_height
                )
                path = charts_dir / f"portfolio_composition_{timestamp}.html"
                pio.write_html(fig, str(path))
                chart_paths['portfolio_composition'] = path
        except Exception:
            logger.exception("Failed to create portfolio composition chart")

        # Risk/Return Scatter
        try:
            fig = go.Figure()
            ef = self.optimization_results.get('efficient_frontier', {}) if self.optimization_results else {}
            if ef and ef.get('volatility'):
                fig.add_trace(go.Scatter(
                    x=ef['volatility'], y=ef['returns'], mode='lines', name='Efficient Frontier'
                ))
            opt = self.optimization_results.get('optimal_portfolio', {}) if self.optimization_results else {}
            if opt:
                fig.add_trace(go.Scatter(
                    x=[opt.get('volatility')], y=[opt.get('expected_return')], mode='markers', name='Optimal'
                ))
            ar = self.optimization_results.get('asset_returns', {}) if self.optimization_results else {}
            av = self.optimization_results.get('asset_volatility', {}) if self.optimization_results else {}
            if ar and av:
                fig.add_trace(go.Scatter(
                    x=list(av.values()), y=list(ar.values()), mode='markers+text', text=list(ar.keys()), textposition='top center', name='Assets'
                ))
            fig.update_layout(title_text='Risk/Return Analysis', xaxis_title='Volatility', yaxis_title='Return', width=self.chart_width, height=self.chart_height)
            path = charts_dir / f"risk_return_scatter_{timestamp}.html"
            pio.write_html(fig, str(path))
            chart_paths['risk_return_scatter'] = path
        except Exception:
            logger.exception("Failed to create risk/return scatter chart")

        # Portfolio vs Benchmark
        try:
            pr = self.optimization_results.get('portfolio_returns') if self.optimization_results else None
            br = self.optimization_results.get('benchmark_returns') if self.optimization_results else None
            if isinstance(pr, pd.Series) and br is not None and not pr.empty:
                if isinstance(br, pd.Series):
                    df = pd.concat([pr.rename('Portfolio'),
                                    br.rename('Benchmark')], axis=1).dropna()
                else:
                    df = pd.concat([pr.rename('Portfolio'), br], axis=1).dropna()
                cum = (1 + df).cumprod()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cum.index, y=cum['Portfolio'], mode='lines', name='Portfolio'
                ))
                for bench in cum.columns:
                    if bench == 'Portfolio':
                        continue
                    fig.add_trace(go.Scatter(
                        x=cum.index, y=cum[bench], mode='lines', name=bench
                    ))
                fig.update_layout(
                    title_text='Portfolio vs Benchmarks (Cumulative)',
                    width=self.chart_width,
                    height=self.chart_height
                )
                path = charts_dir / f"portfolio_performance_{timestamp}.html"
                pio.write_html(fig, str(path))
                chart_paths['portfolio_performance'] = path
        except Exception:
            logger.exception("Failed to create portfolio vs benchmark chart")
        # Risk Metrics
        try:
            rm = self.risk_analysis_results.get('risk_metrics') if self.risk_analysis_results else {}
            if isinstance(rm, dict) and rm:
                metrics: Dict[str, Dict[str, float]] = {}
                for t, m in rm.items():
                    for k, v in m.items():
                        metrics.setdefault(k, {})[t] = v
                fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=list(metrics.keys()), vertical_spacing=0.1)
                for i, (k, vals) in enumerate(metrics.items(), 1):
                    fig.add_trace(go.Bar(x=list(vals.keys()), y=list(vals.values()), name=k), row=i, col=1)
                fig.update_layout(title_text='Risk Metrics Comparison', showlegend=False, width=self.chart_width, height=300*len(metrics))
                path = charts_dir / f"risk_metrics_{timestamp}.html"
                pio.write_html(fig, str(path))
                chart_paths['risk_metrics'] = path
        except Exception:
            logger.exception("Failed to create risk metrics chart")

        # Forecasting Charts: one chart per ticker including all models
        if self.forecasting_results:
            fc_paths = self._write_forecast_charts(charts_dir, timestamp)
            chart_paths.update(fc_paths)

        logger.info(f"Created {len(chart_paths)} charts")
        return chart_paths

    def _write_forecast_charts(self, charts_dir: Path, timestamp: str) -> Dict[str, Path]:
        """Generate per-ticker multi-model forecasting charts, including historical data."""
        paths: Dict[str, Path] = {}

        for ticker, fr in self.forecasting_results.items():
            # parse forecast dates
            f_dates = [datetime.strptime(d, '%Y-%m-%d') for d in fr.get('forecast_dates', [])]

            # extract historical series if available
            hist = fr.get('historical_prices')
            if isinstance(hist, pd.Series) and not hist.empty:
                hist_dates = pd.to_datetime(hist.index).tolist()
                hist_vals  = hist.values
            else:
                hist_dates, hist_vals = [], []

            # gather model names
            model_names = list(fr.get('forecasts', {}).keys())
            models_key = "-".join(model_names)

            fig = go.Figure()

            # plot historical (only if non-empty)
            if len(hist_vals) > 0:
                fig.add_trace(go.Scatter(
                    x=hist_dates,
                    y=hist_vals,
                    mode='lines',
                    name='Historical',
                    line=dict(color='gray', dash='dash')
                ))

            # plot each model’s forecast + CI
            for model_name, mf in fr.get('forecasts', {}).items():
                # plot forecast mean
                fig.add_trace(go.Scatter(
                    x=f_dates,
                    y=mf['mean'],
                    mode='lines',
                    name=f'{model_name} Mean'
                ))

                # plot forecast confidence interval
                lo = np.array(mf.get('lower', []))
                up = np.array(mf.get('upper', []))
                if lo.size and up.size and not np.allclose(lo, up):
                    fig.add_trace(go.Scatter(
                        x=f_dates + f_dates[::-1],
                        y=np.concatenate([up, lo[::-1]]),
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(0,0,0,0)'),
                        showlegend=False,
                        name=f'{model_name} 95% CI'
                    ))

            fig.update_layout(
                title=f"{ticker} Forecast — {models_key}",
                xaxis_title="Date",
                yaxis_title="Price",
                width=self.chart_width,
                height=self.chart_height
            )

            out_path = charts_dir / f"forecast_{ticker}_{models_key}_{timestamp}.html"
            pio.write_html(fig, str(out_path))
            paths[f"forecast_{ticker}_{models_key}"] = out_path

        logger.info(f"Created {len(paths)} forecast chart(s)")
        return paths

    def _prepare_template_data(self, chart_paths: Dict[str, Path]) -> Dict[str, Any]:
        """
        Prepare data for HTML template.
        """
        # basic metadata
        data = {
            'title': 'Portfolio Analysis Report',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {},
            # now compute chart paths relative to the report directory
            'charts': {
                key: str(path.relative_to(self.output_dir))
                for key, path in chart_paths.items()
            },
            'include_forecasting': bool(self.forecasting_results)
        }

        # summary block
        if self.optimization_results:
            opt = self.optimization_results.get('optimal_portfolio', {})
            data['summary'] = {
                'expected_return': opt.get('expected_return', 0) * 100,
                'volatility': opt.get('volatility', 0) * 100,
                'sharpe_ratio': opt.get('sharpe_ratio', 0),
            }

        return data

    def _create_default_html_template(self) -> None:
        """Write a default Jinja2 HTML template if none exists."""
        path = self.templates_dir / 'report_template.html'
        content = textwrap.dedent("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link href="https://fonts.googleapis.com/css?family=Open+Sans:400,600&display=swap" rel="stylesheet">
  <title>{{ title }}</title>
  <style>
    /* Base reset */
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Open Sans', sans-serif; background: #f4f7fa; color: #333; }
    a { color: inherit; text-decoration: none; }

    /* Header */
    header { background: #2c3e50; padding: 20px 0; }
    header .container { display: flex; align-items: center; }
    header h1 { color: white; font-weight: 400; font-size: 2rem; }

    /* Layout */
    .container { max-width: 1400px; margin: auto; padding: 30px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); gap: 30px; }

    /* Card */
    .card { 
      background: white; 
      border-radius: 8px; 
      box-shadow: 0 2px 12px rgba(0,0,0,0.1); 
      overflow: hidden; 
      display: flex; 
      flex-direction: column; 
    }
    .card-header { 
      background: #fafafa; 
      padding: 18px; 
      border-bottom: 1px solid #eee; 
      font-weight: 600; 
      position: relative; 
      font-size: 1.1rem;
    }
    .card-body { padding: 20px; flex: 1; }

    /* Toggle button */
    .toggle-btn {
      position: absolute;
      top: 14px;
      right: 18px;
      background: none;
      border: none;
      font-size: 1.2rem;
      cursor: pointer;
      line-height: 1;
      color: #777;
    }

    /* Iframes */
    .chart-frame { 
      width: 100%; 
      height: 450px; 
      border: 0; 
    }
    @media (max-width: 900px) {
      .chart-frame { height: 350px; }
      .grid { grid-template-columns: 1fr; }  /* stack on mobile */
    }

    /* Section titles */
    .section-title { margin: 50px 0 15px; font-size: 1.3rem; color: #2c3e50; }

    /* Footer */
    footer { text-align: center; padding: 20px 0; color: #777; font-size: 0.9rem; }
  </style>
</head>
<body>

  <header>
    <div class="container">
      <h1>{{ title }}</h1>
    </div>
  </header>

  <main class="container">
    <!-- Report Overview Card -->
    <div class="card" style="grid-column: 1 / -1;">
      <div class="card-header">
        Report Overview
        <button class="toggle-btn" aria-label="Toggle">−</button>
      </div>
      <div class="card-body">
        <div style="display: flex; gap: 50px; flex-wrap: wrap;">
          <div>
            <p><strong>Generated on</strong></p>
            <p>{{ generation_date }}</p>
          </div>
          <div>
            <p><strong>Expected Return</strong></p>
            <p>{{ "%.2f"|format(summary.expected_return) }}%</p>
          </div>
          <div>
            <p><strong>Volatility</strong></p>
            <p>{{ "%.2f"|format(summary.volatility) }}%</p>
          </div>
          <div>
            <p><strong>Sharpe Ratio</strong></p>
            <p>{{ "%.2f"|format(summary.sharpe_ratio) }}</p>
          </div>
        </div>
      </div>
    </div>

    <div class="grid">
      <!-- Portfolio Composition -->
      <div class="card">
        <div class="card-header">
          Portfolio Composition
          <button class="toggle-btn">−</button>
        </div>
        <div class="card-body">
          {% if 'portfolio_composition' in charts %}
            <iframe class="chart-frame" src="{{ charts.portfolio_composition }}"></iframe>
          {% else %}
            <p>No composition data available</p>
          {% endif %}
        </div>
      </div>

      <!-- Risk & Return Analysis -->
      <div class="card">
        <div class="card-header">
          Risk &amp; Return Analysis
          <button class="toggle-btn">−</button>
        </div>
        <div class="card-body">
          {% if 'risk_return_scatter' in charts %}
            <iframe class="chart-frame" src="{{ charts.risk_return_scatter }}"></iframe>
          {% else %}
            <p>No risk/return data available</p>
          {% endif %}
        </div>
      </div>

      <!-- Portfolio vs Benchmark -->
      <div class="card">
        <div class="card-header">
          Portfolio vs Benchmark
          <button class="toggle-btn">−</button>
        </div>
        <div class="card-body">
          {% if 'portfolio_performance' in charts %}
            <iframe class="chart-frame" src="{{ charts.portfolio_performance }}"></iframe>
          {% else %}
            <p>No performance data available</p>
          {% endif %}
        </div>
      </div>

      <!-- Risk Metrics -->
      <div class="card">
        <div class="card-header">
          Risk Metrics
          <button class="toggle-btn">−</button>
        </div>
        <div class="card-body">
          {% if 'risk_metrics' in charts %}
            <iframe class="chart-frame" src="{{ charts.risk_metrics }}"></iframe>
          {% else %}
            <p>No risk metrics available</p>
          {% endif %}
        </div>
      </div>
    </div>

    {% if include_forecasting %}
      <h2 class="section-title">Individual Forecasts</h2>
      <div class="grid">
        {% for name, src in charts.items() if name.startswith('forecast_') %}
        <div class="card">
          <div class="card-header">
            {{ name.replace('forecast_','').replace('_',' – ') }}
            <button class="toggle-btn">−</button>
          </div>
          <div class="card-body">
            <iframe class="chart-frame" src="{{ src }}"></iframe>
          </div>
        </div>
        {% endfor %}
      </div>
    {% endif %}

  </main>

  <footer>
    &copy; {{ generation_date[:4] }} Portfolio Scanner
  </footer>

  <script>
    // Collapse / expand cards
    document.querySelectorAll('.toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const body = btn.closest('.card').querySelector('.card-body');
        if (body.style.display === 'none') {
          body.style.display = '';
          btn.textContent = '−';
        } else {
          body.style.display = 'none';
          btn.textContent = '+';
        }
      });
    });
  </script>
</body>
</html>
""")
        path.write_text(content, encoding='utf-8')
