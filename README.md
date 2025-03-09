# Advanced Options Pricing & Analysis Tool

A comprehensive options pricing and analysis application built with Streamlit, implementing various quantitative finance models for option valuation, Greeks calculation, risk assessment, and strategy analysis.

## Overview

This application provides a robust framework for pricing and analyzing financial options, using three core pricing methodologies:

- Black-Scholes Model
- Cox-Ross-Rubinstein Binomial Model
- Monte Carlo Simulation

The tool extends beyond basic pricing to offer Greeks calculation, option strategy evaluation, volatility surface modeling, VaR (Value at Risk) calculations, and scenario analysis - making it suitable for both educational purposes and professional quantitative research.

## Features

### Option Pricing Models
- **Black-Scholes Model**: Closed-form solution for European options
- **Binomial Model**: Discrete-time model supporting both European and American options
- **Monte Carlo Simulation**: Path-dependent pricing with statistical error estimation

### Greeks and Risk Metrics
- **First-order Greeks**: Delta, Gamma, Theta, Vega
- **Higher-order Greeks**: Vanna, Charm, Volga, Veta, Speed, Zomma, Color, Ultima
- **Strategy Greeks**: Combined risk metrics for option strategies

### Option Strategies
- **Call-based**: Covered Call, Long Call, Bull Call Spread, Bear Call Spread
- **Put-based**: Long Put, Protective Put, Bull Put Spread, Bear Put Spread
- **Combined**: Long Straddle, Short Straddle, Iron Butterfly, Iron Condor

### Advanced Quantitative Tools
- **Implied Volatility Calculator**: Numerical solver using optimization
- **Local Volatility Surface**: Dupire's formula implementation
- **Value at Risk (VaR)**: Monte Carlo simulation approach with Expected Shortfall
- **Risk Scenario Analysis**: Stress testing across price, volatility, and time dimensions

### Strategy Analysis
- Strategy P&L profiles with break-even points
- Performance metrics including max profit/loss, profit probability, risk-reward ratio
- Kelly criterion and Sharpe-like ratio calculations
- Time decay visualization and analysis

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
```bash
pip install streamlit numpy pandas matplotlib seaborn scipy plotly
```

### Running the Application
```bash
streamlit run options_pricing_app.py
```

## Mathematical Models

### Black-Scholes Model
The application implements the classic Black-Scholes formula for European options:

For call options:
```
C = S * N(d1) - K * e^(-rT) * N(d2)
```

For put options:
```
P = K * e^(-rT) * N(-d2) - S * N(-d1)
```

Where:
- d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
- d2 = d1 - σ√T

### Binomial Model
The Cox-Ross-Rubinstein binomial model is implemented with:
- up factor: u = e^(σ√Δt)
- down factor: d = 1/u
- risk-neutral probability: p = (e^(rΔt) - d) / (u - d)

### Monte Carlo Simulation
Stock price paths are simulated using Geometric Brownian Motion:
```
S(t+Δt) = S(t) * exp((r - σ²/2)Δt + σ√Δt * Z)
```
Where Z is a standard normal random variable.

### Greeks Calculation
First and higher-order Greeks are calculated using analytical formulas where possible, with careful handling of edge cases.

## Usage Examples

### Basic Option Pricing
Use the sidebar to select a pricing model and input parameters:
- Current asset price
- Strike price
- Time to maturity (in years)
- Volatility
- Risk-free interest rate

### Strategy Analysis
1. Navigate to the "Strategy Analysis" tab
2. Select a strategy category and specific strategy
3. View the P&L profile and risk metrics
4. Enable "Show Detailed Strategy Performance" for comprehensive metrics

### Advanced Quantitative Analysis
1. Navigate to the "Quant Research" tab
2. Select from tools like Implied Volatility, Local Volatility Surface, VaR, or Risk Scenario Analysis
3. Configure tool-specific parameters
4. View the resulting calculations and visualizations

## Project Structure

The application is organized into several functional components:

1. **Pricing Models**: Implementation of Black-Scholes, Binomial, and Monte Carlo methods
2. **Greeks Calculation**: Functions for first and higher-order Greeks
3. **Strategy Analysis**: Tools for evaluating option strategies
4. **Risk Assessment**: VaR and scenario analysis implementations
5. **UI Components**: Streamlit interface elements and visualization functions

## Author

Created by Navnoor Bawa

## License

MIT License
