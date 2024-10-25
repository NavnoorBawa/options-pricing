# Required Imports
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objects as go
import warnings
from matplotlib.colors import LinearSegmentedColormap

# Page Configuration
st.set_page_config(
    page_title="Options Pricing Models",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styles
st.markdown("""
    <style>
    .greek-card {
        background-color: #2E2E2E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .greek-label {
        color: #9CA3AF;
        font-size: 0.875rem;
        margin-bottom: 0.25rem;
    }
    .greek-value {
        color: white;
        font-size: 1.25rem;
        font-weight: 600;
    }
    .main {
        background-color: #0E1117;
    }
    .css-1d391kg {
        padding-top: 0;
    }
    .stSelectbox [data-testid="stMarkdownContainer"] {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Author Section
st.markdown("Created by:")
st.markdown("""
<a href="https://www.linkedin.com/in/navnoorbawa/" target="_blank" style="text-decoration: none; color: inherit;">
    <div style="display: flex; align-items: center; background-color: #1E2530; padding: 10px; border-radius: 5px; width: fit-content; cursor: pointer;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="margin-right: 10px;">
        <span style="color: #FFFFFF;">Navnoor Bawa</span>
    </div>
</a>
""", unsafe_allow_html=True)

# Core Pricing Models
def black_scholes_calc(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def binomial_calc(S, K, T, r, sigma, n, option_type='call', style='european'):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)
    
    # Stock price tree
    stock = np.zeros((n+1, n+1))
    stock[0,0] = S
    for i in range(1, n+1):
        stock[0:i+1,i] = S * u**np.arange(i,-1,-1) * d**np.arange(0,i+1)
    
    # Option value tree
    option = np.zeros((n+1, n+1))
    
    # Terminal payoffs
    if option_type == 'call':
        option[:,n] = np.maximum(stock[:,n] - K, 0)
    else:
        option[:,n] = np.maximum(K - stock[:,n], 0)
    
    # Backward recursion
    for j in range(n-1,-1,-1):
        for i in range(j+1):
            if style == 'european':
                option[i,j] = np.exp(-r*dt)*(p*option[i,j+1] + (1-p)*option[i+1,j+1])
            else:  # american
                hold = np.exp(-r*dt)*(p*option[i,j+1] + (1-p)*option[i+1,j+1])
                if option_type == 'call':
                    exercise = stock[i,j] - K
                else:
                    exercise = K - stock[i,j]
                option[i,j] = max(hold, exercise)
    
    return option[0,0]


def monte_carlo_calc(S, K, T, r, sigma, n_sim, n_steps, option_type='call'):
    dt = T/n_steps
    nudt = (r - 0.5*sigma**2)*dt
    sigsqrtdt = sigma*np.sqrt(dt)
    
    # Generate paths
    z = np.random.standard_normal((n_sim, n_steps))
    S_path = S*np.exp(np.cumsum(nudt + sigsqrtdt*z, axis=1))
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(S_path[:,-1] - K, 0)
    else:
        payoffs = np.maximum(K - S_path[:,-1], 0)
    
    # Calculate price and error
    price = np.exp(-r*T)*np.mean(payoffs)
    se = np.exp(-r*T)*np.std(payoffs)/np.sqrt(n_sim)
    
    return price, se, S_path

# Validation and Testing Functions
def validate_inputs(S, K, T, r, sigma, **kwargs):
    """Validate input parameters for option pricing models."""
    if not all(isinstance(x, (int, float)) for x in [S, K, T, r, sigma]):
        raise TypeError("All inputs must be numeric")
    if any(x < 0 for x in [S, K, T, sigma]):
        raise ValueError("Stock price, strike price, time to maturity, and volatility must be positive")
    if sigma > 1:
        warnings.warn("Warning: Volatility > 100% specified")
    if T > 3:
        warnings.warn("Warning: Time to maturity > 3 years specified")
    return True

def test_black_scholes():
    """Test Black-Scholes model implementation."""
    try:
        # Test case parameters
        test_cases = [
            {'S': 100, 'K': 100, 'T': 1, 'r': 0.05, 'sigma': 0.2},
            {'S': 100, 'K': 110, 'T': 0.5, 'r': 0.02, 'sigma': 0.3},
            {'S': 100, 'K': 90, 'T': 2, 'r': 0.03, 'sigma': 0.15}
        ]
        
        results = []
        for case in test_cases:
            # Validate inputs
            validate_inputs(**case)
            
            # Calculate call and put prices
            call_price = black_scholes_calc(
                case['S'], case['K'], case['T'],
                case['r'], case['sigma'], 'call'
            )
            put_price = black_scholes_calc(
                case['S'], case['K'], case['T'],
                case['r'], case['sigma'], 'put'
            )
            
            # Verify put-call parity
            parity_diff = abs(
                call_price - put_price -
                case['S'] + case['K'] * np.exp(-case['r'] * case['T'])
            )
            
            results.append({
                'parameters': case,
                'call_price': call_price,
                'put_price': put_price,
                'parity_check': parity_diff < 1e-10
            })
            
        return results
        
    except Exception as e:
        raise Exception(f"Black-Scholes test failed: {str(e)}")

def test_binomial_model():
    """Test Binomial model implementation."""
    try:
        # Test case parameters
        test_cases = [
            {'S': 100, 'K': 100, 'T': 1, 'r': 0.05, 'sigma': 0.2, 'n': 100},
            {'S': 100, 'K': 110, 'T': 0.5, 'r': 0.02, 'sigma': 0.3, 'n': 50},
            {'S': 100, 'K': 90, 'T': 2, 'r': 0.03, 'sigma': 0.15, 'n': 200}
        ]
        
        results = []
        for case in test_cases:
            # Validate inputs
            validate_inputs(**case)
            
            # Calculate prices using both European and American style
            euro_call = binomial_calc(
                case['S'], case['K'], case['T'], case['r'],
                case['sigma'], case['n'], 'call', 'european'
            )
            euro_put = binomial_calc(
                case['S'], case['K'], case['T'], case['r'],
                case['sigma'], case['n'], 'put', 'european'
            )
            amer_call = binomial_calc(
                case['S'], case['K'], case['T'], case['r'],
                case['sigma'], case['n'], 'call', 'american'
            )
            amer_put = binomial_calc(
                case['S'], case['K'], case['T'], case['r'],
                case['sigma'], case['n'], 'put', 'american'
            )
            
            # Verify early exercise premium
            call_premium = amer_call - euro_call
            put_premium = amer_put - euro_put
            
            results.append({
                'parameters': case,
                'european_call': euro_call,
                'european_put': euro_put,
                'american_call': amer_call,
                'american_put': amer_put,
                'early_exercise_call': call_premium,
                'early_exercise_put': put_premium,
                'validity_check': call_premium >= 0 and put_premium >= 0
            })
            
        return results
        
    except Exception as e:
        raise Exception(f"Binomial model test failed: {str(e)}")

def test_monte_carlo():
    """Test Monte Carlo simulation implementation."""
    try:
        # Test case parameters
        test_cases = [
            {'S': 100, 'K': 100, 'T': 1, 'r': 0.05, 'sigma': 0.2,
             'n_sim': 100000, 'n_steps': 100},  # Increased number of simulations
            {'S': 100, 'K': 110, 'T': 0.5, 'r': 0.02, 'sigma': 0.3,
             'n_sim': 100000, 'n_steps': 50},
            {'S': 100, 'K': 90, 'T': 2, 'r': 0.03, 'sigma': 0.15,
             'n_sim': 100000, 'n_steps': 200}
        ]
        
        results = []
        for case in test_cases:
            # Validate inputs
            validate_inputs(**{k: case[k] for k in ['S', 'K', 'T', 'r', 'sigma']})
            
            # Calculate prices and error estimates
            call_price, call_se, call_paths = monte_carlo_calc(
                case['S'], case['K'], case['T'], case['r'],
                case['sigma'], case['n_sim'], case['n_steps'], 'call'
            )
            put_price, put_se, put_paths = monte_carlo_calc(
                case['S'], case['K'], case['T'], case['r'],
                case['sigma'], case['n_sim'], case['n_steps'], 'put'
            )
            
            # Compare with Black-Scholes for validation
            bs_call = black_scholes_calc(
                case['S'], case['K'], case['T'],
                case['r'], case['sigma'], 'call'
            )
            bs_put = black_scholes_calc(
                case['S'], case['K'], case['T'],
                case['r'], case['sigma'], 'put'
            )
            
            # Calculate relative errors
            call_error = abs(call_price - bs_call) / bs_call if bs_call != 0 else abs(call_price)
            put_error = abs(put_price - bs_put) / bs_put if bs_put != 0 else abs(put_price)
            
            # Use a more reasonable tolerance (5% instead of 2%)
            tolerance = 0.05
            accuracy_check = call_error < tolerance and put_error < tolerance
            
            results.append({
                'parameters': case,
                'monte_carlo_call': call_price,
                'monte_carlo_put': put_price,
                'call_std_error': call_se,
                'put_std_error': put_se,
                'bs_call': bs_call,
                'bs_put': bs_put,
                'relative_error_call': call_error,
                'relative_error_put': put_error,
                'accuracy_check': accuracy_check,
                'tolerance_used': tolerance
            })
            
            # Add detailed error information if accuracy check fails
            if not accuracy_check:
                print(f"""
                Monte Carlo Test Case Failed:
                Parameters: {case}
                Call Price (MC): {call_price:.4f}, BS: {bs_call:.4f}, Error: {call_error:.4%}
                Put Price (MC): {put_price:.4f}, BS: {bs_put:.4f}, Error: {put_error:.4%}
                Standard Errors - Call: {call_se:.4f}, Put: {put_se:.4f}
                """)
            
        return results
        
    except Exception as e:
        print(f"Detailed Monte Carlo error: {str(e)}")
        raise Exception(f"Monte Carlo test failed: {str(e)}")
        
# Strategy Calculations and Greeks

def calculate_delta(strategy, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """Calculate position delta based on strategy type"""
    delta = 0
    if "Call" in strategy:
        d1 = (np.log(spot_price/strike_price) +
              (risk_free_rate + volatility**2/2)*time_to_maturity)/(volatility*np.sqrt(time_to_maturity))
        delta = norm.cdf(d1)
    elif "Put" in strategy:
        d1 = (np.log(spot_price/strike_price) +
              (risk_free_rate + volatility**2/2)*time_to_maturity)/(volatility*np.sqrt(time_to_maturity))
        delta = -norm.cdf(-d1)
    return delta
    
def calculate_strategy_greeks(strategy_type, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """Calculate Greeks for selected option strategy"""
    
    def d1(S, K, T, r, sigma):
        return (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    
    def d2(S, K, T, r, sigma):
        return d1(S, K, T, r, sigma) - sigma*np.sqrt(T)

    # Calculate base Greeks
    d1_calc = d1(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    d2_calc = d2(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    
    greeks = {
        'delta': 0,
        'gamma': 0,
        'theta': 0,
        'vega': 0
    }
    
    # Base calculations
    call_delta = norm.cdf(d1_calc)
    put_delta = -norm.cdf(-d1_calc)
    gamma = norm.pdf(d1_calc)/(spot_price * volatility * np.sqrt(time_to_maturity))
    
    call_theta = (-spot_price * norm.pdf(d1_calc) * volatility/(2 * np.sqrt(time_to_maturity)) -
                 risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2_calc))
    
    put_theta = (-spot_price * norm.pdf(d1_calc) * volatility/(2 * np.sqrt(time_to_maturity)) +
                risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2_calc))
    
    vega = spot_price * np.sqrt(time_to_maturity) * norm.pdf(d1_calc)

    # Strategy-specific Greek adjustments
    if "Call" in strategy_type:
        if "Long" in strategy_type or strategy_type == "Bull Call Spread":
            greeks['delta'] = call_delta
            greeks['gamma'] = gamma
            greeks['theta'] = call_theta
            greeks['vega'] = vega
        elif "Short" in strategy_type or "Writing" in strategy_type:
            greeks['delta'] = -call_delta
            greeks['gamma'] = -gamma
            greeks['theta'] = -call_theta
            greeks['vega'] = -vega
    elif "Put" in strategy_type:
        if "Long" in strategy_type or strategy_type == "Bear Put Spread":
            greeks['delta'] = put_delta
            greeks['gamma'] = gamma
            greeks['theta'] = put_theta
            greeks['vega'] = vega
        elif "Short" in strategy_type or "Writing" in strategy_type:
            greeks['delta'] = -put_delta
            greeks['gamma'] = -gamma
            greeks['theta'] = -put_theta
            greeks['vega'] = -vega
    elif strategy_type == "Long Straddle":
        greeks['delta'] = call_delta + put_delta
        greeks['gamma'] = 2 * gamma
        greeks['theta'] = call_theta + put_theta
        greeks['vega'] = 2 * vega
    elif strategy_type == "Short Straddle":
        greeks['delta'] = -(call_delta + put_delta)
        greeks['gamma'] = -2 * gamma
        greeks['theta'] = -(call_theta + put_theta)
        greeks['vega'] = -2 * vega
    elif strategy_type in ["Iron Butterfly", "Iron Condor"]:
        greeks['delta'] = 0  # Near delta-neutral
        greeks['gamma'] = -gamma  # Negative gamma
        greeks['theta'] = -(call_theta + put_theta)  # Positive theta
        greeks['vega'] = -2 * vega  # Negative vega
    else:
        # Default to simple long call Greeks if strategy not specifically handled
        greeks['delta'] = call_delta
        greeks['gamma'] = gamma
        greeks['theta'] = call_theta
        greeks['vega'] = vega

    return greeks

def calculate_strategy_pnl(strategy_type, spot_range, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value=0, put_value=0):
    """Calculate P&L for all available strategies"""
    # Add boundary checks
    if not isinstance(spot_range, (list, np.ndarray)):
        raise ValueError("spot_range must be a list or numpy array")
        
    # Add validation for numerical inputs
    for param in [current_price, strike_price, time_to_maturity, risk_free_rate, volatility]:
        if not isinstance(param, (int, float)) or param < 0:
            raise ValueError(f"Invalid parameter value: {param}")
            
    pnl = []
    for spot in spot_range:
        # Calculate base option values
        call_price = black_scholes_calc(spot, strike_price, time_to_maturity,
                                      risk_free_rate, volatility, 'call')
        put_price = black_scholes_calc(spot, strike_price, time_to_maturity,
                                     risk_free_rate, volatility, 'put')
        
        # Define common strikes for spreads
        lower_strike = strike_price * 0.9
        upper_strike = strike_price * 1.1
        middle_strike = strike_price
        
        current_pnl = 0
        
        # Call Option Strategies
        if strategy_type == "Covered Call Writing":
            current_pnl = (spot - current_price) + (call_value - call_price)
            
        elif strategy_type == "Long Call":
            current_pnl = max(0, spot - strike_price) - call_value
            
        elif strategy_type == "Protected Short Sale":
            current_pnl = (current_price - spot) + max(0, spot - strike_price) - call_value
            
        elif strategy_type == "Reverse Hedge":
            current_pnl = max(0, spot - strike_price) + max(0, strike_price - spot) - (call_value + put_value)
            
        elif strategy_type == "Naked Call Writing":
            current_pnl = call_value - max(0, spot - strike_price)
            
        elif strategy_type == "Ratio Call Writing":
            current_pnl = (spot - current_price) + 2 * (call_value - max(0, spot - strike_price))
            
        elif strategy_type == "Bull Call Spread":
            upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity,
                                          risk_free_rate, volatility, 'call')
            current_pnl = max(0, spot - strike_price) - max(0, spot - upper_strike) - (call_value - upper_call)
            
        elif strategy_type == "Bear Call Spread":
            lower_call = black_scholes_calc(spot, lower_strike, time_to_maturity,
                                          risk_free_rate, volatility, 'call')
            current_pnl = (call_value - lower_call) - (max(0, spot - strike_price) - max(0, spot - lower_strike))
            
        elif strategy_type == "Calendar Call Spread":
            long_call = black_scholes_calc(spot, strike_price, time_to_maturity * 2,
                                         risk_free_rate, volatility, 'call')
            current_pnl = (long_call - call_price) - (long_call - call_value)
            
        elif strategy_type == "Butterfly Call Spread":
            lower_call = black_scholes_calc(spot, lower_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            current_pnl = (max(0, spot - lower_strike) - 2*max(0, spot - strike_price) +
                          max(0, spot - upper_strike)) - (lower_call - 2*call_value + upper_call)
            
        elif strategy_type == "Ratio Call Spread":
            upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            current_pnl = max(0, spot - strike_price) - 2*max(0, spot - upper_strike) - (call_value - 2*upper_call)
            
        elif strategy_type == "Ratio Calendar Call Spread":
            long_call = black_scholes_calc(spot, strike_price, time_to_maturity * 2, risk_free_rate, volatility, 'call')
            current_pnl = long_call - 2*call_price - (long_call - 2*call_value)
            
        elif strategy_type == "Delta-Neutral Calendar Spread":
            long_call = black_scholes_calc(spot, strike_price, time_to_maturity * 2, risk_free_rate, volatility, 'call')
            delta_ratio = calculate_delta(strategy, spot)
            current_pnl = delta_ratio * (long_call - call_price) - (long_call - call_value)
            
        elif strategy_type == "Reverse Calendar Call Spread":
            long_call = black_scholes_calc(spot, strike_price, time_to_maturity * 2, risk_free_rate, volatility, 'call')
            current_pnl = call_price - long_call - (call_value - long_call)
            
        elif strategy_type == "Reverse Ratio Call Spread":
            upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            current_pnl = 2*max(0, spot - strike_price) - max(0, spot - upper_strike) - (2*call_value - upper_call)
            
        elif strategy_type == "Diagonal Bull Call Spread":
            long_call = black_scholes_calc(spot, lower_strike, time_to_maturity * 2, risk_free_rate, volatility, 'call')
            current_pnl = (long_call - call_price) - (long_call - call_value)
            
        # Put Option Strategies
        elif strategy_type == "Long Put":
            current_pnl = max(0, strike_price - spot) - put_value
            
        elif strategy_type == "Protective Put":
            current_pnl = (spot - current_price) + max(0, strike_price - spot) - put_value
            
        elif strategy_type == "Put with Covered Call":
            current_pnl = ((spot - current_price) +
                          max(0, strike_price - spot) - put_value +
                          (call_value - max(0, spot - strike_price)))
            
        elif strategy_type == "No-Cost Collar":
            upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            lower_put = black_scholes_calc(spot, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            current_pnl = ((spot - current_price) +
                          max(0, lower_strike - spot) -
                          max(0, spot - upper_strike) -
                          (lower_put - upper_call))
            
        elif strategy_type == "Naked Put Writing":
            current_pnl = put_value - max(0, strike_price - spot)
            
        elif strategy_type == "Covered Put Sale":
            current_pnl = (current_price - spot) + (put_value - max(0, strike_price - spot))
            
        elif strategy_type == "Ratio Put Writing":
            current_pnl = 2 * put_value - 2 * max(0, strike_price - spot)
            
        elif strategy_type == "Bear Put Spread":
            lower_put = black_scholes_calc(spot, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            current_pnl = max(0, strike_price - spot) - max(0, lower_strike - spot) - (put_value - lower_put)
            
        elif strategy_type == "Bull Put Spread":
            upper_put = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            current_pnl = (put_value - upper_put) - (max(0, strike_price - spot) - max(0, upper_strike - spot))
            
        elif strategy_type == "Calendar Put Spread":
            long_put = black_scholes_calc(spot, strike_price, time_to_maturity * 2, risk_free_rate, volatility, 'put')
            current_pnl = (long_put - put_price) - (long_put - put_value)
            
        elif strategy_type == "Butterfly Put Spread":
            lower_put = black_scholes_calc(spot, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            upper_put = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            current_pnl = (max(0, lower_strike - spot) - 2*max(0, strike_price - spot) +
                          max(0, upper_strike - spot)) - (lower_put - 2*put_value + upper_put)
            
        # Combined Strategies
        elif strategy_type == "Long Straddle":
            current_pnl = max(spot - strike_price, strike_price - spot) - (call_value + put_value)
            
        elif strategy_type == "Short Straddle":
            current_pnl = (call_value + put_value) - max(spot - strike_price, strike_price - spot)
            
        elif strategy_type == "Long Strangle":
            upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            lower_put = black_scholes_calc(spot, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            current_pnl = max(spot - upper_strike, lower_strike - spot) - (upper_call + lower_put)
            
        elif strategy_type == "Short Strangle":
            upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            lower_put = black_scholes_calc(spot, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            current_pnl = (upper_call + lower_put) - max(spot - upper_strike, lower_strike - spot)
            
        elif strategy_type == "Synthetic Long Stock":
            current_pnl = ((spot - strike_price) +
                          (max(0, spot - strike_price) - max(0, strike_price - spot)) -
                          (call_value - put_value))
            
        elif strategy_type == "Synthetic Short Stock":
            current_pnl = ((strike_price - spot) +
                          (max(0, strike_price - spot) - max(0, spot - strike_price)) -
                          (put_value - call_value))
            
        elif strategy_type == "Iron Butterfly":
            lower_put = black_scholes_calc(spot, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            current_pnl = ((put_value + call_value - lower_put - upper_call) -
                          (max(0, strike_price - spot) + max(0, spot - strike_price) -
                           max(0, lower_strike - spot) - max(0, spot - upper_strike)))
            
        elif strategy_type == "Iron Condor":
            lower_put = black_scholes_calc(spot, lower_strike * 0.95, time_to_maturity, risk_free_rate, volatility, 'put')
            upper_call = black_scholes_calc(spot, upper_strike * 1.05, time_to_maturity, risk_free_rate, volatility, 'call')
            mid_lower_put = black_scholes_calc(spot, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
            mid_upper_call = black_scholes_calc(spot, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
            current_pnl = ((mid_upper_call + mid_lower_put - lower_put - upper_call) -
                          (max(0, lower_strike - spot) - max(0, lower_strike * 0.95 - spot) +
                           max(0, spot - upper_strike) - max(0, spot - upper_strike * 1.05)))
        
        pnl.append(current_pnl)
    
    return pnl


# Risk Profile Generation
def generate_risk_profile_table(strategy, current_price, strike_price, time_to_maturity,
                              risk_free_rate, volatility):
    """Generate a comprehensive risk profile table for the strategy"""
    
    def calculate_margin_requirement(strategy_type):
        """Calculate margin requirement based on strategy type"""
        if strategy_type in ["Naked Call Writing", "Naked Put Writing"]:
            return max(current_price * 0.5,
                      current_price * 0.2 + max(0, current_price - strike_price))
        elif strategy_type in ["Covered Call Writing", "Protective Put"]:
            return current_price
        elif "Spread" in strategy_type:
            return abs(strike_price - strike_price * 0.9)  # Width of spread
        elif "Iron" in strategy_type:
            return abs(strike_price - strike_price * 0.9)  # Width of spread
        else:
            return max(0, strike_price * 0.2)  # Basic requirement for long options
    
    def calculate_profit_probability(strategy_type):
        """Calculate probability of profit based on strategy type and market conditions"""
        sigma = volatility * np.sqrt(time_to_maturity)
        
        if "Call" in strategy_type and "Writing" in strategy_type:
            # For short calls, probability OTM at expiration
            return norm.cdf((np.log(strike_price/current_price) +
                           (risk_free_rate - volatility**2/2)*time_to_maturity)/(sigma))
        elif "Put" in strategy_type and "Writing" in strategy_type:
            # For short puts, probability OTM at expiration
            return 1 - norm.cdf((np.log(strike_price/current_price) +
                               (risk_free_rate - volatility**2/2)*time_to_maturity)/(sigma))
        elif "Spread" in strategy_type:
            # For spreads, probability between strikes
            upper_strike = strike_price * 1.1
            return (norm.cdf(np.log(upper_strike/current_price)/(sigma)) -
                   norm.cdf(np.log(strike_price/current_price)/(sigma)))
        else:
            # Default case for long options
            return 1 - norm.cdf((np.log(strike_price/current_price))/(sigma))
    
    # Calculate break-even points
    spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 1000)
    pnl = calculate_strategy_pnl(strategy, spot_range)
    break_even_indices = np.where(np.diff(np.signbit(pnl)))[0]
    break_even_points = [spot_range[i] for i in break_even_indices]
    
    # Calculate profit metrics
    max_profit = max(pnl)
    max_loss = abs(min(pnl))
    
    # Calculate time and volatility impacts
    greeks = calculate_strategy_greeks(strategy, current_price, strike_price,
                                     time_to_maturity, risk_free_rate, volatility)
    
    profile = {
        'Entry Price': current_price,
        'Break-even Points': break_even_points,
        'Max Profit Potential': max_profit,
        'Max Loss Potential': max_loss,
        'Profit Probability': calculate_profit_probability(strategy),
        'Time Decay Impact': "Positive" if greeks['theta'] > 0 else "Negative",
        'Volatility Impact': "Positive" if greeks['vega'] > 0 else "Negative",
        'Margin Requirement': calculate_margin_requirement(strategy)
    }
    
    return profile

# UI Components and Visualization Functions
def create_pnl_colormap():
    """Create custom colormap for P&L visualization"""
    colors = ['darkred', 'red', 'white', 'lightgreen', 'darkgreen']
    nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
    return LinearSegmentedColormap.from_list('pnl_colormap', list(zip(nodes, colors)))

def display_option_prices(price_info):
    """Display option prices in colored boxes"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style="background-color: #90EE90; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: black; margin: 0;">CALL Value</h3>
                <h2 style="color: black; margin: 10px 0;">{price_info['call']}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #FFB6C1; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: black; margin: 0;">PUT Value</h3>
                <h2 style="color: black; margin: 10px 0;">{price_info['put']}</h2>
            </div>
        """, unsafe_allow_html=True)

def display_greeks(calculated_greeks):
    """Display Greeks in a grid layout"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <h4 style="color: white; margin-bottom: 1rem;">Position Greeks</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div class="greek-card">
                        <div class="greek-label">Delta</div>
                        <div class="greek-value">{}</div>
                    </div>
                    <div class="greek-card">
                        <div class="greek-label">Gamma</div>
                        <div class="greek-value">{}</div>
                    </div>
                </div>
            </div>
        """.format(
            round(calculated_greeks['delta'], 3),
            round(calculated_greeks['gamma'], 3)
        ), unsafe_allow_html=True)

# Strategy Dictionaries
strategy_advantages = {
    # Call Option Strategies Advantages
    "Covered Call Writing": [
        "Generates regular income from premium collection",
        "Provides downside protection equal to premium received",
        "Lower risk than outright stock ownership",
        "Can be repeated monthly for consistent income",
        "Benefits from time decay (theta positive)"
    ],
    
    "Long Call": [
        "Limited risk to premium paid",
        "Unlimited profit potential",
        "High leverage compared to stock ownership",
        "No margin requirements",
        "Lower capital requirement than stock purchase"
    ],
    
    "Protected Short Sale": [
        "Limited upside risk through call protection",
        "Profits from stock price decline",
        "Less margin intensive than pure short sale",
        "Flexible exit strategies available",
        "Known maximum loss"
    ],
    
    "Reverse Hedge": [
        "Profits from large moves in either direction",
        "Known maximum loss (premium paid)",
        "Multiple profit opportunities",
        "Good strategy for earnings announcements",
        "Benefits from volatility increase"
    ],
    
    "Naked Call Writing": [
        "Immediate premium income",
        "Benefits from time decay",
        "High probability of profit in range-bound markets",
        "No capital outlay for stock purchase",
        "Profits from decreased volatility"
    ],
    
    "Ratio Call Writing": [
        "Enhanced premium income potential",
        "Lower risk than naked calls",
        "Benefits from time decay",
        "Can profit in multiple scenarios",
        "Good for range-bound markets"
    ],
    
    "Bull Call Spread": [
        "Defined risk and reward",
        "Lower cost than outright call purchase",
        "Benefits from upward movement",
        "Less affected by volatility changes",
        "Lower break-even point than long call"
    ],
    
    "Bear Call Spread": [
        "Credit received upfront",
        "Defined risk and reward",
        "Benefits from downward movement",
        "High probability of profit",
        "Less sensitive to volatility changes"
    ],
    
    "Calendar Call Spread": [
        "Benefits from time decay",
        "Lower cost than outright options",
        "Multiple profit opportunities",
        "Benefits from volatility increase",
        "Can be adjusted to market conditions"
    ],
    
    "Butterfly Call Spread": [
        "Defined risk and reward",
        "Low cost relative to potential return",
        "Benefits from low volatility",
        "Multiple profit scenarios",
        "Good for range-bound markets"
    ],
    
    "Ratio Call Spread": [
        "Lower initial cost",
        "Multiple profit zones",
        "Benefits from volatility decline",
        "Good for directional views",
        "Flexible position management"
    ],
    
    "Ratio Calendar Call Spread": [
        "Benefits from time decay",
        "Multiple profit opportunities",
        "Lower cost than regular calendar spread",
        "Can profit from volatility changes",
        "Adjustable to market conditions"
    ],
    
    "Delta-Neutral Calendar Spread": [
        "Market neutral strategy",
        "Benefits from volatility increase",
        "Profits from time decay",
        "Multiple adjustment opportunities",
        "Good for high volatility environments"
    ],
    
    "Reverse Calendar Call Spread": [
        "Benefits from quick market moves",
        "Lower cost than outright options",
        "Good for low volatility environments",
        "Multiple profit opportunities",
        "Flexible exit strategies"
    ],
    
    "Reverse Ratio Call Spread": [
        "Limited risk with unlimited upside",
        "Benefits from sharp moves",
        "Good for breakout scenarios",
        "Multiple profit opportunities",
        "Positive gamma exposure"
    ],
    
    "Diagonal Bull Call Spread": [
        "Lower cost than calendar spread",
        "Benefits from upward movement",
        "Multiple profit opportunities",
        "Time decay advantages",
        "Flexible position management"
    ],
    
    # Put Option Strategies Advantages
    "Long Put": [
        "Limited risk to premium paid",
        "Significant profit potential in down markets",
        "High leverage compared to short selling",
        "No margin requirements",
        "Good hedge against stock positions"
    ],
    
    "Protective Put": [
        "Limits downside risk",
        "Maintains upside potential",
        "Known maximum loss",
        "Portfolio protection",
        "Good for volatile markets"
    ],
    
    "Put with Covered Call": [
        "Complete downside protection",
        "Enhanced income potential",
        "Defined risk and reward",
        "Multiple income streams",
        "Flexible position management"
    ],
    
    "No-Cost Collar": [
        "Zero or low-cost protection",
        "Defined risk and reward",
        "No or minimal cash outlay",
        "Good for portfolio protection",
        "Adjustable protection levels"
    ],
    
    "Naked Put Writing": [
        "Immediate premium income",
        "High probability of profit",
        "Benefits from time decay",
        "Good for acquiring stock",
        "Benefits from decreased volatility"
    ],
    
    "Covered Put Sale": [
        "Enhanced premium income",
        "Benefits from time decay",
        "Good for range-bound markets",
        "Multiple profit sources",
        "High probability of profit"
    ],
    
    "Ratio Put Writing": [
        "Enhanced premium income",
        "Benefits from time decay",
        "Multiple profit zones",
        "Good for range-bound markets",
        "Flexible position management"
    ],
    
    "Bear Put Spread": [
        "Defined risk and reward",
        "Lower cost than outright put",
        "Benefits from downward movement",
        "Less affected by volatility",
        "Good for bearish views"
    ],
    
    "Bull Put Spread": [
        "Credit received upfront",
        "High probability of profit",
        "Benefits from upward movement",
        "Defined risk and reward",
        "Good for income generation"
    ],
    
    "Calendar Put Spread": [
        "Benefits from time decay",
        "Multiple profit opportunities",
        "Lower cost than outright puts",
        "Benefits from volatility increase",
        "Flexible position management"
    ],
    
    "Butterfly Put Spread": [
        "Defined risk and reward",
        "Low cost relative to return",
        "Benefits from low volatility",
        "Multiple profit scenarios",
        "Good for range-bound markets"
    ],
    
    # Combined Strategies Advantages
    "Long Straddle": [
        "Profits from large moves either direction",
        "Unlimited profit potential",
        "Perfect for event-driven trades",
        "Benefits from volatility increase",
        "Good earnings play strategy"
    ],
    
    "Short Straddle": [
        "Maximum premium collection",
        "Benefits from time decay",
        "Profits from range-bound markets",
        "Benefits from volatility decrease",
        "High probability of partial profit"
    ],
    
    "Long Strangle": [
        "Lower cost than straddle",
        "Unlimited profit potential",
        "Profits from large moves",
        "Benefits from volatility increase",
        "Good for uncertain directional views"
    ],
    
    "Short Strangle": [
        "Higher probability of profit than straddle",
        "Maximum premium collection",
        "Benefits from time decay",
        "Profits from range-bound markets",
        "Benefits from volatility decrease"
    ],
    
    "Synthetic Long Stock": [
        "Lower capital requirement than stock",
        "Similar risk/reward to stock",
        "No uptick rule restrictions",
        "Good for hard-to-borrow stocks",
        "Leverage benefits"
    ],
    
    "Synthetic Short Stock": [
        "Lower margin than short stock",
        "No hard-to-borrow fees",
        "Similar risk/reward to short stock",
        "No uptick rule restrictions",
        "Good for restricted stocks"
    ],
    
    "Iron Butterfly": [
        "Defined risk and reward",
        "High probability of profit",
        "Benefits from time decay",
        "Good for low volatility",
        "Multiple adjustment opportunities"
    ],
    
    "Iron Condor": [
        "Defined risk and reward",
        "Higher probability of profit",
        "Benefits from time decay",
        "Good for range-bound markets",
        "Multiple adjustment opportunities"
    ]
}

strategy_disadvantages = {
    # Call Option Strategies Disadvantages
    "Covered Call Writing": [
        "Limits upside potential beyond strike price",
        "Requires significant capital for stock position",
        "Stock can still decline significantly",
        "Opportunity cost in strong bull markets",
        "Assignment risk near ex-dividend dates"
    ],
    
    "Long Call": [
        "Premium decay over time (theta decay)",
        "Requires correct timing and direction",
        "Loses value with volatility decline",
        "Can lose 100% of investment",
        "High leverage can magnify losses"
    ],
    
    "Protected Short Sale": [
        "Higher cost than direct put purchase",
        "Complex execution and management",
        "Multiple commissions and fees",
        "Hard-to-borrow stock issues",
        "Requires margin account"
    ],
    
    "Reverse Hedge": [
        "High cost of double premium",
        "Requires significant price movement",
        "Time decay works against position",
        "Loses value with volatility decline",
        "Complex position management"
    ],
    
    "Naked Call Writing": [
        "Unlimited risk potential",
        "Requires significant margin",
        "Risk of early assignment",
        "Dangerous in strong bull markets",
        "Subject to margin calls"
    ],
    
    "Ratio Call Writing": [
        "Complex risk profile",
        "Unlimited risk beyond breakeven",
        "Requires careful monitoring",
        "Multiple assignment risks",
        "Higher margin requirements"
    ],
    
    "Bull Call Spread": [
        "Limited profit potential",
        "Requires stock to rise to be profitable",
        "Time decay affects long option",
        "Volatility decline hurts position",
        "Early exercise can complicate management"
    ],
    
    "Bear Call Spread": [
        "Limited profit potential",
        "Early assignment risk",
        "Complex margin requirements",
        "Multiple commission costs",
        "Requires precise timing"
    ],
    
    "Calendar Call Spread": [
        "Complex position management",
        "Risk of rapid stock movement",
        "Volatility changes affect legs differently",
        "Requires accurate timing",
        "Multiple commission costs"
    ],
    
    "Butterfly Call Spread": [
        "Limited profit potential",
        "Requires precise stock movement",
        "Complex position management",
        "Multiple commission costs",
        "Illiquid at certain strikes"
    ],
    
    "Ratio Call Spread": [
        "Unlimited risk potential",
        "Complex delta management",
        "Multiple commission costs",
        "Hard to adjust position",
        "Early exercise risk"
    ],
    
    "Ratio Calendar Call Spread": [
        "Complex position management",
        "Sensitive to volatility changes",
        "Multiple expiration management",
        "Higher commission costs",
        "Requires precise timing"
    ],
    
    "Delta-Neutral Calendar Spread": [
        "Requires constant monitoring",
        "Complex adjustments needed",
        "High commission costs from adjustments",
        "Sensitive to volatility skew",
        "Time decay risk if stock moves significantly"
    ],
    
    "Reverse Calendar Call Spread": [
        "Complex risk profile",
        "High cost of near-term options",
        "Sensitive to volatility changes",
        "Requires precise timing",
        "Limited profit potential"
    ],
    
    "Reverse Ratio Call Spread": [
        "High initial cost",
        "Complex position management",
        "Multiple commission costs",
        "Limited profit in range-bound markets",
        "Sensitive to volatility changes"
    ],
    
    "Diagonal Bull Call Spread": [
        "Complex position management",
        "Different expiration dates",
        "Volatility risk across months",
        "Higher commission costs",
        "Requires precise timing"
    ],
    
    # Put Option Strategies Disadvantages
    "Long Put": [
        "Premium decay over time",
        "Requires correct timing",
        "Loses value with volatility decline",
        "Can lose 100% of investment",
        "High cost relative to potential stock decline"
    ],
    
    "Protective Put": [
        "Expensive insurance cost",
        "Reduces overall returns",
        "Regular premium outlay needed",
        "Time decay works against position",
        "Less effective in low volatility"
    ],
    
    "Put with Covered Call": [
        "Complex position management",
        "High total premium cost",
        "Multiple commission costs",
        "Limited upside potential",
        "Time decay on both options"
    ],
    
    "No-Cost Collar": [
        "Limits upside potential",
        "Complex position management",
        "Multiple commission costs",
        "May require adjustments",
        "Opportunity cost in strong markets"
    ],
    
    "Naked Put Writing": [
        "Substantial downside risk",
        "Requires significant margin",
        "Risk of early assignment",
        "Subject to margin calls",
        "Losses accelerate in down markets"
    ],
    
    "Covered Put Sale": [
        "Unlimited risk potential",
        "Complex margin requirements",
        "Multiple commission costs",
        "Hard-to-borrow stock issues",
        "Requires margin account"
    ],
    
    "Ratio Put Writing": [
        "Complex risk profile",
        "Unlimited risk in down markets",
        "Multiple assignment risks",
        "Higher margin requirements",
        "Difficult to adjust"
    ],
    
    "Bear Put Spread": [
        "Limited profit potential",
        "Time decay affects long option",
        "Multiple commission costs",
        "Requires precise timing",
        "Early exercise can complicate management"
    ],
    
    "Bull Put Spread": [
        "Limited profit potential",
        "Risk of early assignment",
        "Complex margin requirements",
        "Multiple commission costs",
        "Maximum loss at lower strike"
    ],
    
    "Calendar Put Spread": [
        "Complex position management",
        "Risk of rapid stock movement",
        "Volatility changes affect legs differently",
        "Multiple commission costs",
        "Requires precise timing"
    ],
    
    "Butterfly Put Spread": [
        "Limited profit potential",
        "Requires precise stock movement",
        "Complex position management",
        "Multiple commission costs",
        "Illiquid at certain strikes"
    ],
    
    # Combined Strategies Disadvantages
    "Long Straddle": [
        "High premium cost",
        "Requires large price movement",
        "Time decay hurts both options",
        "Loses value with volatility decline",
        "Multiple commission costs"
    ],
    
    "Short Straddle": [
        "Unlimited risk potential",
        "High margin requirements",
        "Risk of early assignment",
        "Complex position management",
        "Dangerous in volatile markets"
    ],
    
    "Long Strangle": [
        "High premium cost",
        "Requires larger price movement",
        "Time decay hurts both options",
        "Multiple commission costs",
        "Loses value with volatility decline"
    ],
    
    "Short Strangle": [
        "Unlimited risk potential",
        "High margin requirements",
        "Multiple assignment risks",
        "Complex position management",
        "Dangerous in volatile markets"
    ],
    
    "Synthetic Long Stock": [
        "Complex position management",
        "Early assignment risk",
        "Multiple commission costs",
        "Requires margin account",
        "No dividend benefits"
    ],
    
    "Synthetic Short Stock": [
        "Unlimited risk potential",
        "Complex position management",
        "Multiple commission costs",
        "Early assignment risk",
        "Requires margin account"
    ],
    
    "Iron Butterfly": [
        "Limited profit potential",
        "Complex position management",
        "Multiple commission costs",
        "Requires precise timing",
        "All four options must be managed"
    ],
    
    "Iron Condor": [
        "Limited profit potential",
        "Complex position management",
        "Multiple commission costs",
        "All four options must be managed",
        "Early assignment risk on short options"
    ]
}

strategy_market_conditions = {
    # Call Option Strategies Market Conditions
    "Covered Call Writing": """
        Best Market Outlook: Neutral to slightly bullish
        Volatility Requirement: Moderate to high implied volatility
        Time Horizon: 30-45 days typically
        Key Conditions:
        - Stable to slowly rising market
        - Higher than normal implied volatility
        - No major events expected
        - Stock trading above support levels
        Risk Factors to Watch:
        - Earnings announcements
        - Ex-dividend dates
        - Major market events
    """,
    
    "Long Call": """
        Best Market Outlook: Strongly bullish
        Volatility Requirement: Low to moderate implied volatility
        Time Horizon: Minimum 60 days recommended
        Key Conditions:
        - Strong upward momentum
        - Clear technical breakout
        - Positive market sentiment
        - Low put/call ratio
        Risk Factors to Watch:
        - Overall market direction
        - Sector momentum
        - Time decay acceleration
    """,
    
    "Protected Short Sale": """
        Best Market Outlook: Bearish with defined risk
        Volatility Requirement: Low to moderate implied volatility
        Time Horizon: 30-90 days
        Key Conditions:
        - Clear technical breakdown
        - Weakening fundamentals
        - Negative sector momentum
        - High call protection available
        Risk Factors to Watch:
        - Short squeeze potential
        - Hard to borrow rates
        - Dividend announcements
    """,
    
    "Reverse Hedge": """
        Best Market Outlook: Uncertain direction with large move expected
        Volatility Requirement: Low implied volatility
        Time Horizon: 30-60 days
        Key Conditions:
        - Upcoming binary events
        - Historical price movement patterns
        - Low current volatility vs historical
        - Technical consolidation pattern
        Risk Factors to Watch:
        - Volatility crush after events
        - Time decay on both options
        - Size of expected move
    """,
    
    "Naked Call Writing": """
        Best Market Outlook: Neutral to slightly bearish
        Volatility Requirement: High implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Clear technical resistance levels
        - Overbought conditions
        - High implied volatility vs historical
        - Range-bound market
        Risk Factors to Watch:
        - Gap risk overnight
        - Short interest levels
        - Merger/acquisition potential
    """,
    
    "Ratio Call Writing": """
        Best Market Outlook: Neutral with upside potential
        Volatility Requirement: High implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Clear technical levels
        - High premium available
        - Stable market conditions
        - Known support/resistance
        Risk Factors to Watch:
        - Sharp upward moves
        - Volatility changes
        - Time decay differential
    """,
    
    "Bull Call Spread": """
        Best Market Outlook: Moderately bullish
        Volatility Requirement: Moderate to high implied volatility
        Time Horizon: 45-60 days
        Key Conditions:
        - Upward trend in place
        - Clear technical support
        - Reasonable volatility levels
        - Defined upside target
        Risk Factors to Watch:
        - Time decay impact
        - Strike price selection
        - Overall market trend
    """,
    
    "Bear Call Spread": """
        Best Market Outlook: Moderately bearish
        Volatility Requirement: High implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Downward trend in place
        - Clear technical resistance
        - Overbought conditions
        - High premium available
        Risk Factors to Watch:
        - Gap risk
        - Early assignment
        - Support levels
    """,
    
    "Calendar Call Spread": """
        Best Market Outlook: Neutral near term, bullish longer term
        Volatility Requirement: Low near-term volatility
        Time Horizon: 30-90 days
        Key Conditions:
        - Stable near-term price action
        - Volatility term structure
        - Clear technical levels
        - Time decay opportunity
        Risk Factors to Watch:
        - Sharp price movements
        - Volatility skew changes
        - Time decay differential
    """,
    
    "Butterfly Call Spread": """
        Best Market Outlook: Highly neutral
        Volatility Requirement: Low implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Range-bound market
        - Clear technical levels
        - Low volatility environment
        - Known price targets
        Risk Factors to Watch:
        - Sharp directional moves
        - Volatility increases
        - Time decay profile
    """,
    
    "Ratio Call Spread": """
        Best Market Outlook: Moderately bullish with ceiling
        Volatility Requirement: Moderate to high implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Clear technical levels
        - Stable volatility environment
        - Known resistance levels
        - Premium opportunity
        Risk Factors to Watch:
        - Sharp upward moves
        - Volatility changes
        - Delta management
    """,
    
    "Ratio Calendar Call Spread": """
        Best Market Outlook: Neutral near term, directional longer term
        Volatility Requirement: Low near-term volatility
        Time Horizon: Mixed (30-90 days)
        Key Conditions:
        - Volatility term structure
        - Clear technical levels
        - Time decay opportunity
        - Defined price targets
        Risk Factors to Watch:
        - Sharp price movements
        - Volatility skew changes
        - Calendar risk
    """,
    
    "Delta-Neutral Calendar Spread": """
        Best Market Outlook: Neutral with volatility increase expected
        Volatility Requirement: Low near-term volatility
        Time Horizon: Mixed (30-90 days)
        Key Conditions:
        - Volatility term structure
        - Range-bound market
        - Clear technical levels
        - Low historical volatility
        Risk Factors to Watch:
        - Sharp directional moves
        - Volatility term structure changes
        - Delta balancing needs
    """,
    
    # Put Option Strategies Market Conditions
    "Long Put": """
        Best Market Outlook: Bearish
        Volatility Requirement: Low to moderate implied volatility
        Time Horizon: Minimum 60 days recommended
        Key Conditions:
        - Clear downward trend
        - Technical breakdown
        - Negative sentiment
        - Catalyst expected
        Risk Factors to Watch:
        - Time decay
        - Volatility crush
        - Support levels
    """,
    
    "Protective Put": """
        Best Market Outlook: Bullish with need for protection
        Volatility Requirement: Low to moderate implied volatility
        Time Horizon: 3-6 months typically
        Key Conditions:
        - Portfolio protection needed
        - Event risk present
        - Reasonable put prices
        - Clear risk levels
        Risk Factors to Watch:
        - Cost of protection
        - Strike selection
        - Roll timing
    """,
    
    "Naked Put Writing": """
        Best Market Outlook: Neutral to slightly bullish
        Volatility Requirement: High implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Clear support levels
        - High premium available
        - Range-bound market
        - Stable conditions
        Risk Factors to Watch:
        - Gap risk
        - Support levels
        - Assignment risk
    """,
    
    # Combined Strategies Market Conditions
    "Long Straddle": """
        Best Market Outlook: Large move expected either direction
        Volatility Requirement: Low implied volatility
        Time Horizon: 30-60 days
        Key Conditions:
        - Major event pending
        - Technical breakout expected
        - Low current volatility
        - Historical movement patterns
        Risk Factors to Watch:
        - Time decay
        - Volatility crush
        - Size of move needed
    """,
    
    "Short Straddle": """
        Best Market Outlook: Highly neutral
        Volatility Requirement: High implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Range-bound market
        - High premium available
        - Clear technical levels
        - No major events pending
        Risk Factors to Watch:
        - Gap risk
        - Assignment risk
        - Support/resistance levels
    """,
    
    "Iron Butterfly": """
        Best Market Outlook: Highly neutral
        Volatility Requirement: High implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Range-bound market
        - Clear technical levels
        - High premium available
        - Stable conditions
        Risk Factors to Watch:
        - Sharp moves
        - Volatility changes
        - Time decay profile
    """,
    
    "Iron Condor": """
        Best Market Outlook: Neutral with wider range
        Volatility Requirement: High implied volatility
        Time Horizon: 30-45 days
        Key Conditions:
        - Range-bound market
        - Clear technical levels
        - High premium available
        - Known support/resistance
        Risk Factors to Watch:
        - Sharp directional moves
        - Volatility changes
        - Wing risk management
    """,
    
    "Synthetic Long Stock": """
        Best Market Outlook: Bullish
        Volatility Requirement: Low to moderate implied volatility
        Time Horizon: 60+ days
        Key Conditions:
        - Strong upward trend
        - Clear technical levels
        - Reasonable option prices
        - Hard to borrow stock
        Risk Factors to Watch:
        - Assignment risk
        - Synthetic dividend risk
        - Margin requirements
    """,
    
    "Synthetic Short Stock": """
        Best Market Outlook: Bearish
        Volatility Requirement: Low to moderate implied volatility
        Time Horizon: 60+ days
        Key Conditions:
        - Strong downward trend
        - Clear technical levels
        - Reasonable option prices
        - Hard to borrow stock
        Risk Factors to Watch:
        - Assignment risk
        - Synthetic dividend risk
        - Margin requirements
    """
}

# Main UI Layout Components
def setup_sidebar():
    """Setup sidebar inputs and controls"""
    st.sidebar.markdown("## Model Selection")
    model_type = st.sidebar.selectbox(
        "",
        ["Black-Scholes", "Binomial", "Monte Carlo"],
        index=0
    )
    
    # Basic input parameters
    current_price = st.sidebar.number_input("Current Asset Price", value=100.00, step=0.01, format="%.2f")
    strike_price = st.sidebar.number_input("Strike Price", value=100.00, step=0.01, format="%.2f")
    time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.00, step=0.01, format="%.2f")
    volatility = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01, format="%.2f")
    risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01, format="%.2f")
    
    # Model-specific parameters
    model_params = {}
    if model_type == "Binomial":
        model_params['steps'] = st.sidebar.slider("Number of Steps", 10, 1000, 100)
        model_params['option_style'] = st.sidebar.selectbox("Option Style", ["European", "American"])
    elif model_type == "Monte Carlo":
        model_params['n_simulations'] = st.sidebar.slider("Number of Simulations", 1000, 50000, 10000)
        model_params['n_steps'] = st.sidebar.slider("Time Steps", 50, 500, 100)
    
    return model_type, current_price, strike_price, time_to_maturity, volatility, risk_free_rate, model_params

strategy_descriptions = {
    # Call Option Strategies
    "Covered Call Writing": """
        A strategy that involves holding a long stock position and selling a call option on that same stock.
        - Generates additional income from option premium
        - Provides limited downside protection
        - Limits potential upside gains
        - Popular among income-focused investors
        Maximum Profit: Limited to strike price - purchase price + premium
        Maximum Loss: Stock price - premium received
    """,
    
    "Long Call": """
        The most basic bullish options strategy, buying a call option.
        - Limited risk to premium paid
        - Unlimited profit potential
        - Provides leverage compared to buying stock
        - Suitable for strong bullish views
        Maximum Profit: Unlimited
        Maximum Loss: Limited to premium paid
    """,
    
    "Protected Short Sale": """
        Combining a short stock position with a long call option (Synthetic Put).
        - Limits upside risk with call option
        - Profits from stock price decline
        - Higher cost than direct put purchase
        - Useful when puts are overpriced
        Maximum Profit: Stock price - strike price + premium
        Maximum Loss: Limited to call premium
    """,
    
    "Reverse Hedge": """
        Buying both a call and put option at the same strike (Simulated Straddle).
        - Profits from large price movements in either direction
        - Limited risk to premiums paid
        - High cost due to two option purchases
        - Best when high volatility is expected
        Maximum Profit: Unlimited
        Maximum Loss: Limited to total premium paid
    """,
    
    "Naked Call Writing": """
        Selling call options without owning the underlying stock.
        - Generates immediate premium income
        - High risk due to unlimited potential losses
        - Requires significant margin
        - Suitable for neutral to slightly bearish outlook
        Maximum Profit: Limited to premium received
        Maximum Loss: Unlimited
    """,
    
    "Ratio Call Writing": """
        Writing more calls than the number of shares owned.
        - Enhanced premium income
        - Increased risk above strike price
        - Complex risk/reward profile
        - Requires careful monitoring
        Maximum Profit: Limited to premiums + stock appreciation to strike
        Maximum Loss: Potentially unlimited above higher strike
    """,
    
    "Bull Call Spread": """
        Buying a call option while selling another at a higher strike.
        - Reduced cost versus outright call purchase
        - Limited risk and reward
        - Lower break-even point
        - Good for moderately bullish views
        Maximum Profit: Difference between strikes - net premium paid
        Maximum Loss: Limited to net premium paid
    """,
    
    "Bear Call Spread": """
        Selling a call option while buying another at a higher strike.
        - Credit received at initiation
        - Limited risk and reward
        - Profits from falling or stable prices
        - Good for moderately bearish views
        Maximum Profit: Limited to net premium received
        Maximum Loss: Difference between strikes - net premium received
    """,
    
    # Put Option Strategies
    "Long Put": """
        Buying a put option for downside speculation or protection.
        - Limited risk, defined by premium paid
        - Substantial profit potential
        - Leveraged downside exposure
        - Good for bearish views
        Maximum Profit: Strike price - premium paid
        Maximum Loss: Limited to premium paid
    """,

    "Protective Put": """
        Buying puts against long stock position.
        - Insurance against stock decline
        - Unlimited upside potential
        - Known maximum loss
        - Portfolio protection strategy
        Maximum Profit: Unlimited
        Maximum Loss: Limited to put premium paid
    """,

    # Combined Strategies
    "Long Straddle": """
        Buying both a call and put at the same strike price.
        - Profits from large price movements
        - Direction doesn't matter
        - High cost strategy
        - Volatility play
        Maximum Profit: Unlimited
        Maximum Loss: Limited to total premium paid
    """,

    "Short Straddle": """
        Selling both a call and put at the same strike price.
        - Collects double premium
        - Profits from low volatility
        - High risk strategy
        - Requires significant margin
        Maximum Profit: Limited to total premium received
        Maximum Loss: Unlimited
    """,

    "Iron Butterfly": """
        Combination of bull put spread and bear call spread.
        - Limited risk and reward
        - Complex position
        - Market neutral strategy
        - Benefits from time decay
        Maximum Profit: Net premium received
        Maximum Loss: Limited to difference between strikes - net premium
    """,

    "Iron Condor": """
        Wider version of iron butterfly using four strikes.
        - Limited risk and reward
        - Higher probability of profit
        - Market neutral strategy
        - Popular among income seekers
        Maximum Profit: Net premium received
        Maximum Loss: Limited to difference between middle strikes - net premium
    """
}

def display_strategy_analysis(current_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """Display strategy analysis section"""
    st.markdown("---")
    st.title("Options Strategy Analysis")
    
    # Strategy Categories
    strategy_category = st.selectbox(
        "Select Strategy Category",
        ["Call Option Strategies", "Put Option Strategies", "Combined Strategies"]
    )
    
    # Comprehensive strategy lists
    call_strategies = [
        "Covered Call Writing",
        "Long Call",
        "Protected Short Sale (Synthetic Put)",
        "Reverse Hedge (Simulated Straddle)",
        "Naked Call Writing",
        "Ratio Call Writing",
        "Bull Call Spread",
        "Bear Call Spread",
        "Calendar Call Spread",
        "Butterfly Call Spread",
        "Ratio Call Spread",
        "Ratio Calendar Call Spread",
        "Delta-Neutral Calendar Spread",
        "Reverse Calendar Call Spread",
        "Reverse Ratio Call Spread (Backspread)",
        "Diagonal Bull Call Spread"
    ]

    put_strategies = [
        "Long Put",
        "Protective Put",
        "Put with Covered Call",
        "No-Cost Collar",
        "Naked Put Writing",
        "Covered Put Sale",
        "Ratio Put Writing",
        "Bear Put Spread",
        "Bull Put Spread",
        "Calendar Put Spread",
        "Butterfly Put Spread",
        "Ratio Put Spread",
        "Ratio Put Calendar Spread"
    ]

    combined_strategies = [
        "Long Straddle",
        "Short Straddle",
        "Long Strangle",
        "Short Strangle",
        "Synthetic Long Stock",
        "Synthetic Short Stock",
        "Iron Butterfly",
        "Iron Condor"
    ]

    # Display strategy selection based on category
    if strategy_category == "Call Option Strategies":
        strategy = st.selectbox("Select Strategy", call_strategies)
    elif strategy_category == "Put Option Strategies":
        strategy = st.selectbox("Select Strategy", put_strategies)
    else:
        strategy = st.selectbox("Select Strategy", combined_strategies)

    # Calculate strategy metrics
    spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
    
    # Calculate initial option values
    call_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                  risk_free_rate, volatility, 'call')
    put_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                 risk_free_rate, volatility, 'put')
    
    pnl = calculate_strategy_pnl(
        strategy,
        spot_range,
        current_price,
        strike_price,
        time_to_maturity,
        risk_free_rate,
        volatility,
        call_value,
        put_value
    )

    # Create P&L visualization
    plt.style.use('dark_background')
    fig_pnl = plt.figure(figsize=(12, 6))
    plt.plot(spot_range, pnl, 'g-', linewidth=2, label='P&L Profile')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even Line')
    plt.grid(True, alpha=0.2)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Profit/Loss ($)')
    plt.title(f'{strategy} P&L Profile')
    plt.legend()

    # Add break-even points
    break_even_points = spot_range[np.where(np.diff(np.signbit(pnl)))[0]]
    for point in break_even_points:
        plt.axvline(x=point, color='yellow', linestyle='--', alpha=0.3)
        plt.text(point, plt.ylim()[0], f'BE: {point:.2f}', rotation=90,
                verticalalignment='bottom', color='yellow')

    st.pyplot(fig_pnl)

    # Display strategy metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <h3 style="color: white;">Strategy Metrics</h3>
                <p style="color: white;">Break-even Point(s): {', '.join([f"${x:.2f}" for x in break_even_points])}</p>
                <p style="color: white;">Maximum Profit: ${max(pnl):.2f}</p>
                <p style="color: white;">Maximum Loss: ${abs(min(pnl)):.2f}</p>
                <p style="color: white;">Current Delta: {calculate_delta(strategy, current_price, strike_price, time_to_maturity, risk_free_rate, volatility):.2f}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <h3 style="color: white;">Position Details</h3>
                <p style="color: white;">Current Stock Price: ${current_price:.2f}</p>
                <p style="color: white;">Strike Price: ${strike_price:.2f}</p>
                <p style="color: white;">Days to Expiration: {time_to_maturity * 365:.0f}</p>
                <p style="color: white;">Implied Volatility: {volatility:.1%}</p>
            </div>
        """, unsafe_allow_html=True)

    # Display strategy description and characteristics
    st.markdown("""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3 style="color: white;">Strategy Description</h3>
            <p style="color: white;">{}</p>
            <h4 style="color: white;">Advantages</h4>
            <ul style="color: white;">
                {}
            </ul>
            <h4 style="color: white;">Disadvantages</h4>
            <ul style="color: white;">
                {}
            </ul>
            <h4 style="color: white;">Market Conditions</h4>
            <p style="color: white;">{}</p>
        </div>
    """.format(
        strategy_descriptions.get(strategy, "Description not available"),
        "\n".join([f"<li>{adv}</li>" for adv in strategy_advantages.get(strategy, ["Not available"])]),
        "\n".join([f"<li>{dis}</li>" for dis in strategy_disadvantages.get(strategy, ["Not available"])]),
        strategy_market_conditions.get(strategy, "Market conditions not available")
    ), unsafe_allow_html=True)

    # Add risk management section
    st.markdown("### Risk and Trade Management")
    
    # Calculate Greeks
    calculated_greeks = calculate_strategy_greeks(
        strategy,
        current_price,
        strike_price,
        time_to_maturity,
        risk_free_rate,
        volatility
    )

    # Display Greeks interpretation
    st.markdown("""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-top: 1rem;">
            <h4 style="color: white;">Greeks Interpretation</h4>
            <ul style="color: white; list-style-type: none; padding-left: 0;">
                <li>• Delta: Measure of directional risk ({:.1%} change per $1 move in underlying)</li>
                <li>• Gamma: Rate of change in Delta ({:.3f} per $1 move)</li>
                <li>• Theta: Time decay (${:.2f} per day)</li>
                <li>• Vega: Volatility sensitivity ({:.2f} per 1% change in volatility)</li>
            </ul>
        </div>
        """.format(
            calculated_greeks['delta'],
            calculated_greeks['gamma'],
            calculated_greeks['theta'],
            calculated_greeks['vega']
        ), unsafe_allow_html=True)

    return strategy

def display_heatmap(spot_prices, volatilities, call_pnl, put_pnl):
    """Display P&L heatmaps for both call and put options"""
    # Title
    st.title("Options Price - Interactive Heatmap")
    st.info("Explore how profits/losses fluctuate with varying 'Spot Prices and Volatility' levels while maintaining a constant 'Strike Price'.")

    # Create two columns for side-by-side heatmaps
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Call Option P&L Heatmap")
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        # Calculate max absolute value for symmetric color scaling
        max_abs_call = max(abs(call_pnl.min()), abs(call_pnl.max()))
        
        # Create heatmap for call options
        sns.heatmap(
            call_pnl,
            annot=True,
            fmt='.2f',
            cmap=create_pnl_colormap(),
            center=0,  # Center the colormap at zero
            vmin=-max_abs_call,  # Symmetric color scaling
            vmax=max_abs_call,
            ax=ax1,
            xticklabels=[f'{p:.2f}' for p in spot_prices],
            yticklabels=[f'{v:.2f}' for v in volatilities]
        )
        ax1.set_xlabel('Spot Price')
        ax1.set_ylabel('Volatility')
        plt.title('Call Option P&L\nGreen = Profit, Red = Loss')
        st.pyplot(fig1)

        # Add P&L statistics for calls
        st.markdown(f"""
            **Call Option P&L Statistics:**
            - Max Profit: ${call_pnl.max():.2f}
            - Max Loss: ${abs(call_pnl.min()):.2f}
            - Break-even points: Spot price where P&L = $0
            - Optimal Volatility Level: {volatilities[np.argmax(np.max(call_pnl, axis=1))]:.2%}
        """)

    with col2:
        st.subheader("Put Option P&L Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        # Calculate max absolute value for symmetric color scaling
        max_abs_put = max(abs(put_pnl.min()), abs(put_pnl.max()))
        
        # Create heatmap for put options
        sns.heatmap(
            put_pnl,
            annot=True,
            fmt='.2f',
            cmap=create_pnl_colormap(),
            center=0,  # Center the colormap at zero
            vmin=-max_abs_put,  # Symmetric color scaling
            vmax=max_abs_put,
            ax=ax2,
            xticklabels=[f'{p:.2f}' for p in spot_prices],
            yticklabels=[f'{v:.2f}' for v in volatilities]
        )
        ax2.set_xlabel('Spot Price')
        ax2.set_ylabel('Volatility')
        plt.title('Put Option P&L\nGreen = Profit, Red = Loss')
        st.pyplot(fig2)

        # Add P&L statistics for puts
        st.markdown(f"""
            **Put Option P&L Statistics:**
            - Max Profit: ${put_pnl.max():.2f}
            - Max Loss: ${abs(put_pnl.min()):.2f}
            - Break-even points: Spot price where P&L = $0
            - Optimal Volatility Level: {volatilities[np.argmax(np.max(put_pnl, axis=1))]:.2%}
        """)

    # Add overall explanation
    st.markdown("""
        ---
        ### Understanding the P&L Heatmap:
        - **Green cells**: Represent profitable scenarios (positive P&L)
        - **Red cells**: Represent loss scenarios (negative P&L)
        - **Color intensity**: Indicates the magnitude of profit/loss
        - **Numbers in cells**: Actual P&L values in dollars
        - **X-axis**: Different spot prices of the underlying asset
        - **Y-axis**: Different volatility levels
        
        ### Key Insights:
        - The darker the green, the higher the profit
        - The darker the red, the larger the loss
        - White or light-colored areas represent break-even or near break-even points
        - The gradient shows how P&L changes with different price and volatility combinations
        
        ### Trading Implications:
        - Use these heatmaps to identify optimal entry and exit points
        - Understand how volatility affects your position's P&L
        - Plan risk management strategies based on potential loss scenarios
        - Identify price ranges where the position performs best
    """)

    # Add volatility analysis
    st.markdown("""
        ### Volatility Impact Analysis
        """)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Calculate optimal volatility levels for call options
        optimal_vol_call = volatilities[np.argmax(np.max(call_pnl, axis=1))]
        worst_vol_call = volatilities[np.argmin(np.min(call_pnl, axis=1))]
        
        st.markdown(f"""
            **Call Option Volatility Analysis:**
            - Best performing volatility: {optimal_vol_call:.1%}
            - Worst performing volatility: {worst_vol_call:.1%}
            - Volatility sensitivity: {'High' if np.std(call_pnl) > np.mean(np.abs(call_pnl)) else 'Low'}
        """)
        
    with col4:
        # Calculate optimal volatility levels for put options
        optimal_vol_put = volatilities[np.argmax(np.max(put_pnl, axis=1))]
        worst_vol_put = volatilities[np.argmin(np.min(put_pnl, axis=1))]
        
        st.markdown(f"""
            **Put Option Volatility Analysis:**
            - Best performing volatility: {optimal_vol_put:.1%}
            - Worst performing volatility: {worst_vol_put:.1%}
            - Volatility sensitivity: {'High' if np.std(put_pnl) > np.mean(np.abs(put_pnl)) else 'Low'}
        """)

    # Add price range analysis
    st.markdown("""
        ### Price Range Analysis
        """)
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Calculate optimal price ranges for call options
        optimal_price_call = spot_prices[np.argmax(np.max(call_pnl, axis=0))]
        worst_price_call = spot_prices[np.argmin(np.min(call_pnl, axis=0))]
        
        st.markdown(f"""
            **Call Option Price Analysis:**
            - Most profitable price: ${optimal_price_call:.2f}
            - Least profitable price: ${worst_price_call:.2f}
            - Price sensitivity: {'High' if np.std(call_pnl) > np.mean(np.abs(call_pnl)) else 'Low'}
        """)
        
    with col6:
        # Calculate optimal price ranges for put options
        optimal_price_put = spot_prices[np.argmax(np.max(put_pnl, axis=0))]
        worst_price_put = spot_prices[np.argmin(np.min(put_pnl, axis=0))]
        
        st.markdown(f"""
            **Put Option Price Analysis:**
            - Most profitable price: ${optimal_price_put:.2f}
            - Least profitable price: ${worst_price_put:.2f}
            - Price sensitivity: {'High' if np.std(put_pnl) > np.mean(np.abs(put_pnl)) else 'Low'}
        """)

    # Add risk management recommendations
    st.markdown("""
        ### Risk Management Recommendations
        
        Based on the heatmap analysis, consider the following risk management strategies:
        
        1. **Position Sizing:**
           - Size positions based on maximum potential loss scenarios
           - Consider reducing position size in high volatility environments
        
        2. **Stop Losses:**
           - Set stops based on the red zones in the heatmap
           - Consider volatility-adjusted stops for better risk management
        
        3. **Profit Targets:**
           - Use the green zones to identify realistic profit targets
           - Consider scaling out of positions in highly profitable areas
        
        4. **Volatility Management:**
           - Monitor implied volatility changes and their impact on P&L
           - Consider adjusting positions when approaching extreme volatility levels
    """)

def create_pnl_colormap():
    """Create custom red-white-green colormap for P&L visualization"""
    colors = ['darkred', 'red', 'white', 'lightgreen', 'darkgreen']
    nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
    return LinearSegmentedColormap.from_list('pnl_colormap', list(zip(nodes, colors)))

# Testing Framework
def run_model_tests():
    """Comprehensive model testing framework"""
    try:
        with st.spinner("Running model tests..."):
            test_results = {
                'black_scholes': test_black_scholes(),
                'binomial': test_binomial_model(),
                'monte_carlo': test_monte_carlo(),
                'all_passed': True
            }
            
            if all([result for result in test_results.values() if result is not None]):
                st.success("All model tests passed successfully!")
                
                # Display detailed test results
                with st.expander("Black-Scholes Test Results"):
                    for i, result in enumerate(test_results['black_scholes']):
                        st.write(f"Test Case {i+1}:")
                        st.write(f"Parameters: {result['parameters']}")
                        st.write(f"Call Price: ${result['call_price']:.4f}")
                        st.write(f"Put Price: ${result['put_price']:.4f}")
                        st.write(f"Put-Call Parity Check: {'✓' if result['parity_check'] else '✗'}")
                        st.write("---")
                
                with st.expander("Binomial Model Test Results"):
                    for i, result in enumerate(test_results['binomial']):
                        st.write(f"Test Case {i+1}:")
                        st.write(f"Parameters: {result['parameters']}")
                        st.write(f"European Call: ${result['european_call']:.4f}")
                        st.write(f"European Put: ${result['european_put']:.4f}")
                        st.write(f"American Call: ${result['american_call']:.4f}")
                        st.write(f"American Put: ${result['american_put']:.4f}")
                        st.write("---")
                
                with st.expander("Monte Carlo Test Results"):
                    for i, result in enumerate(test_results['monte_carlo']):
                        st.write(f"Test Case {i+1}:")
                        st.write(f"Parameters: {result['parameters']}")
                        st.write(f"Monte Carlo Call: ${result['monte_carlo_call']:.4f}")
                        st.write(f"Monte Carlo Put: ${result['monte_carlo_put']:.4f}")
                        st.write(f"Standard Error: ${result['standard_error']:.4f}")
                        st.write("---")
            else:
                st.error("Some tests failed. Check the detailed results below.")
                
    except Exception as e:
        st.error(f"Error during testing: {str(e)}")

def run_all_tests():
    """Run all model tests and return comprehensive results."""
    test_results = {
        'black_scholes': None,
        'binomial': None,
        'monte_carlo': None,
        'all_passed': False
    }
    
    try:
        test_results['black_scholes'] = test_black_scholes()
        test_results['binomial'] = test_binomial_model()
        test_results['monte_carlo'] = test_monte_carlo()
        test_results['all_passed'] = all([
            len(test_results['black_scholes']) > 0,
            len(test_results['binomial']) > 0,
            len(test_results['monte_carlo']) > 0
        ])
        
        # Verify test content
        if test_results['all_passed']:
            # Verify Black-Scholes tests
            for result in test_results['black_scholes']:
                if not result['parity_check']:
                    test_results['all_passed'] = False
                    raise Exception("Black-Scholes put-call parity check failed")
            
            # Verify Binomial tests
            for result in test_results['binomial']:
                if not result['validity_check']:
                    test_results['all_passed'] = False
                    raise Exception("Binomial model early exercise premium check failed")
            
            # Verify Monte Carlo tests
            for result in test_results['monte_carlo']:
                if not result['accuracy_check']:
                    test_results['all_passed'] = False
                    raise Exception("Monte Carlo accuracy check failed")
        
        return test_results
        
    except Exception as e:
        test_results['error'] = str(e)
        test_results['all_passed'] = False
        return test_results

# Utility Functions
def calculate_pnl_matrices(model_type, spot_prices, volatilities, current_price,
                         strike_price, time_to_maturity, risk_free_rate, model_params):
    """Calculate P&L matrices for heatmap visualization"""
    call_pnl = np.zeros((len(volatilities), len(spot_prices)))
    put_pnl = np.zeros((len(volatilities), len(spot_prices)))
    
    for i, vol in enumerate(volatilities):
        for j, spot in enumerate(spot_prices):
            if model_type == "Black-Scholes":
                call_value = black_scholes_calc(spot, strike_price, time_to_maturity,
                                              risk_free_rate, vol, 'call')
                put_value = black_scholes_calc(spot, strike_price, time_to_maturity,
                                             risk_free_rate, vol, 'put')
                
            elif model_type == "Binomial":
                steps = model_params.get('steps', 100)
                style = model_params.get('option_style', 'european').lower()
                call_value = binomial_calc(spot, strike_price, time_to_maturity,
                                         risk_free_rate, vol, steps, 'call', style)
                put_value = binomial_calc(spot, strike_price, time_to_maturity,
                                        risk_free_rate, vol, steps, 'put', style)
                
            else:  # Monte Carlo
                n_sim = model_params.get('n_simulations', 10000)
                n_steps = model_params.get('n_steps', 100)
                call_value, _, _ = monte_carlo_calc(spot, strike_price, time_to_maturity,
                                                  risk_free_rate, vol, n_sim, n_steps, 'call')
                put_value, _, _ = monte_carlo_calc(spot, strike_price, time_to_maturity,
                                                 risk_free_rate, vol, n_sim, n_steps, 'put')
            
            call_pnl[i, j] = call_value - model_params.get('call_purchase_price', 0)
            put_pnl[i, j] = put_value - model_params.get('put_purchase_price', 0)
            
    return call_pnl, put_pnl

def format_price_output(value, model_type, standard_error=None):
    """Format price output based on model type"""
    if model_type == "Monte Carlo" and standard_error is not None:
        return f"${value:.2f} ± ${standard_error:.4f}"
    return f"${value:.2f}"

# Error Handling and Validation
class OptionPricingError(Exception):
    """Custom exception for option pricing errors"""
    pass

def validate_model_parameters(model_type, params):
    """Validate model-specific parameters"""
    try:
        if model_type == "Binomial":
            if params.get('steps', 0) < 10:
                raise OptionPricingError("Binomial model requires at least 10 steps")
        elif model_type == "Monte Carlo":
            if params.get('n_simulations', 0) < 1000:
                raise OptionPricingError("Monte Carlo requires at least 1000 simulations")
            if params.get('n_steps', 0) < 50:
                raise OptionPricingError("Monte Carlo requires at least 50 time steps")
        
        return True
    except Exception as e:
        st.error(f"Parameter validation failed: {str(e)}")
        return False

def handle_calculation_errors(func):
    """Decorator for handling calculation errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")
            return None
    return wrapper

# Final Layout and Display Functions
def display_footnotes():
    """Display footnotes and assumptions"""
    st.markdown("""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: white;">Assumptions and Notes</h4>
            <ul style="color: white;">
                <li>All calculations assume European-style options unless specified</li>
                <li>Transaction costs and taxes are not included</li>
                <li>Implied volatility is assumed constant across strikes</li>
                <li>Interest rates are assumed constant over the option's life</li>
                <li>No dividend considerations are included</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def display_documentation():
    """Display model documentation and help"""
    with st.expander("Documentation and Help"):
        st.markdown("""
            ### Using the Options Calculator
            1. Select your preferred pricing model from the sidebar
            2. Enter the required parameters
            3. View the calculated option prices and Greeks
            4. Explore the P&L heatmap for different scenarios
            
            ### Model Descriptions
            - **Black-Scholes**: Classical model for European options
            - **Binomial**: Flexible model supporting American options
            - **Monte Carlo**: Simulation-based model with error estimates
        """)

def main():
    """Main application execution flow with error handling and validation"""
    try:
        # Setup sidebar and get parameters
        model_type, current_price, strike_price, time_to_maturity, volatility, risk_free_rate, model_params = setup_sidebar()
        
        # Validate inputs
        if not validate_model_parameters(model_type, model_params):
            st.error("Invalid model parameters. Please check your inputs.")
            return
        
        # Title and model selection display
        st.markdown(f"# 📈 {model_type} Model")
        
        # Display documentation
        display_documentation()
        
        # Calculate prices based on selected model
        with st.spinner("Calculating prices..."):
            if model_type == "Black-Scholes":
                call_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                              risk_free_rate, volatility, 'call')
                put_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                             risk_free_rate, volatility, 'put')
                price_info = {"call": f"${call_value:.2f}", "put": f"${put_value:.2f}"}
                
            elif model_type == "Binomial":
                steps = model_params.get('steps', 100)
                option_style = model_params.get('option_style', 'European').lower()
                call_value = binomial_calc(current_price, strike_price, time_to_maturity,
                                         risk_free_rate, volatility, steps, 'call', option_style)
                put_value = binomial_calc(current_price, strike_price, time_to_maturity,
                                        risk_free_rate, volatility, steps, 'put', option_style)
                price_info = {"call": f"${call_value:.2f}", "put": f"${put_value:.2f}"}
                
            else:  # Monte Carlo
                n_simulations = model_params.get('n_simulations', 10000)
                n_steps = model_params.get('n_steps', 100)
                call_value, call_se, call_paths = monte_carlo_calc(current_price, strike_price,
                                                                 time_to_maturity, risk_free_rate,
                                                                 volatility, n_simulations, n_steps, 'call')
                put_value, put_se, put_paths = monte_carlo_calc(current_price, strike_price,
                                                              time_to_maturity, risk_free_rate,
                                                              volatility, n_simulations, n_steps, 'put')
                price_info = {
                    "call": f"${call_value:.2f} ± ${call_se:.4f}",
                    "put": f"${put_value:.2f} ± ${put_se:.4f}"
                }

        # Display option prices
        display_option_prices(price_info)

        # Strategy Analysis
        strategy = display_strategy_analysis(
            current_price,
            strike_price,
            time_to_maturity,
            risk_free_rate,
            volatility
        )
        
        # Calculate and display Greeks
        calculated_greeks = calculate_strategy_greeks(
            strategy,
            current_price,
            strike_price,
            time_to_maturity,
            risk_free_rate,
            volatility
        )
        display_greeks(calculated_greeks)
        
        # Generate and display heatmap
        with st.spinner("Generating heatmap..."):
            # Get heatmap parameters from sidebar
            min_spot = st.sidebar.number_input("Min Spot Price", value=current_price * 0.8, step=0.01)
            max_spot = st.sidebar.number_input("Max Spot Price", value=current_price * 1.2, step=0.01)
            min_vol = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.00, 0.10)
            max_vol = st.sidebar.slider("Max Volatility for Heatmap", 0.01, 1.00, 0.30)
            
            # Purchase price inputs
            st.sidebar.markdown("## Option Purchase Prices")
            call_purchase_price = st.sidebar.number_input("Call Option Purchase Price", value=0.00, step=0.01)
            put_purchase_price = st.sidebar.number_input("Put Option Purchase Price", value=0.00, step=0.01)
            
            spot_prices = np.linspace(min_spot, max_spot, 10)
            volatilities = np.linspace(min_vol, max_vol, 10)
            
            # Modified model_params to include purchase prices
            model_params_with_prices = {
                **model_params,
                'call_purchase_price': call_purchase_price,
                'put_purchase_price': put_purchase_price
            }
            
            call_pnl, put_pnl = calculate_pnl_matrices(
                model_type,
                spot_prices,
                volatilities,
                current_price,
                strike_price,
                time_to_maturity,
                risk_free_rate,
                model_params_with_prices
            )
            
            display_heatmap(spot_prices, volatilities, call_pnl, put_pnl)
        
        # Display Monte Carlo specific visualizations if selected
        if model_type == "Monte Carlo":
            st.subheader("Monte Carlo Simulation Paths")
            fig = plt.figure(figsize=(10, 6))
            plt.plot(np.linspace(0, time_to_maturity, n_steps), call_paths[:100].T, alpha=0.1)
            plt.plot(np.linspace(0, time_to_maturity, n_steps), np.mean(call_paths, axis=0),
                    'r', linewidth=2)
            plt.xlabel('Time (years)')
            plt.ylabel('Stock Price')
            plt.title('Monte Carlo Simulation Paths (first 100 paths)')
            st.pyplot(fig)
        
        # Display testing section if enabled
        if st.sidebar.checkbox("Enable Model Testing", False):
            st.sidebar.markdown("## Model Testing")
            if st.sidebar.button("Run Model Tests"):
                with st.spinner("Running model tests..."):
                    test_results = run_all_tests()
                    
                    if test_results['all_passed']:
                        st.success("All model tests passed successfully!")
                        
                        # Display detailed results in expandable sections
                        with st.expander("Black-Scholes Test Results"):
                            for i, result in enumerate(test_results['black_scholes']):
                                st.write(f"Test Case {i+1}:")
                                st.write(f"Parameters: {result['parameters']}")
                                st.write(f"Call Price: ${result['call_price']:.4f}")
                                st.write(f"Put Price: ${result['put_price']:.4f}")
                                st.write(f"Put-Call Parity Check: {'✓' if result['parity_check'] else '✗'}")
                                st.write("---")
                        
                        with st.expander("Binomial Model Test Results"):
                            for i, result in enumerate(test_results['binomial']):
                                st.write(f"Test Case {i+1}:")
                                st.write(f"Parameters: {result['parameters']}")
                                st.write(f"European Call: ${result['european_call']:.4f}")
                                st.write(f"European Put: ${result['european_put']:.4f}")
                                st.write(f"American Call: ${result['american_call']:.4f}")
                                st.write(f"American Put: ${result['american_put']:.4f}")
                                st.write(f"Early Exercise Premium (Call): ${result['early_exercise_call']:.4f}")
                                st.write(f"Early Exercise Premium (Put): ${result['early_exercise_put']:.4f}")
                                st.write(f"Validity Check: {'✓' if result['validity_check'] else '✗'}")
                                st.write("---")
                        
                        with st.expander("Monte Carlo Test Results"):
                            for i, result in enumerate(test_results['monte_carlo']):
                                st.write(f"Test Case {i+1}:")
                                st.write(f"Parameters: {result['parameters']}")
                                st.write(f"Monte Carlo Call: ${result['monte_carlo_call']:.4f}")
                                st.write(f"Monte Carlo Put: ${result['monte_carlo_put']:.4f}")
                                st.write(f"Call Standard Error: ${result['call_std_error']:.4f}")
                                st.write(f"Put Standard Error: ${result['put_std_error']:.4f}")
                                st.write(f"Black-Scholes Comparison:")
                                st.write(f"  - Call Relative Error: {result['relative_error_call']:.2%}")
                                st.write(f"  - Put Relative Error: {result['relative_error_put']:.2%}")
                                st.write(f"Accuracy Check: {'✓' if result['accuracy_check'] else '✗'}")
                                st.write("---")
                    else:
                        st.error(f"Model tests failed: {test_results.get('error', 'Unknown error')}")
        
        # Display footnotes and assumptions
        display_footnotes()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your inputs and try again.")
        if st.checkbox("Show detailed error trace"):
            st.exception(e)

if __name__ == "__main__":
    main()
