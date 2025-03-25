import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objects as go
import warnings
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize


# Black-Scholes Option Pricing Model
def black_scholes_calc(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes pricing model for European options
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (decimal)
    sigma : float
        Volatility (decimal)
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float: Option price
    """
    if T <= 0 or sigma <= 0:
        # Handle edge cases for expiration or zero volatility
        if option_type == 'call':
            return max(0, S - K) if S > K else 0
        else:  # put
            return max(0, K - S) if K > S else 0
    
    # Calculate d1 and d2 parameters
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Calculate option price based on type
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Cox-Ross-Rubinstein Binomial Option Pricing Model
def binomial_calc(S, K, T, r, sigma, n, option_type='call', style='european'):
    """
    Binomial option pricing model
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (decimal)
    sigma : float
        Volatility (decimal)
    n : int
        Number of time steps
    option_type : str
        'call' or 'put'
    style : str
        'european' or 'american'
        
    Returns:
    --------
    float: Option price
    """
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))  # Up factor
    d = 1/u                         # Down factor
    p = (np.exp(r*dt) - d)/(u - d)  # Risk-neutral probability
    
    # Initialize stock price tree
    stock = np.zeros((n+1, n+1))
    stock[0,0] = S
    
    # Generate stock price tree
    for i in range(1, n+1):
        stock[0:i+1,i] = S * u**np.arange(i,-1,-1) * d**np.arange(0,i+1)
    
    # Initialize option value tree
    option = np.zeros((n+1, n+1))
    
    # Set terminal option values (at expiration)
    if option_type == 'call':
        option[:,n] = np.maximum(stock[:,n] - K, 0)
    else:  # put
        option[:,n] = np.maximum(K - stock[:,n], 0)
    
    # Backward recursion for option values
    for j in range(n-1,-1,-1):
        for i in range(j+1):
            if style == 'european':
                # European option: simply use risk-neutral pricing
                option[i,j] = np.exp(-r*dt)*(p*option[i,j+1] + (1-p)*option[i+1,j+1])
            else:  # american
                # American option: consider early exercise
                hold = np.exp(-r*dt)*(p*option[i,j+1] + (1-p)*option[i+1,j+1])
                if option_type == 'call':
                    exercise = stock[i,j] - K
                else:
                    exercise = K - stock[i,j]
                option[i,j] = max(hold, exercise)
    
    return option[0,0]

# Monte Carlo Simulation for Option Pricing
def monte_carlo_calc(S, K, T, r, sigma, n_sim, n_steps, option_type='call'):
    """
    Monte Carlo simulation for option pricing
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (decimal)
    sigma : float
        Volatility (decimal)
    n_sim : int
        Number of price path simulations
    n_steps : int
        Number of time steps per path
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    tuple: (option_price, standard_error, price_paths)
    """
    dt = T/n_steps
    nudt = (r - 0.5*sigma**2)*dt
    sigsqrtdt = sigma*np.sqrt(dt)
    
    # Generate random standard normal samples
    Z = np.random.standard_normal((n_sim, n_steps))
    
    # Initialize price paths array
    S_path = np.zeros((n_sim, n_steps+1))
    S_path[:,0] = S
    
    # Simulate price paths using Geometric Brownian Motion
    for t in range(1, n_steps+1):
        S_path[:,t] = S_path[:,t-1] * np.exp(nudt + sigsqrtdt*Z[:,t-1])
    
    # Calculate payoffs at expiration
    if option_type == 'call':
        payoffs = np.maximum(S_path[:,-1] - K, 0)
    else:  # put
        payoffs = np.maximum(K - S_path[:,-1], 0)
    
    # Calculate option price (present value of expected payoff)
    price = np.exp(-r*T)*np.mean(payoffs)
    
    # Calculate standard error
    se = np.exp(-r*T)*np.std(payoffs)/np.sqrt(n_sim)
    
    return price, se, S_path

# Calculate First-Order Greeks
def calculate_greeks(option_type, S, K, T, r, sigma):
    """
    Calculate first-order option Greeks
    
    Parameters:
    -----------
    option_type : str
        "Call" or "Put"
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (decimal)
    sigma : float
        Volatility (decimal)
        
    Returns:
    --------
    dict: Dictionary of first-order Greeks (delta, gamma, theta, vega)
    """
    if T <= 0 or sigma <= 0:
        # Handle edge cases
        if option_type == "Call":
            return {
                'delta': 1.0 if S > K else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
        else:  # Put
            return {
                'delta': -1.0 if S < K else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
    
    # Calculate d1 and d2 parameters
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Common calculations
    n_d1 = norm.pdf(d1)  # Standard normal probability density at d1
    
    # Gamma - second derivative of option price with respect to underlying price
    # Same for both call and put
    gamma = n_d1/(S * sigma * np.sqrt(T))
    
    # Vega - first derivative of option price with respect to volatility
    # Same for both call and put, typically expressed per 1% change in volatility
    vega = S * np.sqrt(T) * n_d1 * 0.01
    
    if option_type == "Call":
        # Delta - first derivative of option price with respect to underlying price
        delta = norm.cdf(d1)
        
        # Theta - first derivative of option price with respect to time
        # Typically expressed as daily decay (divided by 365)
        theta = (-S * n_d1 * sigma/(2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:  # Put
        delta = -norm.cdf(-d1)
        theta = (-S * n_d1 * sigma/(2 * np.sqrt(T)) +
                r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

# Calculate Higher-Order Greeks
def calculate_advanced_greeks(option_type, S, K, T, r, sigma):
    """
    Calculate higher-order option Greeks
    
    Parameters:
    -----------
    option_type : str
        "call" or "put"
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (decimal)
    sigma : float
        Volatility (decimal)
        
    Returns:
    --------
    dict: Dictionary of higher-order Greeks
    """
    if T <= 0 or sigma <= 0:
        return {
            'vanna': 0.0, 'charm': 0.0, 'volga': 0.0,
            'veta': 0.0, 'speed': 0.0, 'zomma': 0.0,
            'color': 0.0, 'ultima': 0.0
        }
    
    # Calculate parameters
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt_t)
    d2 = d1 - sigma*sqrt_t
    
    # Standard normal PDF values
    nd1 = norm.pdf(d1)
    nd2 = norm.pdf(d2)
    
    # Higher-order Greeks (most are same for calls and puts)
    
    # Vanna/DdeltaDvol - sensitivity of delta to volatility changes
    vanna = -nd1 * d2 / sigma
    
    # Volga/Vomma - second derivative of option price with respect to volatility
    volga = S * sqrt_t * nd1 * d1 * d2 / sigma
    
    # Charm/DdeltaDtime - rate of change of delta with respect to time
    if option_type.lower() == 'call':
        charm = -nd1 * (r/(sigma*sqrt_t) - d2/(2*T))
    else:  # put
        charm = nd1 * (r/(sigma*sqrt_t) - d2/(2*T))
    
    # Veta/DvegaDtime - rate of change of vega with respect to time
    veta = -S * nd1 * sqrt_t * (r*d1/(sigma*sqrt_t) - (1+d1*d2)/(2*T))
    
    # Speed - third derivative of option price with respect to underlying price
    speed = -nd1 * d1/(S**2 * sigma * sqrt_t) * (1 + d1/(sigma * sqrt_t))
    
    # Zomma - sensitivity of gamma to volatility changes
    zomma = nd1 * (d1*d2 - 1)/(S * sigma)
    
    # Color/DgammaDtime - rate of change of gamma with respect to time
    color = -nd1 * (r*d2 + d1*d2/(2*T) - (1+d1*d2)/(2*T) + r*d1/(sigma*sqrt_t))/(S * sigma * sqrt_t)
    
    # Ultima - sensitivity of volga to volatility changes
    ultima = -S * sqrt_t * nd1 / (sigma**2) * (d1*d2*(1-d1*d2) + d1**2 + d2**2)
    
    return {
        'vanna': vanna, 'charm': charm, 'volga': volga,
        'veta': veta, 'speed': speed, 'zomma': zomma,
        'color': color, 'ultima': ultima
    }

# Calculate Option Strategy Payoffs
def calculate_strategy_pnl(strategy_type, spot_range, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value=0, put_value=0):
    """Calculate P&L for option strategies based on precise mathematical formulas"""
    # Validate inputs
    if not isinstance(spot_range, (list, np.ndarray)):
        raise ValueError("spot_range must be a list or numpy array")
        
    # Add validation for numerical inputs
    for param in [current_price, strike_price, time_to_maturity, risk_free_rate, volatility]:
        if not isinstance(param, (int, float)) or param < 0:
            raise ValueError(f"Invalid parameter value: {param}")
    
    # Define common strikes for spreads
    lower_strike = strike_price * 0.9
    upper_strike = strike_price * 1.1
    
    # Calculate strikes for Iron Condor
    very_lower_strike = lower_strike * 0.95
    very_upper_strike = upper_strike * 1.05
    
    # Initialize PnL array - IMPORTANT: same length as spot_range
    pnl = np.zeros(len(spot_range))
    
    # Calculate option values at entry (for comparison in P&L)
    atm_call_value = call_value if call_value > 0 else black_scholes_calc(
        current_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'call')
    atm_put_value = put_value if put_value > 0 else black_scholes_calc(
        current_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'put')
    
    # For spreads - pre-calculate option values at other strikes
    lower_call_value = black_scholes_calc(
        current_price, lower_strike, time_to_maturity, risk_free_rate, volatility, 'call')
    upper_call_value = black_scholes_calc(
        current_price, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
    lower_put_value = black_scholes_calc(
        current_price, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
    upper_put_value = black_scholes_calc(
        current_price, upper_strike, time_to_maturity, risk_free_rate, volatility, 'put')
    
    # For Iron Condor
    very_lower_put_value = black_scholes_calc(
        current_price, very_lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
    very_upper_call_value = black_scholes_calc(
        current_price, very_upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
    
    for i, spot in enumerate(spot_range):
        # Calculate Call Option Strategies
        if strategy_type == "Covered Call Writing":
            # (ST - S0) + C if ST ≤ K, (K - S0) + C if ST > K
            if spot <= strike_price:
                pnl[i] = (spot - current_price) + atm_call_value
            else:
                pnl[i] = (strike_price - current_price) + atm_call_value
                
        elif strategy_type == "Long Call":
            # max(0, ST - K) - C
            pnl[i] = max(0, spot - strike_price) - atm_call_value
            
        elif strategy_type == "Bull Call Spread":
            # Long lower strike call, short higher strike call
            # max(0, ST - K1) - max(0, ST - K2) - Net Debit
            pnl[i] = max(0, spot - lower_strike) - max(0, spot - upper_strike) - (lower_call_value - upper_call_value)
            
        elif strategy_type == "Bear Call Spread":
            # Short lower strike call, long higher strike call
            # Net Credit - max(0, ST - K1) + max(0, ST - K2)
            pnl[i] = (lower_call_value - upper_call_value) - max(0, spot - lower_strike) + max(0, spot - upper_strike)
            
        # Calculate Put Option Strategies
        elif strategy_type == "Long Put":
            # max(0, K - ST) - P
            pnl[i] = max(0, strike_price - spot) - atm_put_value
            
        elif strategy_type == "Protective Put":
            # (ST - S0) + max(0, K - ST) - P
            pnl[i] = (spot - current_price) + max(0, strike_price - spot) - atm_put_value
            
        elif strategy_type == "Bull Put Spread":
            # Short higher strike put, long lower strike put
            # Net Credit - max(0, K1 - ST) + max(0, K2 - ST)
            pnl[i] = (upper_put_value - lower_put_value) - max(0, upper_strike - spot) + max(0, lower_strike - spot)
            
        elif strategy_type == "Bear Put Spread":
            # Long higher strike put, short lower strike put
            # max(0, K1 - ST) - max(0, K2 - ST) - Net Debit
            pnl[i] = max(0, upper_strike - spot) - max(0, lower_strike - spot) - (upper_put_value - lower_put_value)
            
        # Calculate Combined Strategies
        elif strategy_type == "Long Straddle":
            # max(0, ST - K) + max(0, K - ST) - (C + P)
            pnl[i] = max(0, spot - strike_price) + max(0, strike_price - spot) - (atm_call_value + atm_put_value)
            
        elif strategy_type == "Short Straddle":
            # (C + P) - max(0, ST - K) - max(0, K - ST)
            pnl[i] = (atm_call_value + atm_put_value) - max(0, spot - strike_price) - max(0, strike_price - spot)
            
        elif strategy_type == "Iron Butterfly":
            # Short ATM put, short ATM call, long OTM put, long OTM call
            # Net Credit - max(0, ST - K) + max(0, ST - (K + Δ)) - max(0, K - ST) + max(0, (K - Δ) - ST)
            net_credit = atm_call_value + atm_put_value - upper_call_value - lower_put_value
            pnl[i] = net_credit - max(0, spot - strike_price) + max(0, spot - upper_strike) - max(0, strike_price - spot) + max(0, lower_strike - spot)
            
        elif strategy_type == "Iron Condor":
            # Bull put spread + bear call spread
            # Net Credit - max(0, K1 - ST) + max(0, (K1 - Δ) - ST) - max(0, ST - K2) + max(0, ST - (K2 + Δ))
            net_credit = (lower_put_value - very_lower_put_value) + (upper_call_value - very_upper_call_value)
            pnl[i] = net_credit - max(0, lower_strike - spot) + max(0, very_lower_strike - spot) - max(0, spot - upper_strike) + max(0, spot - very_upper_strike)
        
        else:
            # Default for unknown strategies
            pnl[i] = max(0, spot - strike_price) - atm_call_value
    
    # Make sure we return an array with exactly the same length as spot_range
    return pnl

# Calculate Implied Volatility
#def implied_volatility(market_price, S, K, T, r, option_type='call', initial_guess=0.2, precision=1e-8):
#    """
#    Calculate implied volatility using optimization
#    
#    Parameters:
#    -----------
#    market_price : float
#        Observed market price of the option
#    S, K, T, r : float
#        Stock price, strike price, time to maturity (years), risk-free rate
#    option_type : str
#        'call' or 'put'
#    initial_guess : float
#        Initial volatility estimate
#    precision : float
#        Convergence threshold
#        
#    Returns:
#    --------
#    float: Implied volatility value
#    """
#    def objective(sigma):
#        price = black_scholes_calc(S, K, T, r, sigma, option_type)
#        return abs(price - market_price)
#    
#    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0.001, 5.0)])
#    if result.success:
#        return result.x[0]
#    else:
#        raise ValueError(f"Implied volatility calculation failed: {result.message}")
def implied_volatility(market_price, S, K, T, r, option_type='call', precision=1e-8, max_iterations=20):
    """
    Calculate implied volatility using enhanced analytical approximation and Newton-Raphson refinement
    
    Parameters:
    -----------
    market_price : float
        Observed market price of the option
    S, K, T, r : float
        Stock price, strike price, time to maturity (years), risk-free rate
    option_type : str
        'call' or 'put'
    precision : float
        Convergence threshold
    max_iterations : int
        Maximum number of Newton-Raphson iterations
        
    Returns:
    --------
    float: Implied volatility value
    """
    # Handle degenerate cases
    if T <= 0.01:
        return 0.3  # Default for very short-dated options
    
    # Set a more reasonable minimum volatility floor
    min_sigma = 0.05  # 5% minimum volatility
    
    # Calculate intrinsic value
    if option_type.lower() == 'call':
        intrinsic = max(0, S - K * np.exp(-r * T))
    else:  # put
        intrinsic = max(0, K * np.exp(-r * T) - S)
    
    # Time value
    time_value = max(0.01, market_price - intrinsic)
    
    # Moneyness
    moneyness = np.log(S / K) + r * T
    
    # Initial guess selection based on option characteristics
    if option_type.lower() == 'call':
        if K > S:  # OTM call
            # For OTM calls with low market prices, start with a reasonable volatility
            # instead of letting the algorithm drive it too low
            sigma = 0.15  # Start with 15% for OTM options
            
            # If market price is very low relative to spot, adjust initial guess
            if market_price < 0.01 * S:
                # Use bisection to find better initial guess
                low_vol, high_vol = 0.05, 0.5
                for _ in range(5):
                    mid_vol = (low_vol + high_vol) / 2
                    price = black_scholes_calc(S, K, T, r, mid_vol, option_type)
                    if price < market_price:
                        low_vol = mid_vol
                    else:
                        high_vol = mid_vol
                sigma = (low_vol + high_vol) / 2
        else:  # ITM or ATM call
            if abs(moneyness) < 0.1:  # Near ATM
                # Brenner-Subrahmanyam approximation for near-ATM options
                sigma = np.sqrt(2 * np.pi / T) * time_value / (S * np.exp(-r * T))
            else:  # ITM
                sigma = 0.2  # Start with 20% for ITM options
    else:  # put
        if K < S:  # OTM put
            # Similar approach for OTM puts
            sigma = 0.15  # Start with 15% for OTM options
            
            # Similar bisection for low-price OTM puts
            if market_price < 0.01 * S:
                low_vol, high_vol = 0.05, 0.5
                for _ in range(5):
                    mid_vol = (low_vol + high_vol) / 2
                    price = black_scholes_calc(S, K, T, r, mid_vol, option_type)
                    if price < market_price:
                        low_vol = mid_vol
                    else:
                        high_vol = mid_vol
                sigma = (low_vol + high_vol) / 2
        else:  # ITM or ATM put
            if abs(moneyness) < 0.1:  # Near ATM
                sigma = np.sqrt(2 * np.pi / T) * time_value / (S * np.exp(-r * T))
            else:  # ITM
                sigma = 0.2  # Start with 20% for ITM options
                
    # Ensure initial guess is reasonable
    sigma = max(min_sigma, min(sigma, 1.0))
    
    # Newton-Raphson refinement with robust safeguards
    for i in range(max_iterations):
        # Calculate option price
        price = black_scholes_calc(S, K, T, r, sigma, option_type)
        
        # Calculate difference
        diff = price - market_price
        
        # Check convergence
        if abs(diff) < precision:
            return sigma
        
        # Calculate vega (derivative of price with respect to volatility)
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        # More robust handling for tiny vegas
        if abs(vega) < 1e-5:
            vega = 1e-5 * (1 if vega >= 0 else -1)
        
        # Newton-Raphson update with dampening for stability
        delta_sigma = diff / vega
        
        # Apply dampening for large steps
        if abs(delta_sigma) > 0.1:
            delta_sigma = 0.1 * (delta_sigma / abs(delta_sigma))
            
        sigma = sigma - delta_sigma
        
        # Ensure sigma stays within reasonable bounds
        sigma = max(min_sigma, min(sigma, 1.0))
    
    # If we didn't converge, try grid search as a last resort
    if abs(price - market_price) > precision:
        vol_range = np.linspace(min_sigma, 1.0, 20)
        best_vol = min_sigma
        min_diff = float('inf')
        
        for vol in vol_range:
            price = black_scholes_calc(S, K, T, r, vol, option_type)
            diff = abs(price - market_price)
            
            if diff < min_diff:
                min_diff = diff
                best_vol = vol
        
        return best_vol
    
    return sigma
    
# Calculate Strategy Greeks
def calculate_strategy_greeks(strategy_type, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """
    Calculate Greeks for option strategies
    
    Parameters:
    -----------
    strategy_type : str
        Type of option strategy
    spot_price : float
        Current price of underlying
    strike_price : float
        Strike price
    time_to_maturity : float
        Time to expiration in years
    risk_free_rate : float
        Risk-free interest rate (decimal)
    volatility : float
        Volatility (decimal)
        
    Returns:
    --------
    dict: Strategy Greeks
    """
    # Define strikes for spreads
    lower_strike = strike_price * 0.9
    upper_strike = strike_price * 1.1
    
    # Initialize Greeks
    greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    # Calculate basic Greeks for standard options
    call_greeks = calculate_greeks("Call", spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    put_greeks = calculate_greeks("Put", spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    
    # Calculate Greeks for spread options
    upper_call_greeks = calculate_greeks("Call", spot_price, upper_strike, time_to_maturity, risk_free_rate, volatility)
    lower_call_greeks = calculate_greeks("Call", spot_price, lower_strike, time_to_maturity, risk_free_rate, volatility)
    upper_put_greeks = calculate_greeks("Put", spot_price, upper_strike, time_to_maturity, risk_free_rate, volatility)
    lower_put_greeks = calculate_greeks("Put", spot_price, lower_strike, time_to_maturity, risk_free_rate, volatility)
    
    # Calculate Greeks for Iron Condor
    very_lower_strike = lower_strike * 0.95
    very_upper_strike = upper_strike * 1.05
    very_lower_put_greeks = calculate_greeks("Put", spot_price, very_lower_strike, time_to_maturity, risk_free_rate, volatility)
    very_upper_call_greeks = calculate_greeks("Call", spot_price, very_upper_strike, time_to_maturity, risk_free_rate, volatility)
    
    # Calculate strategy-specific Greeks based on portfolio compositions
    if strategy_type == "Covered Call Writing":
        greeks['delta'] = 1 - call_greeks['delta']
        greeks['gamma'] = -call_greeks['gamma']
        greeks['theta'] = -call_greeks['theta']
        greeks['vega'] = -call_greeks['vega']
        
    elif strategy_type == "Long Call":
        greeks = call_greeks
        
    elif strategy_type == "Protected Short Sale":
        greeks['delta'] = -1 + call_greeks['delta']
        greeks['gamma'] = call_greeks['gamma']
        greeks['theta'] = call_greeks['theta']
        greeks['vega'] = call_greeks['vega']
        
    elif strategy_type == "Reverse Hedge":
        greeks['delta'] = call_greeks['delta'] + put_greeks['delta']
        greeks['gamma'] = call_greeks['gamma'] + put_greeks['gamma']
        greeks['theta'] = call_greeks['theta'] + put_greeks['theta']
        greeks['vega'] = call_greeks['vega'] + put_greeks['vega']
        
    elif strategy_type == "Naked Call Writing":
        greeks['delta'] = -call_greeks['delta']
        greeks['gamma'] = -call_greeks['gamma']
        greeks['theta'] = -call_greeks['theta']
        greeks['vega'] = -call_greeks['vega']
        
    elif strategy_type == "Bull Call Spread":
        greeks['delta'] = lower_call_greeks['delta'] - upper_call_greeks['delta']
        greeks['gamma'] = lower_call_greeks['gamma'] - upper_call_greeks['gamma']
        greeks['theta'] = lower_call_greeks['theta'] - upper_call_greeks['theta']
        greeks['vega'] = lower_call_greeks['vega'] - upper_call_greeks['vega']
        
    elif strategy_type == "Bear Call Spread":
        greeks['delta'] = -lower_call_greeks['delta'] + upper_call_greeks['delta']
        greeks['gamma'] = -lower_call_greeks['gamma'] + upper_call_greeks['gamma']
        greeks['theta'] = -lower_call_greeks['theta'] + upper_call_greeks['theta']
        greeks['vega'] = -lower_call_greeks['vega'] + upper_call_greeks['vega']
        
    elif strategy_type == "Long Put":
        greeks = put_greeks
        
    elif strategy_type == "Protective Put":
        greeks['delta'] = 1 + put_greeks['delta']
        greeks['gamma'] = put_greeks['gamma']
        greeks['theta'] = put_greeks['theta']
        greeks['vega'] = put_greeks['vega']
        
    elif strategy_type == "Bull Put Spread":
        # Short higher strike put, long lower strike put
        greeks['delta'] = -upper_put_greeks['delta'] + lower_put_greeks['delta']
        greeks['gamma'] = -upper_put_greeks['gamma'] + lower_put_greeks['gamma']
        greeks['theta'] = -upper_put_greeks['theta'] + lower_put_greeks['theta']
        greeks['vega'] = -upper_put_greeks['vega'] + lower_put_greeks['vega']
        
    elif strategy_type == "Bear Put Spread":
        # Long higher strike put, short lower strike put
        greeks['delta'] = upper_put_greeks['delta'] - lower_put_greeks['delta']
        greeks['gamma'] = upper_put_greeks['gamma'] - lower_put_greeks['gamma']
        greeks['theta'] = upper_put_greeks['theta'] - lower_put_greeks['theta']
        greeks['vega'] = upper_put_greeks['vega'] - lower_put_greeks['vega']
        
    elif strategy_type == "Long Straddle":
        greeks['delta'] = call_greeks['delta'] + put_greeks['delta']
        greeks['gamma'] = call_greeks['gamma'] + put_greeks['gamma']
        greeks['theta'] = call_greeks['theta'] + put_greeks['theta']
        greeks['vega'] = call_greeks['vega'] + put_greeks['vega']
        
    elif strategy_type == "Short Straddle":
        greeks['delta'] = -call_greeks['delta'] - put_greeks['delta']
        greeks['gamma'] = -call_greeks['gamma'] - put_greeks['gamma']
        greeks['theta'] = -call_greeks['theta'] - put_greeks['theta']
        greeks['vega'] = -call_greeks['vega'] - put_greeks['vega']
        
    elif strategy_type == "Iron Butterfly":
        greeks['delta'] = lower_put_greeks['delta'] - put_greeks['delta'] - call_greeks['delta'] + upper_call_greeks['delta']
        greeks['gamma'] = lower_put_greeks['gamma'] - put_greeks['gamma'] - call_greeks['gamma'] + upper_call_greeks['gamma']
        greeks['theta'] = lower_put_greeks['theta'] - put_greeks['theta'] - call_greeks['theta'] + upper_call_greeks['theta']
        greeks['vega'] = lower_put_greeks['vega'] - put_greeks['vega'] - call_greeks['vega'] + upper_call_greeks['vega']
        
    elif strategy_type == "Iron Condor":
        greeks['delta'] = very_lower_put_greeks['delta'] - lower_put_greeks['delta'] - upper_call_greeks['delta'] + very_upper_call_greeks['delta']
        greeks['gamma'] = very_lower_put_greeks['gamma'] - lower_put_greeks['gamma'] - upper_call_greeks['gamma'] + very_upper_call_greeks['gamma']
        greeks['theta'] = very_lower_put_greeks['theta'] - lower_put_greeks['theta'] - upper_call_greeks['theta'] + very_upper_call_greeks['theta']
        greeks['vega'] = very_lower_put_greeks['vega'] - lower_put_greeks['vega'] - upper_call_greeks['vega'] + very_upper_call_greeks['vega']
    
    return greeks

def simulate_garch_volatility(initial_vol, n_steps, n_simulations, alpha=0.1, beta=0.8, omega=0.000001):
    """
    Simulate volatility paths using a GARCH(1,1) model
    
    Parameters:
    -----------
    initial_vol : float
        Initial volatility
    n_steps : int
        Number of time steps
    n_simulations : int
        Number of simulation paths
    alpha, beta, omega : float
        GARCH parameters
        
    Returns:
    --------
    numpy.ndarray: Matrix of volatility paths
    """
    # Initialize volatility array
    vol = np.zeros((n_simulations, n_steps+1))
    vol[:, 0] = initial_vol**2  # Variance
    
    # Simulate GARCH process
    for t in range(1, n_steps+1):
        # Generate random shocks
        z = np.random.standard_normal(n_simulations)
        
        # Calculate next period's variance using GARCH(1,1) formula
        # σ²(t) = ω + α * ε²(t-1) + β * σ²(t-1)
        # where ε(t-1) = σ(t-1) * z(t-1)
        epsilon_squared = vol[:, t-1] * z**2
        vol[:, t] = omega + alpha * epsilon_squared + beta * vol[:, t-1]
    
    # Convert variance to volatility
    return np.sqrt(vol)

def var_calculator(strategies, quantities, spot_price, strikes, maturities, rates, vols,
                 confidence=0.95, horizon=1/252, n_simulations=10000,
                 use_t_dist=True, degrees_of_freedom=5, use_garch=False):
    """
    Calculate Value-at-Risk for an options portfolio using enhanced Monte Carlo simulation
    with Student's t-distribution and optional GARCH volatility modeling
    
    Parameters:
    -----------
    strategies : list
        List of strategy types
    quantities : list
        Number of positions for each strategy
    spot_price, strikes, maturities, rates, vols : lists
        Parameters for each position
    confidence : float
        Confidence level (default: 95%)
    horizon : float
        Risk horizon in years (default: 1 day)
    n_simulations : int
        Number of Monte Carlo simulations
    use_t_dist : bool
        Whether to use Student's t-distribution (True) or normal distribution (False)
    degrees_of_freedom : int
        Degrees of freedom for t-distribution (typically 3-10, lower = fatter tails)
    use_garch : bool
        Whether to use GARCH volatility modeling
    
    Returns:
    --------
    dict: VaR results and risk metrics
    """
    # Validate inputs
    if len(strategies) != len(quantities) or len(quantities) != len(strikes):
        raise ValueError("Input arrays must have the same length")
    
    n_positions = len(strategies)
    
    # Convert lists to arrays
    strikes = np.array(strikes)
    maturities = np.array(maturities)
    rates = np.array(rates) if isinstance(rates, (list, np.ndarray)) else np.ones(n_positions) * rates
    vols = np.array(vols) if isinstance(vols, (list, np.ndarray)) else np.ones(n_positions) * vols
    quantities = np.array(quantities)
    
    # Estimate portfolio volatility
    annual_vol = np.sqrt(np.mean(vols**2))  # Portfolio volatility estimate
    
    if use_garch:
        # Generate price paths with GARCH volatility
        n_steps = max(int(horizon * 252) + 1, 5)  # Ensure minimum steps for short horizons
        vol_paths = simulate_garch_volatility(annual_vol, n_steps, n_simulations)
        
        # Generate price paths with time-varying volatility
        price_paths = np.zeros((n_simulations, n_steps))
        price_paths[:, 0] = spot_price
        
        for i in range(n_simulations):
            for t in range(1, n_steps):
                # Daily drift and vol
                daily_horizon = 1/252
                drift = (rates.mean() - 0.5 * vol_paths[i, t-1]**2) * daily_horizon
                
                # Use t-distribution or normal distribution based on user choice
                if use_t_dist:
                    random_shock = np.random.standard_t(df=degrees_of_freedom)
                    # Scale to match volatility
                    diffusion = vol_paths[i, t-1] * np.sqrt(daily_horizon) * random_shock * np.sqrt((degrees_of_freedom-2)/degrees_of_freedom)
                else:
                    diffusion = vol_paths[i, t-1] * np.sqrt(daily_horizon) * np.random.normal()
                    
                price_paths[i, t] = price_paths[i, t-1] * np.exp(drift + diffusion)
        
        # Use final prices for VaR calculation
        simulated_prices = price_paths[:, -1]
    else:
        # Generate random price paths using t-distribution or normal distribution
        np.random.seed(42)  # For reproducibility
        
        if use_t_dist:
            # Student's t-distribution for fat tails
            # Generate t-distributed random variables
            t_random = np.random.standard_t(df=degrees_of_freedom, size=n_simulations)
            # Scale to match volatility and add drift
            price_changes = (rates.mean() - 0.5 * annual_vol**2) * horizon + \
                            annual_vol * np.sqrt(horizon) * t_random * np.sqrt((degrees_of_freedom-2)/degrees_of_freedom)
        else:
            # Regular normal distribution
            price_changes = np.random.normal(
                (rates.mean() - 0.5 * annual_vol**2) * horizon,
                annual_vol * np.sqrt(horizon),
                n_simulations
            )
        
        simulated_prices = spot_price * np.exp(price_changes)
    
    # Calculate current portfolio value
    current_portfolio_value = 0
    for i in range(n_positions):
        strategy = strategies[i]
        
        if strategy == "Protective Put":
            # For Protective Put: Current value = Stock value + Put value
            stock_value = spot_price
            put_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'put')
            position_value = stock_value + put_value
        elif strategy == "Covered Call Writing":
            # For Covered Call: Current value = Stock value - Call value
            stock_value = spot_price
            call_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'call')
            position_value = stock_value - call_value
        elif "Spread" in strategy or "Straddle" in strategy or "Iron" in strategy:
            # For spreads, straddles, and iron strategies, delegate to a specific calculator
            # This is a simplified approach - a real implementation would need separate logic for each
            # Just using a default option price for demonstration
            if "Call" in strategy:
                position_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'call')
            else:
                position_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'put')
        elif 'Call' in strategy:
            position_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'call')
        else:  # Put options
            position_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'put')
        
        current_portfolio_value += quantities[i] * position_value
    
    # Calculate simulated portfolio values
    simulated_portfolio_values = np.zeros(n_simulations)
    for j in range(n_simulations):
        sim_price = simulated_prices[j]
        portfolio_value = 0
        
        for i in range(n_positions):
            strategy = strategies[i]
            remaining_maturity = max(0, maturities[i] - horizon)
            
            if strategy == "Protective Put":
                # For Protective Put: Simulated value = Simulated stock value + Put value
                stock_value = sim_price
                put_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'put')
                position_value = stock_value + put_value
            elif strategy == "Covered Call Writing":
                # For Covered Call: Simulated value = Simulated stock value - Call value
                stock_value = sim_price
                call_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
                position_value = stock_value - call_value
            elif "Bull Call Spread" in strategy:
                # Long lower strike call, short higher strike call
                lower_strike = strikes[i] * 0.9  # This is a simplification
                upper_strike = strikes[i] * 1.1  # This is a simplification
                long_call = black_scholes_calc(sim_price, lower_strike, remaining_maturity, rates[i], vols[i], 'call')
                short_call = black_scholes_calc(sim_price, upper_strike, remaining_maturity, rates[i], vols[i], 'call')
                position_value = long_call - short_call
            elif "Bear Call Spread" in strategy:
                # Short lower strike call, long higher strike call
                lower_strike = strikes[i] * 0.9  # This is a simplification
                upper_strike = strikes[i] * 1.1  # This is a simplification
                short_call = black_scholes_calc(sim_price, lower_strike, remaining_maturity, rates[i], vols[i], 'call')
                long_call = black_scholes_calc(sim_price, upper_strike, remaining_maturity, rates[i], vols[i], 'call')
                position_value = short_call - long_call
            elif "Bull Put Spread" in strategy:
                # Short higher strike put, long lower strike put
                lower_strike = strikes[i] * 0.9  # This is a simplification
                upper_strike = strikes[i] * 1.1  # This is a simplification
                short_put = black_scholes_calc(sim_price, upper_strike, remaining_maturity, rates[i], vols[i], 'put')
                long_put = black_scholes_calc(sim_price, lower_strike, remaining_maturity, rates[i], vols[i], 'put')
                position_value = short_put - long_put
            elif "Bear Put Spread" in strategy:
                # Long higher strike put, short lower strike put
                lower_strike = strikes[i] * 0.9  # This is a simplification
                upper_strike = strikes[i] * 1.1  # This is a simplification
                long_put = black_scholes_calc(sim_price, upper_strike, remaining_maturity, rates[i], vols[i], 'put')
                short_put = black_scholes_calc(sim_price, lower_strike, remaining_maturity, rates[i], vols[i], 'put')
                position_value = long_put - short_put
            elif "Long Straddle" in strategy:
                # Long call and long put
                call_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
                put_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'put')
                position_value = call_value + put_value
            elif "Short Straddle" in strategy:
                # Short call and short put
                call_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
                put_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'put')
                position_value = -(call_value + put_value)
            elif "Iron Butterfly" in strategy or "Iron Condor" in strategy:
                # These are complex strategies with 4 legs
                # Simplified approach here - a real implementation would need specific logic
                position_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
            elif 'Call' in strategy:
                position_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
            else:  # Put options
                position_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'put')
            
            portfolio_value += quantities[i] * position_value
            
        simulated_portfolio_values[j] = portfolio_value
    
    # Calculate P&L
    pnl = simulated_portfolio_values - current_portfolio_value
    
    # Sort P&L from worst to best
    sorted_pnl = np.sort(pnl)
    
    # Calculate VaR
    var_index = int(n_simulations * (1 - confidence))
    var = -sorted_pnl[var_index]
    
    # Calculate Expected Shortfall (Conditional VaR)
    es = -np.mean(sorted_pnl[:var_index])
    
    # Calculate additional risk metrics
    volatility = np.std(pnl)
    skewness = np.mean((pnl - np.mean(pnl))**3) / (volatility**3)
    kurtosis = np.mean((pnl - np.mean(pnl))**4) / (volatility**4) - 3
    
    return {
        'VaR': var,
        'Expected_Shortfall': es,
        'Volatility': volatility,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Worst_Case': -sorted_pnl[0],
        'Best_Case': sorted_pnl[-1],
        'Confidence_Level': confidence,
        'Horizon_Days': horizon * 252,
        'Distribution': 'Student-t' if use_t_dist else 'Normal',
        'DF': degrees_of_freedom if use_t_dist else None,
        'Volatility_Model': 'GARCH' if use_garch else 'Constant'
    }

def stress_test_portfolio(strategies, quantities, spot_price, strikes, maturities, rates, vols, stress_scenarios=None):
    """Perform stress testing on an options portfolio using historical scenarios"""
    # Convert string to list if needed
    if isinstance(strategies, str):
        strategies = [strategies]
    
    # Convert to lists if single values passed
    if not isinstance(quantities, (list, np.ndarray)):
        quantities = [quantities]
    if not isinstance(strikes, (list, np.ndarray)):
        strikes = [strikes]
    if not isinstance(maturities, (list, np.ndarray)):
        maturities = [maturities]
    
    # Ensure all lists have same length
    n = len(strategies)
    if len(quantities) == 1 and n > 1:
        quantities = quantities * n
    if len(strikes) == 1 and n > 1:
        strikes = strikes * n
    if len(maturities) == 1 and n > 1:
        maturities = maturities * n
    
    # Convert rates and vols to lists if they are single values
    if not isinstance(rates, (list, np.ndarray)):
        rates = [rates] * n
    if not isinstance(vols, (list, np.ndarray)):
        vols = [vols] * n
    
    # Ensure rates and vols have same length if they're lists
    if len(rates) == 1 and n > 1:
        rates = rates * n
    if len(vols) == 1 and n > 1:
        vols = vols * n
        
    # Default stress scenarios
    if stress_scenarios is None:
        stress_scenarios = {
            "2008_Crisis": {"price_change": -0.50, "vol_multiplier": 3.0, "rate_change": -0.02},
            "Covid_Crash": {"price_change": -0.30, "vol_multiplier": 2.5, "rate_change": -0.01},
            "Tech_Bubble": {"price_change": -0.40, "vol_multiplier": 2.0, "rate_change": 0.01},
            "2022_Inflation": {"price_change": -0.20, "vol_multiplier": 1.5, "rate_change": 0.03}
        }
    
    # Calculate current portfolio value
    current_portfolio_value = 0
    for i in range(len(strategies)):
        if 'Call' in strategies[i]:
            option_price = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'call')
        else:
            option_price = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'put')
        current_portfolio_value += quantities[i] * option_price
    
    # Apply stress scenarios
    results = {}
    for scenario_name, scenario in stress_scenarios.items():
        stressed_portfolio_value = 0
        stressed_spot = spot_price * (1 + scenario["price_change"])
        
        for i in range(len(strategies)):
            stressed_vol = vols[i] * scenario["vol_multiplier"]
            stressed_rate = max(0.001, rates[i] + scenario["rate_change"])
            
            if 'Call' in strategies[i]:
                option_price = black_scholes_calc(stressed_spot, strikes[i], maturities[i],
                                                stressed_rate, stressed_vol, 'call')
            else:
                option_price = black_scholes_calc(stressed_spot, strikes[i], maturities[i],
                                                stressed_rate, stressed_vol, 'put')
                
            stressed_portfolio_value += quantities[i] * option_price
        
        # Calculate P&L and percent change
        pnl = stressed_portfolio_value - current_portfolio_value
        pct_change = (pnl / current_portfolio_value) * 100 if current_portfolio_value != 0 else 0
        
        results[scenario_name] = {
            "P&L": pnl,
            "Portfolio_Change_Pct": pct_change,
            "Stressed_Value": stressed_portfolio_value,
            "Applied_Scenario": {
                "Price Change": f"{scenario['price_change']*100:.1f}%",
                "Volatility Multiplier": f"{scenario['vol_multiplier']:.1f}x",
                "Rate Change": f"{scenario['rate_change']*100:+.1f}%"
            }
        }
    
    return results


# Calculate Strategy Performance
def calculate_strategy_performance(strategy_type, spot_price, strike_price, time_to_maturity,
                                 risk_free_rate, volatility, call_value, put_value):
    """
    Calculate comprehensive performance metrics for option strategies
    
    Parameters:
    -----------
    strategy_type : str
        Type of option strategy
    spot_price, strike_price, time_to_maturity, risk_free_rate, volatility : float
        Market parameters
    call_value, put_value : float
        Current option values
    
    Returns:
    --------
    dict: Performance metrics
    """
    # Calculate strategy Greeks
    greeks = calculate_strategy_greeks(
        strategy_type, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility
    )
    
    # Calculate advanced Greeks
    advanced_greeks = calculate_advanced_greeks("Call", spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    
    # Define spot price range for P&L calculation
    spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
    
    # Calculate P&L
    pnl = calculate_strategy_pnl(
        strategy_type, spot_range, spot_price, strike_price,
        time_to_maturity, risk_free_rate, volatility, call_value, put_value
    )
    
    # Find break-even points
    be_indices = np.where(np.diff(np.signbit(pnl)))[0]
    break_even_points = [spot_range[i] for i in be_indices] if len(be_indices) > 0 else []
    
    # Profit probability estimation using lognormal distribution
    if len(break_even_points) > 0:
        # Sort break-even points
        break_even_points.sort()
        
        # Calculate probability below lower BE and above upper BE
        if len(break_even_points) == 1:
            # One break-even point
            be = break_even_points[0]
            if pnl[0] > 0:  # Profitable below BE
                profit_prob = norm.cdf(np.log(be/spot_price) / (volatility * np.sqrt(time_to_maturity)))
            else:  # Profitable above BE
                profit_prob = 1 - norm.cdf(np.log(be/spot_price) / (volatility * np.sqrt(time_to_maturity)))
        else:
            # Multiple break-even points, assume first and last define profitable region
            lower_be = break_even_points[0]
            upper_be = break_even_points[-1]
            
            if pnl[0] > 0:  # Profitable outside the range
                profit_prob = norm.cdf(np.log(lower_be/spot_price) / (volatility * np.sqrt(time_to_maturity))) + \
                             (1 - norm.cdf(np.log(upper_be/spot_price) / (volatility * np.sqrt(time_to_maturity))))
            else:  # Profitable inside the range
                profit_prob = norm.cdf(np.log(upper_be/spot_price) / (volatility * np.sqrt(time_to_maturity))) - \
                             norm.cdf(np.log(lower_be/spot_price) / (volatility * np.sqrt(time_to_maturity)))
    else:
        # No break-even points, strategy is always profitable or always unprofitable
        profit_prob = 1.0 if np.mean(pnl) > 0 else 0.0
    
    # Calculate maximum profit and loss
    max_profit = max(pnl)
    max_loss = abs(min(pnl))
    
    # Calculate risk-reward ratio
    risk_reward = max_profit / max_loss if max_loss > 0 else float('inf')
    
    # Calculate time value decay
    time_decay_rate = greeks['theta']
    
    # Calculate Sharpe-like ratio (expected return / volatility)
    expected_pnl = np.mean(pnl)
    pnl_volatility = np.std(pnl)
    sharpe = expected_pnl / pnl_volatility if pnl_volatility > 0 else 0
    
    # Kelly criterion - optimal position size
    if max_loss > 0:
        win_prob = profit_prob
        loss_prob = 1 - win_prob
        avg_win = max_profit
        avg_loss = max_loss
        kelly = (win_prob * avg_win - loss_prob * avg_loss) / (avg_win * avg_loss) if avg_win * avg_loss > 0 else 0
        kelly = max(0, min(1, kelly))  # Bound between 0 and 1
    else:
        kelly = 1  # No risk of loss
    
    # Return performance metrics
    return {
        'profitability': {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'expected_pnl': expected_pnl,
            'break_even_points': break_even_points,
            'profit_probability': profit_prob,
            'risk_reward_ratio': risk_reward
        },
        'risk_metrics': {
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'vanna': advanced_greeks['vanna'],
            'volga': advanced_greeks['volga'],
            'pnl_volatility': pnl_volatility,
            'sharpe_ratio': sharpe,
            'kelly_criterion': kelly
        },
        'time_decay': {
            'daily_theta': greeks['theta'],
            'weekly_decay': greeks['theta'] * 5,
            'monthly_decay': greeks['theta'] * 21
        },
        'sensitivity': {
            'price_move_10pct_up': np.interp(spot_price * 1.1, spot_range, pnl) - np.interp(spot_price, spot_range, pnl),
            'price_move_10pct_down': np.interp(spot_price * 0.9, spot_range, pnl) - np.interp(spot_price, spot_range, pnl),
            'vol_move_up': calculate_strategy_pnl(strategy_type, [spot_price], spot_price, strike_price,
                                              time_to_maturity, risk_free_rate, volatility * 1.1, call_value, put_value)[0] -
                          calculate_strategy_pnl(strategy_type, [spot_price], spot_price, strike_price,
                                              time_to_maturity, risk_free_rate, volatility, call_value, put_value)[0]
        }
    }
# Strategy Visualization Function
def create_strategy_visualization(strategy, spot_range, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value):
    """
    Create visualization for option strategies with accurate payoffs and key levels
    
    Parameters:
    -----------
    strategy : str
        Strategy type
    spot_range : array
        Range of spot prices for calculation
    current_price, strike_price, time_to_maturity, risk_free_rate, volatility : float
        Market parameters
    call_value, put_value : float
        Current option values
    
    Returns:
    --------
    fig: matplotlib figure object
    """
    # Calculate P&L
    pnl = calculate_strategy_pnl(
        strategy, spot_range, current_price, strike_price,
        time_to_maturity, risk_free_rate, volatility, call_value, put_value
    )
    
    # Define common strikes for reference
    lower_strike = strike_price * 0.9
    upper_strike = strike_price * 1.1
    
    # Create figure with improved styling
    plt.style.use('dark_background')
    fig_pnl = plt.figure(figsize=(12, 6))
    
    # Plot P&L profile with better visibility
    plt.plot(spot_range, pnl, 'g-', linewidth=2.5, label='P&L Profile')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.6, label='Break-even Line')
    
    # Add reference lines at current price and strike price(s)
    plt.axvline(x=current_price, color='cyan', linestyle=':', alpha=0.5, label='Current Price')
    plt.axvline(x=strike_price, color='magenta', linestyle=':', alpha=0.5, label='Strike Price')
    
    # For spread strategies, add reference lines for upper/lower strikes
    if 'Spread' in strategy or 'Iron' in strategy:
        if 'Call Spread' in strategy or 'Iron' in strategy:
            plt.axvline(x=upper_strike, color='yellow', linestyle=':', alpha=0.3, label='Upper Strike')
        if 'Put Spread' in strategy or 'Iron' in strategy:
            plt.axvline(x=lower_strike, color='orange', linestyle=':', alpha=0.3, label='Lower Strike')
    
    # Find and mark break-even points
    break_even_indices = np.where(np.diff(np.signbit(pnl)))[0]
    break_even_points = [spot_range[i] for i in break_even_indices]
    
    # Mark key profit/loss points
    max_profit_idx = np.argmax(pnl)
    max_loss_idx = np.argmin(pnl)
    
    # Add annotations
    plt.annotate(f'Max Profit: ${pnl[max_profit_idx]:.2f}',
                xy=(spot_range[max_profit_idx], pnl[max_profit_idx]),
                xytext=(10, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))
    
    plt.annotate(f'Max Loss: ${pnl[max_loss_idx]:.2f}',
                xy=(spot_range[max_loss_idx], pnl[max_loss_idx]),
                xytext=(10, -15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))
    
    # Mark break-even points
    for point in break_even_points:
        plt.axvline(x=point, color='white', linestyle='--', alpha=0.5)
        plt.text(point, plt.ylim()[0] * 0.9, f'BE: {point:.2f}', rotation=90,
               verticalalignment='bottom', color='white', fontweight='bold')
    
    # Improve chart aesthetics
    plt.grid(True, alpha=0.3)
    plt.xlabel('Stock Price ($)', fontsize=12)
    plt.ylabel('Profit/Loss ($)', fontsize=12)
    plt.title(f'{strategy} P&L Profile', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    
    # Add strategy description based on mathematical formulas
    if strategy == "Covered Call Writing":
        description = "Long stock + Short call\nMax Profit: (K - S₀) + C when S_T > K\nMax Loss: S₀ - C (if S_T → 0)"
    elif strategy == "Long Call":
        description = "Max Profit: Unlimited as S_T → ∞\nMax Loss: Premium paid (C)"
    elif strategy == "Bull Call Spread":
        description = "Long lower strike call + Short higher strike call\nMax Profit: (K₂ - K₁) - Net Debit\nMax Loss: Net debit paid"
    elif strategy == "Bear Call Spread":
        description = "Short lower strike call + Long higher strike call\nMax Profit: Net credit received\nMax Loss: (K₂ - K₁) - Net Credit"
    elif strategy == "Long Put":
        description = "Max Profit: K - P (if S_T → 0)\nMax Loss: Premium paid (P)"
    elif strategy == "Protective Put":
        description = "Long stock + Long put\nMax Profit: Unlimited as S_T → ∞\nMax Loss: S₀ + P - K"
    elif strategy == "Long Straddle":
        description = "Long call + Long put (same strike)\nMax Profit: Unlimited\nMax Loss: C + P"
    elif strategy == "Short Straddle":
        description = "Short call + Short put (same strike)\nMax Profit: C + P\nMax Loss: Unlimited"
    elif strategy == "Iron Butterfly":
        description = "Short ATM put + Short ATM call + Long OTM put + Long OTM call\nMax Profit: Net credit\nMax Loss: Δ - Net Credit"
    elif strategy == "Iron Condor":
        description = "Bull put spread + Bear call spread\nMax Profit: Net credit\nMax Loss: Δ - Net Credit"
    else:
        description = ""
    
    # Add description text box
    plt.figtext(0.02, 0.02, description, fontsize=10, bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))
    
    return fig_pnl

# Local Volatility Surface Calculation
def enhanced_local_volatility_surface(strikes, maturities, implied_vols, spot, rates,
                               dividend_yield=0.0, smoothing_level=0.1):
    """
    Generate enhanced local volatility surface using Dupire's formula with proper mathematical 
    formulation and numerical stability improvements
    
    Parameters:
    -----------
    strikes : array-like
        Array of strike prices
    maturities : array-like
        Array of maturities (in years)
    implied_vols : 2D array-like
        Matrix of implied volatilities for each strike/maturity pair
    spot : float
        Current spot price
    rates : array-like or float
        Risk-free rates for each maturity or a single rate
    dividend_yield : float or array-like
        Continuous dividend yield(s)
    smoothing_level : float
        Controls smoothing of input data (0.0-1.0)
        
    Returns:
    --------
    tuple: (local_vol_surface, smoothed_implied_vols, diagnostics)
        - local_vol_surface: Matrix of local volatilities
        - smoothed_implied_vols: Matrix of smoothed implied volatilities
        - diagnostics: Dictionary with diagnostic information
    """
    import numpy as np
    from scipy.interpolate import RectBivariateSpline, interp1d
    
    # Ensure inputs are numpy arrays
    strikes = np.array(strikes, dtype=float)
    maturities = np.array(maturities, dtype=float)
    implied_vols = np.array(implied_vols, dtype=float)
    
    # Convert rates and dividend yield to arrays if they're scalars
    if np.isscalar(rates):
        rates = np.full_like(maturities, rates)
    if np.isscalar(dividend_yield):
        dividend_yield = np.full_like(maturities, dividend_yield)
    
    # Diagnostics container
    diagnostics = {
        'boundary_points': 0,
        'denominator_fixes': 0,
        'extreme_values_clipped': 0
    }
    
    # 1. Smooth the implied volatility surface to reduce noise in derivatives
    # Adjust smoothing factor based on user input
    smoothing_factor = smoothing_level * (len(strikes) * len(maturities))
    
    # Create spline with appropriate smoothing
    iv_spline = RectBivariateSpline(
        maturities, strikes, implied_vols,
        kx=min(3, len(maturities)-1),
        ky=min(3, len(strikes)-1),
        s=smoothing_factor
    )
    
    # Create a denser grid for more accurate derivatives
    dense_maturities = np.linspace(maturities.min(), maturities.max(), max(50, len(maturities)*2))
    dense_strikes = np.linspace(strikes.min(), strikes.max(), max(50, len(strikes)*2))
    
    # Evaluate smoothed implied volatility on the dense grid
    K_dense, T_dense = np.meshgrid(dense_strikes, dense_maturities)
    smoothed_iv = iv_spline(dense_maturities, dense_strikes)
    
    # 2. Interpolate rates and dividends to match the dense grid
    rate_interp = interp1d(maturities, rates, kind='linear', fill_value='extrapolate')
    div_interp = interp1d(maturities, dividend_yield, kind='linear', fill_value='extrapolate')
    
    dense_rates = rate_interp(dense_maturities)
    dense_dividends = div_interp(dense_maturities)
    
    # 3. Calculate option prices from implied volatilities
    # This is critical - Dupire's formula applies to option prices, not volatilities directly
    option_prices = np.zeros_like(smoothed_iv)
    for i, t in enumerate(dense_maturities):
        for j, k in enumerate(dense_strikes):
            # Use Black-Scholes to get option prices from implied vols
            option_prices[i, j] = black_scholes_calc(
                spot, k, t, dense_rates[i] - dense_dividends[i], smoothed_iv[i, j], 'call'
            )
    
    # 4. Calculate derivatives with proper boundary handling
    # Initialize derivative arrays
    dC_dT = np.zeros_like(option_prices)
    dC_dK = np.zeros_like(option_prices)
    d2C_dK2 = np.zeros_like(option_prices)
    
    # Time steps for finite difference calculations
    dt_forward = np.diff(dense_maturities)
    dt_backward = np.diff(dense_maturities, prepend=dense_maturities[0])
    
    # Strike steps
    dk_forward = np.diff(dense_strikes)
    dk_backward = np.diff(dense_strikes, prepend=dense_strikes[0])
    
    # Calculate time derivatives (dC/dT)
    for j in range(len(dense_strikes)):
        # Forward difference for first point
        dC_dT[0, j] = (option_prices[1, j] - option_prices[0, j]) / dt_forward[0]
        
        # Central difference for interior points
        for i in range(1, len(dense_maturities)-1):
            dt_central = dense_maturities[i+1] - dense_maturities[i-1]
            dC_dT[i, j] = (option_prices[i+1, j] - option_prices[i-1, j]) / dt_central
        
        # Backward difference for last point
        last_idx = len(dense_maturities)-1
        dC_dT[last_idx, j] = (option_prices[last_idx, j] - option_prices[last_idx-1, j]) / dt_backward[last_idx]
    
    # Calculate strike derivatives (dC/dK and d2C/dK2)
    for i in range(len(dense_maturities)):
        # Forward differences for first point
        dC_dK[i, 0] = (option_prices[i, 1] - option_prices[i, 0]) / dk_forward[0]
        d2C_dK2[i, 0] = (option_prices[i, 2] - 2*option_prices[i, 1] + option_prices[i, 0]) / (dk_forward[0]**2)
        diagnostics['boundary_points'] += 1
        
        # Central differences for interior points
        for j in range(1, len(dense_strikes)-1):
            dk = dense_strikes[j+1] - dense_strikes[j-1]
            dC_dK[i, j] = (option_prices[i, j+1] - option_prices[i, j-1]) / dk
            d2C_dK2[i, j] = (option_prices[i, j+1] - 2*option_prices[i, j] + option_prices[i, j-1]) / ((dense_strikes[j+1] - dense_strikes[j-1])/2)**2
        
        # Backward differences for last point
        last_idx = len(dense_strikes)-1
        dC_dK[i, last_idx] = (option_prices[i, last_idx] - option_prices[i, last_idx-1]) / dk_backward[last_idx]
        d2C_dK2[i, last_idx] = (option_prices[i, last_idx] - 2*option_prices[i, last_idx-1] + option_prices[i, last_idx-2]) / (dk_backward[last_idx]**2)
        diagnostics['boundary_points'] += 1
    
    # 5. Apply Dupire's formula with robust numerical safeguards
    local_vol = np.zeros_like(option_prices)
    
    for i in range(len(dense_maturities)):
        for j in range(len(dense_strikes)):
            # Extract parameters at this grid point
            r = dense_rates[i]
            q = dense_dividends[i]
            K = dense_strikes[j]
            T = dense_maturities[i]
            
            # Proper Dupire formula
            # numerator = dC/dT + (r-q)*K*dC/dK + q*C
            numerator = dC_dT[i, j] + (r - q) * K * dC_dK[i, j] + q * option_prices[i, j]
            
            # denominator = 0.5 * K^2 * d2C/dK2
            denominator = 0.5 * K**2 * d2C_dK2[i, j]
            
            # Apply robust numerical safeguards
            if denominator > 1e-6 and numerator > 0:
                local_vol[i, j] = np.sqrt(numerator / denominator)
            else:
                # Fallback to implied vol if formula gives invalid result
                local_vol[i, j] = smoothed_iv[i, j]
                diagnostics['denominator_fixes'] += 1
            
            # Apply reasonable bounds for stability
            if not (0.01 <= local_vol[i, j] <= 2.0):
                local_vol[i, j] = np.clip(local_vol[i, j], 0.01, 2.0)
                diagnostics['extreme_values_clipped'] += 1
    
    # 6. Smooth the output local volatility surface to remove artifacts
    output_smoothing = smoothing_factor * 2  # More aggressive smoothing for output
    local_vol_spline = RectBivariateSpline(
        dense_maturities, dense_strikes, local_vol,
        kx=min(3, len(dense_maturities)-1),
        ky=min(3, len(dense_strikes)-1),
        s=output_smoothing
    )
    
    # 7. Interpolate back to the original grid points for consistency
    result_local_vol = local_vol_spline(maturities, strikes)
    result_implied_vol = iv_spline(maturities, strikes)
    
    # Add grid information to diagnostics
    diagnostics['grid_info'] = {
        'original_size': (len(maturities), len(strikes)),
        'dense_size': (len(dense_maturities), len(dense_strikes)),
        'smoothing_factor': smoothing_factor
    }
    
    return result_local_vol, result_implied_vol, diagnostics

def analyze_vol_surface(local_vols, implied_vols, strikes, maturities, spot_price):
    """
    Analyze local volatility surface to extract key quantitative features
    
    Parameters:
    -----------
    local_vols : 2D array
        Local volatility surface
    implied_vols : 2D array
        Implied volatility surface
    strikes : array
        Strike prices
    maturities : array
        Maturity dates
    spot_price : float
        Current spot price
        
    Returns:
    --------
    dict: Analysis results
    """
    import numpy as np
    
    # Calculate moneyness for better comparisons
    moneyness = np.array([k/spot_price for k in strikes])
    
    # Initialize results
    results = {}
    
    # 1. Extract ATM volatility term structure
    atm_index = np.argmin(np.abs(moneyness - 1.0))
    results['atm_local_vol_term'] = local_vols[:, atm_index]
    results['atm_implied_vol_term'] = implied_vols[:, atm_index]
    
    # 2. Analyze volatility skew at various maturities
    skew_results = []
    
    # Select representative maturities for skew analysis
    maturity_indices = []
    if len(maturities) >= 3:
        # Short, medium and long term
        maturity_indices = [0, len(maturities)//2, len(maturities)-1]
    else:
        maturity_indices = list(range(len(maturities)))
    
    for idx in maturity_indices:
        # Calculate skew as slope of volatility curve near ATM
        if atm_index > 0 and atm_index < len(strikes) - 1:
            local_skew = (local_vols[idx, atm_index+1] - local_vols[idx, atm_index-1]) / (moneyness[atm_index+1] - moneyness[atm_index-1])
            implied_skew = (implied_vols[idx, atm_index+1] - implied_vols[idx, atm_index-1]) / (moneyness[atm_index+1] - moneyness[atm_index-1])
        else:
            # Handle boundary case
            if atm_index == 0:
                local_skew = (local_vols[idx, 1] - local_vols[idx, 0]) / (moneyness[1] - moneyness[0])
                implied_skew = (implied_vols[idx, 1] - implied_vols[idx, 0]) / (moneyness[1] - moneyness[0])
            else:
                local_skew = (local_vols[idx, -1] - local_vols[idx, -2]) / (moneyness[-1] - moneyness[-2])
                implied_skew = (implied_vols[idx, -1] - implied_vols[idx, -2]) / (moneyness[-1] - moneyness[-2])
        
        skew_results.append({
            'maturity': maturities[idx],
            'local_skew': local_skew,
            'implied_skew': implied_skew
        })
    
    results['skew_analysis'] = skew_results
    
    # 3. Volatility surface curvature (smile strength)
    # Calculate for a middle-term maturity
    mid_idx = len(maturities) // 2
    
    # Find OTM put, ATM, and OTM call volatilities
    otm_put_idx = max(0, np.argmin(np.abs(moneyness - 0.9)))
    otm_call_idx = min(len(moneyness)-1, np.argmin(np.abs(moneyness - 1.1)))
    
    # Calculate smile strength as average of deviations from ATM vol
    local_smile = 0.5 * ((local_vols[mid_idx, otm_put_idx] - local_vols[mid_idx, atm_index]) +
                         (local_vols[mid_idx, otm_call_idx] - local_vols[mid_idx, atm_index]))
    
    implied_smile = 0.5 * ((implied_vols[mid_idx, otm_put_idx] - implied_vols[mid_idx, atm_index]) +
                           (implied_vols[mid_idx, otm_call_idx] - implied_vols[mid_idx, atm_index]))
    
    results['smile_strength'] = {
        'local_vol_smile': local_smile,
        'implied_vol_smile': implied_smile
    }
    
    # 4. Implied vs local volatility differences
    results['vol_comparison'] = {
        'mean_diff': np.mean(local_vols - implied_vols),
        'max_diff': np.max(local_vols - implied_vols),
        'min_diff': np.min(local_vols - implied_vols),
        'rms_diff': np.sqrt(np.mean((local_vols - implied_vols)**2))
    }
    
    # 5. Surface stability assessment - higher values indicate potential numerical issues
    local_vol_gradients = np.gradient(local_vols)
    smoothness_metric = np.mean(np.abs(np.gradient(local_vol_gradients[0])) + np.abs(np.gradient(local_vol_gradients[1])))
    
    results['surface_stability'] = {
        'smoothness_metric': smoothness_metric,
        'stability_assessment': 'Good' if smoothness_metric < 0.05 else 'Moderate' if smoothness_metric < 0.1 else 'Poor'
    }
    
    return results

# Risk Scenario Analysis
def enhanced_risk_scenario_analysis(strategy, current_price, strike_price, time_to_maturity,
                                  risk_free_rate, current_vol, pnl_function, call_value=0, put_value=0):
    """
    Perform comprehensive stress testing and scenario analysis for an option strategy
    
    Parameters:
    -----------
    strategy : str
        Strategy type
    current_price, strike_price, time_to_maturity, risk_free_rate, current_vol : float
        Current market parameters
    pnl_function : function
        Function to calculate strategy P&L
    call_value, put_value : float
        Current option values (used for accurate P&L calculation)
    
    Returns:
    --------
    dict: Scenario analysis results with comprehensive metrics
    """
    # Enhanced scenario definitions
    price_scenarios = np.array([0.70, 0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20, 1.30]) * current_price
    vol_scenarios = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]) * current_vol
    time_scenarios = np.array([0, 1/252, 5/252, 10/252, 21/252, 63/252, 126/252]) # 0, 1, 5, 10, 21, 63, 126 days
    rate_scenarios = np.array([-0.01, -0.005, 0, 0.005, 0.01]) + risk_free_rate # Rate shifts
    
    # Initialize expanded results container
    results = {
        'price_impact': {},
        'vol_impact': {},
        'time_decay': {},
        'rate_impact': {},
        'combined_scenarios': {},
        'extreme_scenarios': {}
    }
    
    # Get current strategy value as baseline
    baseline_pnl = pnl_function(strategy, np.array([current_price]), current_price, strike_price,
                               time_to_maturity, risk_free_rate, current_vol, call_value, put_value)[0]
    
    # Calculate P&L across price scenarios (keeping other factors constant)
    price_pnl = []
    price_pnl_change = []
    for price in price_scenarios:
        spot_range = np.array([price])
        pnl = pnl_function(strategy, spot_range, current_price, strike_price,
                          time_to_maturity, risk_free_rate, current_vol, call_value, put_value)[0]
        price_pnl.append(pnl)
        price_pnl_change.append(pnl - baseline_pnl)  # Change from current P&L
    
    # Add price delta - how much P&L changes for each 1% move in price
    price_delta = {}
    for i, price in enumerate(price_scenarios):
        if i > 0:  # Skip first entry to calculate change
            price_percent_change = (price - price_scenarios[i-1]) / price_scenarios[i-1] * 100
            pnl_change = price_pnl[i] - price_pnl[i-1]
            price_delta[f"{price_scenarios[i-1]:.2f} to {price:.2f}"] = pnl_change / price_percent_change
    
    results['price_impact'] = {
        'scenarios': price_scenarios,
        'absolute_pnl': price_pnl,
        'pnl_change': price_pnl_change,
        'max_loss': min(price_pnl),
        'max_gain': max(price_pnl),
        'price_delta': price_delta  # P&L change per 1% price move
    }
    
    # Calculate P&L across volatility scenarios
    vol_pnl = []
    vol_pnl_change = []
    for vol in vol_scenarios:
        spot_range = np.array([current_price])
        pnl = pnl_function(strategy, spot_range, current_price, strike_price,
                          time_to_maturity, risk_free_rate, vol, call_value, put_value)[0]
        vol_pnl.append(pnl)
        vol_pnl_change.append(pnl - baseline_pnl)
    
    # Add vol delta - how much P&L changes for each 1 point move in vol
    vol_delta = {}
    for i, vol in enumerate(vol_scenarios):
        if i > 0:  # Skip first entry to calculate change
            vol_point_change = (vol - vol_scenarios[i-1]) * 100  # Convert to vol points (percentage points)
            pnl_change = vol_pnl[i] - vol_pnl[i-1]
            vol_delta[f"{vol_scenarios[i-1]*100:.1f}% to {vol*100:.1f}%"] = pnl_change / vol_point_change
    
    results['vol_impact'] = {
        'scenarios': vol_scenarios,
        'absolute_pnl': vol_pnl,
        'pnl_change': vol_pnl_change,
        'max_loss': min(vol_pnl),
        'max_gain': max(vol_pnl),
        'vol_delta': vol_delta  # P&L change per 1 percentage point vol move
    }
    
    # Calculate time decay impact
    time_pnl = []
    time_pnl_change = []
    for time_left in time_scenarios:
        # Handle expired option case
        if time_left <= 0:
            # For zero time remaining, calculate expiration value based on strategy type
            if strategy == "Long Call":
                pnl = max(0, current_price - strike_price) - call_value
            elif strategy == "Long Put":
                pnl = max(0, strike_price - current_price) - put_value
            elif strategy == "Covered Call Writing":
                pnl = current_price - current_price + max(0, -1 * (current_price - strike_price)) + call_value
            elif strategy == "Protective Put":
                pnl = current_price - current_price + max(0, strike_price - current_price) - put_value
            elif "Bull Call Spread" in strategy:
                # For a bull call spread, approximate using lower and upper strikes
                lower_strike = strike_price * 0.9
                upper_strike = strike_price * 1.1
                pnl = max(0, min(current_price - lower_strike, upper_strike - lower_strike)) - call_value
            elif "Bear Put Spread" in strategy:
                # For a bear put spread, approximate using lower and upper strikes
                lower_strike = strike_price * 0.9
                upper_strike = strike_price * 1.1
                pnl = max(0, min(upper_strike - current_price, upper_strike - lower_strike)) - put_value
            else:
                # For other strategies, use the PnL function with tiny time
                spot_range = np.array([current_price])
                pnl = pnl_function(strategy, spot_range, current_price, strike_price,
                                  0.00001, risk_free_rate, current_vol, call_value, put_value)[0]
        else:
            spot_range = np.array([current_price])
            pnl = pnl_function(strategy, spot_range, current_price, strike_price,
                              time_left, risk_free_rate, current_vol, call_value, put_value)[0]
        
        time_pnl.append(pnl)
        time_pnl_change.append(pnl - baseline_pnl)
    
    # Calculate daily decay rate for different periods
    daily_decay_rates = {}
    for i in range(1, len(time_scenarios)-1):
        days_diff = (time_scenarios[i+1] - time_scenarios[i]) * 252  # Convert to days
        if days_diff > 0:
            daily_rate = (time_pnl[i+1] - time_pnl[i]) / days_diff
            period_name = f"{int(time_scenarios[i]*252)} to {int(time_scenarios[i+1]*252)} days"
            daily_decay_rates[period_name] = daily_rate
    
    results['time_decay'] = {
        'scenarios': time_scenarios * 252,  # Convert to days for readability
        'absolute_pnl': time_pnl,
        'pnl_change': time_pnl_change,
        'daily_decay_rates': daily_decay_rates,
        'one_day_effect': (time_pnl[1] - time_pnl[0]) if len(time_pnl) > 1 else 0,
        'one_week_effect': (time_pnl[3] - time_pnl[0]) if len(time_pnl) > 3 else 0,
        'one_month_effect': (time_pnl[4] - time_pnl[0]) if len(time_pnl) > 4 else 0
    }
    
    # Calculate interest rate impact
    rate_pnl = []
    rate_pnl_change = []
    for rate in rate_scenarios:
        spot_range = np.array([current_price])
        pnl = pnl_function(strategy, spot_range, current_price, strike_price,
                          time_to_maturity, rate, current_vol, call_value, put_value)[0]
        rate_pnl.append(pnl)
        rate_pnl_change.append(pnl - baseline_pnl)
    
    results['rate_impact'] = {
        'scenarios': rate_scenarios * 100,  # Convert to percentage for readability
        'absolute_pnl': rate_pnl,
        'pnl_change': rate_pnl_change,
        'max_loss': min(rate_pnl),
        'max_gain': max(rate_pnl)
    }
    
    # Combined scenarios - major market events
    combined_scenarios = {
        'current': baseline_pnl,
        'market_crash_high_vol': pnl_function(strategy, np.array([current_price * 0.8]), current_price,
                                           strike_price, time_to_maturity, risk_free_rate - 0.005, current_vol * 1.8,
                                           call_value, put_value)[0],
        'market_rally_low_vol': pnl_function(strategy, np.array([current_price * 1.15]), current_price,
                                          strike_price, time_to_maturity, risk_free_rate + 0.005, current_vol * 0.7,
                                          call_value, put_value)[0],
        'flat_price_high_vol': pnl_function(strategy, np.array([current_price]), current_price,
                                          strike_price, time_to_maturity, risk_free_rate, current_vol * 1.5,
                                          call_value, put_value)[0],
        'flat_price_low_vol': pnl_function(strategy, np.array([current_price]), current_price,
                                         strike_price, time_to_maturity, risk_free_rate, current_vol * 0.6,
                                         call_value, put_value)[0]
    }
    
    # Calculate changes from baseline for combined scenarios
    combined_scenario_changes = {k: v - baseline_pnl for k, v in combined_scenarios.items() if k != 'current'}
    
    results['combined_scenarios'] = {
        'absolute_pnl': combined_scenarios,
        'pnl_change': combined_scenario_changes
    }
    
    # Extreme stress scenarios - historical crisis patterns
    extreme_scenarios = {
        'market_crash_2008': pnl_function(strategy, np.array([current_price * 0.65]), current_price,
                                        strike_price, time_to_maturity, risk_free_rate - 0.01, current_vol * 3.0,
                                        call_value, put_value)[0],
        'flash_crash': pnl_function(strategy, np.array([current_price * 0.9]), current_price,
                                   strike_price, time_to_maturity, risk_free_rate, current_vol * 2.5,
                                   call_value, put_value)[0],
        'covid_crash': pnl_function(strategy, np.array([current_price * 0.7]), current_price,
                                  strike_price, time_to_maturity, risk_free_rate - 0.005, current_vol * 2.0,
                                  call_value, put_value)[0],
        'tech_bubble_burst': pnl_function(strategy, np.array([current_price * 0.75]), current_price,
                                        strike_price, time_to_maturity, risk_free_rate - 0.005, current_vol * 1.7,
                                        call_value, put_value)[0],
        'black_monday': pnl_function(strategy, np.array([current_price * 0.6]), current_price,
                                    strike_price, time_to_maturity, risk_free_rate, current_vol * 3.5,
                                    call_value, put_value)[0]
    }
    
    # Calculate changes from baseline for extreme scenarios
    extreme_scenario_changes = {k: v - baseline_pnl for k, v in extreme_scenarios.items()}
    
    results['extreme_scenarios'] = {
        'absolute_pnl': extreme_scenarios,
        'pnl_change': extreme_scenario_changes
    }
    
    # Add strategy risk profile summary
    price_sensitivity = np.mean([abs(d) for d in price_delta.values()]) if price_delta else 0
    vol_sensitivity = np.mean([abs(d) for d in vol_delta.values()]) if vol_delta else 0
    time_sensitivity = abs(results['time_decay']['one_day_effect']) if 'one_day_effect' in results['time_decay'] else 0
    
    results['risk_profile'] = {
        'price_sensitivity': price_sensitivity,  # Average P&L change per 1% price move
        'vol_sensitivity': vol_sensitivity,     # Average P&L change per 1% vol move
        'time_sensitivity': time_sensitivity,   # P&L change for 1 day passing
        'current_pnl': baseline_pnl,
        'strategy': strategy
    }
    
    # Calculate risk-reward metrics
    max_pnl_all = max([
        max(price_pnl) if price_pnl else -np.inf,
        max(vol_pnl) if vol_pnl else -np.inf,
        max(time_pnl) if time_pnl else -np.inf,
        max(rate_pnl) if rate_pnl else -np.inf
    ])
    
    min_pnl_all = min([
        min(price_pnl) if price_pnl else np.inf,
        min(vol_pnl) if vol_pnl else np.inf,
        min(time_pnl) if time_pnl else np.inf,
        min(rate_pnl) if rate_pnl else np.inf
    ])
    
    # Worst-case from extreme scenarios
    extreme_min = min(extreme_scenarios.values()) if extreme_scenarios else np.inf
    
    results['risk_metrics'] = {
        'max_potential_gain': max_pnl_all - baseline_pnl,
        'max_potential_loss': baseline_pnl - min_pnl_all,
        'worst_case_loss': baseline_pnl - extreme_min if extreme_min != np.inf else 0,
        'risk_reward_ratio': (max_pnl_all - baseline_pnl) / (baseline_pnl - min_pnl_all) if (baseline_pnl - min_pnl_all) > 0 else 0
    }
    
    return results
    results['extreme_scenarios'] = extreme_scenarios
    
    return results

# Display functions for the UI
def display_option_prices(price_info):
    """Display option prices in a clean format"""
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
    """Display Greeks in a minimal grid layout"""
    st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
            <h4 style="color: white; margin-bottom: 1rem;">Position Greeks</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;">
                <div class="greek-card">
                    <div class="greek-label">Delta</div>
                    <div class="greek-value">{round(calculated_greeks['delta'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Gamma</div>
                    <div class="greek-value">{round(calculated_greeks['gamma'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Theta</div>
                    <div class="greek-value">{round(calculated_greeks['theta'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Vega</div>
                    <div class="greek-value">{round(calculated_greeks['vega'], 3)}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Setup sidebar inputs
def setup_sidebar():
    """Setup sidebar inputs and controls with quantitative research options"""
    st.sidebar.markdown("## Model Selection")
    model_type = st.sidebar.selectbox(
        "Model Type",
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
    
    # Advanced options
    if st.sidebar.checkbox("Advanced Market Parameters", False):
        st.sidebar.markdown("### Advanced Parameters")
        
        # Volatility term structure
        vol_term_structure = st.sidebar.checkbox("Use Volatility Term Structure", False)
        if vol_term_structure:
            vol_3m = st.sidebar.number_input("3-Month Volatility", value=volatility*0.9, step=0.01, format="%.2f")
            vol_6m = st.sidebar.number_input("6-Month Volatility", value=volatility, step=0.01, format="%.2f")
            vol_12m = st.sidebar.number_input("12-Month Volatility", value=volatility*1.1, step=0.01, format="%.2f")
            model_params['vol_term_structure'] = {
                0.25: vol_3m,
                0.5: vol_6m,
                1.0: vol_12m
            }
        
        # Interest rate term structure
        rate_term_structure = st.sidebar.checkbox("Use Rate Term Structure", False)
        if rate_term_structure:
            rate_3m = st.sidebar.number_input("3-Month Rate", value=risk_free_rate*0.8, step=0.001, format="%.3f")
            rate_6m = st.sidebar.number_input("6-Month Rate", value=risk_free_rate, step=0.001, format="%.3f")
            rate_12m = st.sidebar.number_input("12-Month Rate", value=risk_free_rate*1.2, step=0.001, format="%.3f")
            model_params['rate_term_structure'] = {
                0.25: rate_3m,
                0.5: rate_6m,
                1.0: rate_12m
            }
        
        # Dividend yield
        div_yield = st.sidebar.number_input("Dividend Yield", value=0.0, step=0.001, format="%.3f")
        if div_yield > 0:
            model_params['dividend_yield'] = div_yield
        
        # Market skew parameter
        skew = st.sidebar.slider("Volatility Skew", -0.2, 0.2, 0.0, 0.01)
        if skew != 0:
            model_params['skew'] = skew
    
    return model_type, current_price, strike_price, time_to_maturity, volatility, risk_free_rate, model_params

# Calculate option prices
def calculate_option_prices(model_type, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, model_params):
    """Calculate option prices based on the selected model"""
    if model_type == "Black-Scholes":
        call_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                      risk_free_rate, volatility, 'call')
        put_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                     risk_free_rate, volatility, 'put')
        return {"call": f"${call_value:.2f}", "put": f"${put_value:.2f}"}, call_value, put_value, None
        
    elif model_type == "Binomial":
        steps = model_params.get('steps', 100)
        option_style = model_params.get('option_style', 'European').lower()
        call_value = binomial_calc(current_price, strike_price, time_to_maturity,
                                 risk_free_rate, volatility, steps, 'call', option_style)
        put_value = binomial_calc(current_price, strike_price, time_to_maturity,
                                risk_free_rate, volatility, steps, 'put', option_style)
        return {"call": f"${call_value:.2f}", "put": f"${put_value:.2f}"}, call_value, put_value, None
        
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
        return price_info, call_value, put_value, (call_paths, put_paths, n_steps)

def setup_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Options Pricing Models",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def add_custom_css():
    """Add custom CSS styling to the app"""
    st.markdown("""
        <style>
        .greek-card {
            background-color: #2E2E2E;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem;
        }
        .greek-label { color: #9CA3AF; font-size: 0.875rem; }
        .greek-value { color: white; font-size: 1.25rem; font-weight: 600; }
        .main { background-color: #0E1117; }
        </style>
    """, unsafe_allow_html=True)

def add_author_section():
    """Add author information section"""
    st.markdown("""
        <div style="background-color: #1E2124; padding: 15px; border-radius: 10px; width: fit-content; margin-bottom: 20px;">
            <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 8px;">Created by</div>
            <div style="display: flex; align-items: center;">
                <div style="margin-right: 15px;">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="40" height="40">
                        <path fill="#0A66C2" d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 00.1.4V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z"></path>
                    </svg>
                </div>
                <a href="https://www.linkedin.com/in/navnoorbawa/" target="_blank" style="color: white; text-decoration: none; font-size: 24px; font-weight: 500;">Navnoor Bawa</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_option_prices(price_info):
    """Display option prices in a clean format"""
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
    """Display Greeks in a minimal grid layout"""
    st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
            <h4 style="color: white; margin-bottom: 1rem;">Position Greeks</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;">
                <div class="greek-card">
                    <div class="greek-label">Delta</div>
                    <div class="greek-value">{round(calculated_greeks['delta'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Gamma</div>
                    <div class="greek-value">{round(calculated_greeks['gamma'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Theta</div>
                    <div class="greek-value">{round(calculated_greeks['theta'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Vega</div>
                    <div class="greek-value">{round(calculated_greeks['vega'], 3)}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def handle_basic_analysis_tab(model_type, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, paths_data):
    """Handle the Basic Analysis tab content"""
    # Display Greeks
    calculated_greeks = calculate_greeks("Call", current_price, strike_price,
                                     time_to_maturity, risk_free_rate, volatility)
    display_greeks(calculated_greeks)
    
    # Display advanced Greeks if requested
    if st.checkbox("Show Advanced Greeks"):
        advanced_greeks = calculate_advanced_greeks("Call", current_price, strike_price,
                                                time_to_maturity, risk_free_rate, volatility)
        st.markdown("### Advanced Greeks")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                    <h4 style="color: white;">Second-Order Greeks</h4>
                    <ul style="color: white; list-style-type: none; padding-left: 0;">
                        <li>• Vanna: {advanced_greeks['vanna']:.4f} (Delta-Vega Sensitivity)</li>
                        <li>• Charm: {advanced_greeks['charm']:.4f} (Delta Decay)</li>
                        <li>• Volga: {advanced_greeks['volga']:.4f} (Vega Convexity)</li>
                        <li>• Veta: {advanced_greeks['veta']:.4f} (Vega Decay)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                    <h4 style="color: white;">Third-Order Greeks</h4>
                    <ul style="color: white; list-style-type: none; padding-left: 0;">
                        <li>• Speed: {advanced_greeks['speed']:.4f} (Delta Acceleration)</li>
                        <li>• Zomma: {advanced_greeks['zomma']:.4f} (Gamma-Volga)</li>
                        <li>• Color: {advanced_greeks['color']:.4f} (Gamma Decay)</li>
                        <li>• Ultima: {advanced_greeks['ultima']:.4f} (Volga-Volga)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    # Display Monte Carlo specific visualizations if selected
    if model_type == "Monte Carlo" and paths_data:
        show_monte_carlo_visualizations(paths_data, time_to_maturity, current_price, strike_price, risk_free_rate, volatility)

def show_monte_carlo_visualizations(paths_data, time_to_maturity, current_price, strike_price, risk_free_rate, volatility):
    """Show Monte Carlo simulation visualizations"""
    call_paths, put_paths, n_steps = paths_data

    st.subheader("Monte Carlo Simulation Paths")
    fig = plt.figure(figsize=(10, 6))

    path_shape = call_paths[:100].T.shape
    time_points = path_shape[0]

    time_array = np.linspace(0, time_to_maturity, time_points)

    plt.plot(time_array, call_paths[:100].T, alpha=0.1)
    mean_path = np.mean(call_paths, axis=0)
    plt.plot(time_array, mean_path, 'r', linewidth=2)

    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.title('Monte Carlo Simulation Paths (first 100 paths)')
    st.pyplot(fig)

    # Added Value-at-Risk calculation option
    if st.checkbox("Calculate Value-at-Risk"):
        handle_var_calculation(current_price, strike_price, time_to_maturity, risk_free_rate, volatility)

def handle_var_calculation(current_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """Handle Value-at-Risk calculation and visualization"""
    confidence = st.slider("Confidence Level", 0.9, 0.99, 0.95, 0.01)
    horizon = st.slider("Risk Horizon (days)", 1, 30, 1) / 252

    # Add enhanced VaR options
    use_t_dist = st.checkbox("Use Student's t-distribution (fat tails)", True,
                           help="Better captures extreme market events than normal distribution")
    if use_t_dist:
        degrees_of_freedom = st.slider("Degrees of Freedom", 3, 10, 5, 1,
                                     help="Lower values create fatter tails (3-5 for financial markets)")
    else:
        degrees_of_freedom = 5

    use_garch = st.checkbox("Use GARCH volatility modeling", False,
                          help="Models time-varying volatility instead of constant volatility")

    # Add stress testing option
    stress_test_tab = st.checkbox("Include Stress Testing", False,
                               help="Test portfolio against historical crisis scenarios")

    # Enhanced VaR calculation with new parameters
    var_results = var_calculator(
        strategies=["Long Call"],
        quantities=[1],
        spot_price=current_price,
        strikes=[strike_price],
        maturities=[time_to_maturity],
        rates=risk_free_rate,
        vols=volatility,
        confidence=confidence,
        horizon=horizon,
        n_simulations=10000,
        use_t_dist=use_t_dist,
        degrees_of_freedom=degrees_of_freedom,
        use_garch=use_garch
    )

    display_var_results(var_results, confidence)
    
    # Add stress testing results if enabled
    if stress_test_tab:
        handle_stress_testing(current_price, strike_price, time_to_maturity, risk_free_rate, volatility)

def display_var_results(var_results, confidence):
    """Display Value-at-Risk calculation results"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <h4 style="color: white;">Value-at-Risk Metrics</h4>
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li>• VaR ({confidence*100:.1f}%): ${var_results['VaR']:.2f}</li>
                    <li>• Expected Shortfall: ${var_results['Expected_Shortfall']:.2f}</li>
                    <li>• Horizon: {var_results['Horizon_Days']:.0f} days</li>
                    <li>• Distribution: {var_results['Distribution']}</li>
                    <li>• Volatility Model: {var_results['Volatility_Model']}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <h4 style="color: white;">Risk Distribution</h4>
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li>• Volatility: ${var_results['Volatility']:.2f}</li>
                    <li>• Skewness: {var_results['Skewness']:.2f}</li>
                    <li>• Kurtosis: {var_results['Kurtosis']:.2f}</li>
                    <li>• Worst Case: ${var_results['Worst_Case']:.2f}</li>
                    <li>• Best Case: ${var_results['Best_Case']:.2f}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def handle_stress_testing(current_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """Handle stress testing display and visualization"""
    st.subheader("Stress Test Results")
    with st.spinner("Running stress tests..."):
        # Correct input values for stress_test_portfolio function
        stress_results = stress_test_portfolio(
            strategies=["Long Call"],  # Must be a list, not a string
            quantities=[1],             # Must be a list, not a single integer
            spot_price=current_price,   # Single float value
            strikes=[strike_price],     # Must be a list
            maturities=[time_to_maturity], # Must be a list
            rates=risk_free_rate,       # Single float value
            vols=volatility             # Single float value
        )
        visualize_stress_test_results(stress_results)

def visualize_stress_test_results(stress_results):
    """Visualize stress test results with charts and tables"""
    try:
        # Create DataFrame from results
        stress_data = []
        for scenario, results in stress_results.items():
            try:
                # Handle if results is a dictionary with expected structure
                if isinstance(results, dict) and "P&L" in results:
                    scenario_details = "N/A"
                    if "Applied_Scenario" in results:
                        scenario_details = f"{results['Applied_Scenario']['Price Change']} price, {results['Applied_Scenario']['Volatility Multiplier']} vol, {results['Applied_Scenario']['Rate Change']} rate"
                    
                    stress_data.append({
                        "Scenario": scenario,
                        "P&L": results["P&L"],
                        "% Change": results.get("Portfolio_Change_Pct", 0.0),
                        "Scenario Details": scenario_details
                    })
                else:
                    # Handle if results is just a float
                    stress_data.append({
                        "Scenario": scenario,
                        "P&L": float(results) if isinstance(results, (int, float)) else 0.0,
                        "% Change": 0.0,
                        "Scenario Details": "N/A"
                    })
            except Exception as e:
                st.warning(f"Error processing scenario {scenario}: {str(e)}")
        
        stress_df = pd.DataFrame(stress_data)
        
        # Display results as a table
        st.dataframe(stress_df.style.format({
            "P&L": "${:.2f}",
            "% Change": "{:.2f}%"
        }))
        
        # Create bar chart of stress test results
        fig = plt.figure(figsize=(12, 7))
        ax = plt.gca()
        bars = plt.bar(stress_df["Scenario"], stress_df["P&L"],
               color=['#FF5555' if x < 0 else '#55CC55' for x in stress_df["P&L"]],
               width=0.7)
        plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        plt.ylabel('P&L ($)', fontsize=12)
        plt.xlabel('Scenario', fontsize=12)
        plt.title('Stress Test Results', fontsize=14, fontweight='bold')
        plt.xticks(rotation=30, ha='right', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout(pad=2)

        # Add values on top of bars with better positioning
        for bar in bars:
            height = bar.get_height()
            y_pos = min(-0.7, height - 0.5) if height < 0 else max(0.3, height + 0.5)
            plt.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'${height:.2f}',
                   ha='center',
                   va='bottom' if height >= 0 else 'top',
                   color='white',
                   fontweight='bold',
                   fontsize=11)
            
        # Add more whitespace at the bottom for labels
        plt.subplots_adjust(bottom=0.15)
            
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error processing stress test results: {str(e)}")

def handle_strategy_tab(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value):
    """Handle the Strategy Analysis tab content"""
    # Define strategy categories
    call_strategies = ["Covered Call Writing", "Long Call", "Bull Call Spread", "Bear Call Spread"]
    put_strategies = ["Long Put", "Protective Put", "Bull Put Spread", "Bear Put Spread"]
    combined_strategies = ["Long Straddle", "Short Straddle", "Iron Butterfly", "Iron Condor"]
    
    st.title("Options Strategy Analysis")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        strategy_category = st.selectbox("Select Strategy Category",
                                      ["Call Option Strategies", "Put Option Strategies", "Combined Strategies"],
                                      key="strategy_category")
    
    with col2:
        if strategy_category == "Call Option Strategies":
            strategy = st.selectbox("Select Strategy", call_strategies, key="call_strategy")
        elif strategy_category == "Put Option Strategies":
            strategy = st.selectbox("Select Strategy", put_strategies, key="put_strategy")
        else:
            strategy = st.selectbox("Select Strategy", combined_strategies, key="combined_strategy")
    
    display_strategy_analysis(strategy, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value)

def display_strategy_analysis(strategy, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value):
    """Display strategy analysis and visualizations"""
    strategy_explanations = {
        "Covered Call Writing": "Own the stock and sell a call option. This strategy generates income but caps upside potential.",
        "Long Call": "Purchase a call option to profit from upward price movements with limited risk.",
        "Bull Call Spread": "Buy a lower strike call and sell a higher strike call. Reduces cost but caps profit potential.",
        "Bear Call Spread": "Sell a lower strike call and buy a higher strike call. Generates income with limited risk.",
        "Long Put": "Purchase a put option to profit from downward price movements with limited risk.",
        "Protective Put": "Own the stock and buy a put option as insurance against downside risk.",
        "Bull Put Spread": "Sell a higher strike put and buy a lower strike put. Generates income with limited risk.",
        "Bear Put Spread": "Buy a higher strike put and sell a lower strike put. Reduces cost but caps profit potential.",
        "Long Straddle": "Buy both a call and put at the same strike. Profit from large price movements in either direction.",
        "Short Straddle": "Sell both a call and put at the same strike. Profit from low volatility and small price movements.",
        "Iron Butterfly": "Combination of a bear call spread and bull put spread with the same middle strike.",
        "Iron Condor": "Combination of a bear call spread and bull put spread with a gap between middle strikes."
    }
    
    st.info(strategy_explanations.get(strategy, ""))
    
    # Visualize strategy
    spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 200)
    fig_pnl = create_strategy_visualization(
        strategy, spot_range, current_price, strike_price,
        time_to_maturity, risk_free_rate, volatility, call_value, put_value
    )
    
    st.pyplot(fig_pnl)
    
    # Display strategy greeks
    st.subheader("Strategy Risk Profile")
    calculated_greeks = calculate_strategy_greeks(
        strategy, current_price, strike_price, time_to_maturity, risk_free_rate, volatility
    )
    display_greeks(calculated_greeks)
    
    # Calculate and display strategy performance metrics
    if st.checkbox("Show Detailed Strategy Performance"):
        display_strategy_performance(strategy, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value)

def display_strategy_performance(strategy, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value):
    """Display detailed strategy performance metrics"""
    perf_metrics = calculate_strategy_performance(
        strategy, current_price, strike_price, time_to_maturity,
        risk_free_rate, volatility, call_value, put_value
    )
    
    st.subheader("Strategy Performance Metrics")
    
    # Profitability metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <h4 style="color: white;">Profitability Metrics</h4>
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li>• Max Profit: ${perf_metrics['profitability']['max_profit']:.2f}</li>
                    <li>• Max Loss: ${perf_metrics['profitability']['max_loss']:.2f}</li>
                    <li>• Risk-Reward Ratio: {perf_metrics['profitability']['risk_reward_ratio']:.2f}</li>
                    <li>• Profit Probability: {perf_metrics['profitability']['profit_probability']:.1%}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <h4 style="color: white;">Risk Metrics</h4>
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li>• P&L Volatility: ${perf_metrics['risk_metrics']['pnl_volatility']:.2f}</li>
                    <li>• Sharpe Ratio: {perf_metrics['risk_metrics']['sharpe_ratio']:.2f}</li>
                    <li>• Kelly Criterion: {perf_metrics['risk_metrics']['kelly_criterion']:.1%}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def handle_quant_tab(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value):
    """Handle the Quantitative Analysis tab content"""
    st.subheader("Quantitative Analysis")
    st.info("Select a quant tool to perform advanced analysis")
    
    quant_tool = st.selectbox(
        "Select Quantitative Tool",
        ["Implied Volatility", "Local Volatility Surface", "Value at Risk (VaR)", "Risk Scenario Analysis"]
    )
    
    if quant_tool == "Implied Volatility":
        handle_implied_volatility_tool(current_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    elif quant_tool == "Local Volatility Surface":
        handle_local_volatility_tool(current_price, risk_free_rate, volatility)
    elif quant_tool == "Value at Risk (VaR)":
        handle_var_tool(current_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    elif quant_tool == "Risk Scenario Analysis":
        handle_risk_scenario_tool(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value)

def handle_implied_volatility_tool(current_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """Handle the Implied Volatility calculator tool"""
    st.subheader("Implied Volatility Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        market_price = st.number_input("Market Option Price", value=5.0, step=0.1)
        option_type = st.selectbox("Option Type", ["call", "put"])
    
    with col2:
        initial_guess = st.slider("Initial Volatility Guess", 0.1, 1.0, 0.2, 0.05)
    
    if st.button("Calculate Implied Volatility"):
        try:
            iv = implied_volatility(
                market_price, current_price, strike_price, time_to_maturity,
                risk_free_rate, option_type, precision=1e-8, max_iterations=20
            )
            
            st.success(f"Implied Volatility: {iv:.2%}")
            
            # Display implied vol vs current vol
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Implied Volatility", f"{iv:.2%}", f"{(iv-volatility)/volatility:.2%}")
            
            with col2:
                st.metric("Model Volatility", f"{volatility:.2%}")
        except Exception as iv_error:
            st.error(f"Error calculating implied volatility: {str(iv_error)}")

def handle_local_volatility_tool(current_price, risk_free_rate, volatility):
    """Handle the Local Volatility Surface generator tool"""
    st.subheader("Local Volatility Surface Generator")
    
    # Introduction with theory
    with st.expander("About Local Volatility Models"):
        st.markdown("""
        ### The Dupire Local Volatility Model
        
        The local volatility model, developed by Bruno Dupire and Emanuel Derman, extends the Black-Scholes framework by allowing volatility to vary with both strike price and time. This addresses key market features like volatility skew and smile that constant volatility models cannot capture.
        
        #### Mathematical Foundation:
        
        Dupire's formula relates the local volatility $\sigma_{loc}(K, T)$ to the implied volatility surface:
        
        $$\sigma_{loc}^2(K, T) = \\frac{\\frac{\partial C}{\partial T} + (r-q)K\\frac{\partial C}{\partial K} + qC}{\\frac{1}{2}K^2\\frac{\partial^2 C}{\partial K^2}}$$
        
        where:
        - $C$ is the call option price
        - $K$ is the strike price
        - $T$ is time to maturity
        - $r$ is the risk-free rate
        - $q$ is the dividend yield
        
        This tool implements a numerically stable version of this formula with sophisticated smoothing techniques.
        """)
    
    # Create interface with two columns
    col1, col2 = st.columns([2, 2])
    
    with col1:
        # Data source selection
        data_source = st.radio(
            "Select Data Source",
            ["Simulated Market", "Custom Parameters", "Upload Market Data"],
            key="vol_data_source"
        )
        
        # Get data source parameters based on selection
        data_params = get_local_vol_data_params(data_source, current_price, volatility)
    
    with col2:
        # Common parameters for all modes
        st.subheader("Calculation Parameters")
        
        dividend_yield = st.number_input(
            "Dividend Yield (%)",
            value=0.0, min_value=0.0, max_value=10.0, step=0.1
        ) / 100.0
        
        smoothing_level = st.slider(
            "Surface Smoothing",
            0.0, 1.0, 0.2, 0.05,
            help="Controls smoothness of the output (higher = smoother)"
        )
        
        # Visualization options
        st.subheader("Visualization Options")
        
        display_mode = st.radio(
            "Display Mode",
            ["3D Surface", "Contour Plot", "Term Structure", "Skew Analysis", "All Views"]
        )
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ["viridis", "plasma", "inferno", "magma", "cividis"]
        )

    # Generate volatility surfaces based on selected method
    if st.button("Generate Local Volatility Surface"):
        with st.spinner("Calculating local volatility surface..."):
            try:
                process_local_volatility_calculation(
                    data_source, data_params, current_price, risk_free_rate,
                    volatility, dividend_yield, smoothing_level, display_mode, color_scheme
                )
            except Exception as e:
                st.error(f"Error calculating volatility surface: {str(e)}")
                st.exception(e)

def get_local_vol_data_params(data_source, current_price, volatility):
    """Get parameters for local volatility data source"""
    data_params = {}
    
    if data_source == "Simulated Market":
        # Parameters for simulated data
        st.subheader("Simulated Market Parameters")
        data_params['skew_factor'] = st.slider("Volatility Skew", -0.3, 0.3, -0.1, 0.05,
                                             help="Negative values create downward sloping skew (typical in equity markets)")
        data_params['smile_factor'] = st.slider("Volatility Smile", 0.0, 0.3, 0.05, 0.01,
                                              help="Higher values create stronger smile (U-shape)")
        data_params['term_structure'] = st.slider("Term Structure", -0.1, 0.2, 0.05, 0.01,
                                                help="Positive values mean longer-dated options have higher volatility")
    
    elif data_source == "Custom Parameters":
        # Advanced custom parameters
        st.subheader("Custom Surface Parameters")
        data_params['base_vol'] = st.number_input("Base Volatility", value=volatility, min_value=0.01, max_value=1.0, step=0.01)
        data_params['num_strikes'] = st.slider("Number of Strikes", 5, 15, 9)
        data_params['num_maturities'] = st.slider("Number of Maturities", 3, 10, 6)
        data_params['moneyness_range'] = st.slider("Moneyness Range", 0.5, 2.0, (0.7, 1.3))
        data_params['max_maturity'] = st.slider("Maximum Maturity (Years)", 0.5, 5.0, 2.0)
    
    elif data_source == "Upload Market Data":
        # File upload option
        st.subheader("Upload Market Data")
        data_params['uploaded_file'] = st.file_uploader(
            "Upload option chain CSV",
            type=["csv"],
            help="CSV should contain: Strike, Maturity, ImpliedVol columns"
        )
        
        if data_params.get('uploaded_file') is not None:
            st.info("File uploaded successfully. Configure processing parameters below.")
        
        data_params['interpolation_method'] = st.selectbox(
            "Interpolation Method",
            ["Cubic Spline", "Thin Plate Spline", "Linear"],
            help="Method used to fill gaps in market data"
        )
    
    return data_params

def process_local_volatility_calculation(data_source, data_params, current_price, risk_free_rate,
                                      volatility, dividend_yield, smoothing_level, display_mode, color_scheme):
    """Process local volatility surface calculation and visualization"""
    from scipy.interpolate import SmoothBivariateSpline, Rbf, griddata
    
    # Initialize parameters based on data source
    if data_source == "Simulated Market":
        # Create simulated data with user parameters
        strike_pcts = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
        maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
        
        strikes = current_price * strike_pcts
        
        # Create volatility surface with user-specified skew, smile, and term structure
        implied_vols = np.zeros((len(maturities), len(strikes)))
        for i, t in enumerate(maturities):
            for j, k in enumerate(strike_pcts):
                # Model volatility with skew, smile and term structure
                moneyness_effect = data_params['skew_factor'] * (1.0 - k)
                smile_effect = data_params['smile_factor'] * (k - 1.0)**2
                term_effect = data_params['term_structure'] * (t - maturities.mean())
                
                implied_vols[i, j] = max(0.05, volatility + moneyness_effect + smile_effect + term_effect)
    
    elif data_source == "Custom Parameters":
        # Generate custom grid based on user specifications
        strike_pcts = np.linspace(data_params['moneyness_range'][0], data_params['moneyness_range'][1], data_params['num_strikes'])
        maturities = np.linspace(1/12, data_params['max_maturity'], data_params['num_maturities'])
        
        strikes = current_price * strike_pcts
        
        # Generate implied volatility surface with custom parameters
        implied_vols = np.zeros((len(maturities), len(strikes)))
        for i, t in enumerate(maturities):
            for j, k in enumerate(strike_pcts):
                # Simple volatility model with term structure and skew
                term_factor = 0.05 * (t - maturities.mean())
                skew_factor = -0.1 * (k - 1.0)
                smile_factor = 0.05 * (k - 1.0)**2
                
                implied_vols[i, j] = max(0.05, data_params['base_vol'] + term_factor + skew_factor + smile_factor)
    
    elif data_source == "Upload Market Data" and data_params.get('uploaded_file') is not None:
        # Process uploaded data
        market_data = pd.read_csv(data_params['uploaded_file'])
        
        # Validate required columns
        required_columns = ["Strike", "Maturity", "ImpliedVol"]
        if not all(col in market_data.columns for col in required_columns):
            st.error(f"CSV must contain columns: {', '.join(required_columns)}")
            return
        
        # Extract unique strikes and maturities
        unique_strikes = np.sort(market_data["Strike"].unique())
        unique_maturities = np.sort(market_data["Maturity"].unique())
        
        # Create matrix for implied volatilities
        strikes = unique_strikes
        maturities = unique_maturities
        implied_vols = np.zeros((len(maturities), len(strikes)))
        
        # Fill matrix with market data
        for i, t in enumerate(maturities):
            for j, k in enumerate(strikes):
                mask = (market_data["Maturity"] == t) & (market_data["Strike"] == k)
                if mask.any():
                    implied_vols[i, j] = market_data.loc[mask, "ImpliedVol"].values[0]
                else:
                    # If using interpolation, we'll set missing values to NaN
                    implied_vols[i, j] = np.nan
        
        # Handle missing values in the grid
        if np.isnan(implied_vols).any():
            implied_vols = interpolate_missing_values(implied_vols, maturities, strikes, data_params['interpolation_method'])
    else:
        st.error("Please upload a file when using 'Upload Market Data' option")
        return
    
    # Generate local volatility surface
    local_vols, smoothed_ivs, diagnostics = enhanced_local_volatility_surface(
        strikes, maturities, implied_vols, current_price, risk_free_rate,
        dividend_yield, smoothing_level
    )
    
    # Analyze the volatility surfaces
    analysis_results = analyze_vol_surface(
        local_vols, smoothed_ivs, strikes, maturities, current_price
    )
    
    # Display visualizations based on selected mode
    display_vol_surface_visualizations(
        display_mode, strikes, maturities, smoothed_ivs, local_vols,
        current_price, color_scheme, analysis_results, diagnostics
    )

def interpolate_missing_values(implied_vols, maturities, strikes, interpolation_method):
    """Interpolate missing values in the volatility surface"""
    from scipy.interpolate import SmoothBivariateSpline, Rbf, griddata
    
    if interpolation_method == "Linear":
        # Simple linear interpolation for missing values
        # Get coordinates of non-NaN values
        ii, jj = np.where(~np.isnan(implied_vols))
        known_points = np.column_stack([ii, jj])
        known_values = implied_vols[~np.isnan(implied_vols)]
        
        # Create grid for interpolation
        grid_i, grid_j = np.mgrid[0:len(maturities), 0:len(strikes)]
        
        # Interpolate
        implied_vols = griddata(known_points, known_values, (grid_i, grid_j), method='linear')
        
        # Fill any remaining NaNs with nearest value
        if np.isnan(implied_vols).any():
            mask = np.isnan(implied_vols)
            implied_vols[mask] = griddata(known_points, known_values, (grid_i[mask], grid_j[mask]), method='nearest')
    
    elif interpolation_method == "Cubic Spline":
        # More advanced spline interpolation
        # Get coordinates of non-NaN values
        ii, jj = np.where(~np.isnan(implied_vols))
        known_values = implied_vols[~np.isnan(implied_vols)]
        
        # Convert indices to actual maturity and strike values
        t_points = maturities[ii]
        k_points = strikes[jj]
        
        # Create spline
        spline = SmoothBivariateSpline(t_points, k_points, known_values, s=0.1)
        
        # Evaluate spline at all grid points
        T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing='ij')
        implied_vols = spline(maturities, strikes)
        
    else:  # Thin Plate Spline
        # Get coordinates of non-NaN values
        ii, jj = np.where(~np.isnan(implied_vols))
        known_values = implied_vols[~np.isnan(implied_vols)]
        
        # Convert indices to actual maturity and strike values
        t_points = maturities[ii]
        k_points = strikes[jj]
        
        # Create RBF interpolator
        rbf = Rbf(t_points, k_points, known_values, function='thin_plate')
        
        # Evaluate RBF at all grid points
        T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing='ij')
        implied_vols = rbf(T_mesh, K_mesh)
        
    return implied_vols

def display_vol_surface_visualizations(display_mode, strikes, maturities, smoothed_ivs, local_vols,
                                     current_price, color_scheme, analysis_results, diagnostics):
    """Display visualizations for volatility surface analysis"""
    # Create grid for visualization
    K_grid, T_grid = np.meshgrid(strikes/current_price, maturities)
    
    if display_mode == "3D Surface" or display_mode == "All Views":
        st.subheader("Volatility Surfaces")
        
        fig = plt.figure(figsize=(14, 10))
        
        # Plot two surfaces: implied and local volatility
        ax1 = fig.add_subplot(221, projection='3d')
        surf1 = ax1.plot_surface(K_grid, T_grid, smoothed_ivs, cmap=color_scheme, alpha=0.8)
        ax1.set_xlabel('Moneyness (K/S)')
        ax1.set_ylabel('Maturity (Years)')
        ax1.set_zlabel('Volatility')
        ax1.set_title('Implied Volatility Surface')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        
        ax2 = fig.add_subplot(222, projection='3d')
        surf2 = ax2.plot_surface(K_grid, T_grid, local_vols, cmap=color_scheme, alpha=0.8)
        ax2.set_xlabel('Moneyness (K/S)')
        ax2.set_ylabel('Maturity (Years)')
        ax2.set_zlabel('Volatility')
        ax2.set_title('Local Volatility Surface')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        # Add contour plots below for clarity
        ax3 = fig.add_subplot(223)
        contour1 = ax3.contourf(K_grid, T_grid, smoothed_ivs, levels=20, cmap=color_scheme)
        ax3.set_xlabel('Moneyness (K/S)')
        ax3.set_ylabel('Maturity (Years)')
        ax3.set_title('Implied Volatility Contour Map')
        fig.colorbar(contour1, ax=ax3, shrink=0.5, aspect=5)
        
        ax4 = fig.add_subplot(224)
        contour2 = ax4.contourf(K_grid, T_grid, local_vols, levels=20, cmap=color_scheme)
        ax4.set_xlabel('Moneyness (K/S)')
        ax4.set_ylabel('Maturity (Years)')
        ax4.set_title('Local Volatility Contour Map')
        fig.colorbar(contour2, ax=ax4, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    if display_mode == "Contour Plot" or display_mode == "All Views":
        # High-resolution contour plots
        st.subheader("Volatility Contour Maps")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Implied volatility contour
        im1 = ax1.contourf(K_grid, T_grid, smoothed_ivs, levels=30, cmap=color_scheme)
        ax1.set_xlabel('Moneyness (K/S)')
        ax1.set_ylabel('Maturity (Years)')
        ax1.set_title('Implied Volatility')
        plt.colorbar(im1, ax=ax1)
        
        # Add contour lines for key volatility levels
        cs1 = ax1.contour(K_grid, T_grid, smoothed_ivs, levels=5, colors='white', alpha=0.6, linewidths=0.5)
        ax1.clabel(cs1, inline=True, fontsize=8, fmt='%.2f')
        
        # Local volatility contour
        im2 = ax2.contourf(K_grid, T_grid, local_vols, levels=30, cmap=color_scheme)
        ax2.set_xlabel('Moneyness (K/S)')
        ax2.set_ylabel('Maturity (Years)')
        ax2.set_title('Local Volatility')
        plt.colorbar(im2, ax=ax2)
        
        # Add contour lines for key volatility levels
        cs2 = ax2.contour(K_grid, T_grid, local_vols, levels=5, colors='white', alpha=0.6, linewidths=0.5)
        ax2.clabel(cs2, inline=True, fontsize=8, fmt='%.2f')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    if display_mode == "Term Structure" or display_mode == "All Views":
        # Term structure visualization
        st.subheader("Volatility Term Structure")
        
        # Get ATM volatilities
        atm_index = np.argmin(np.abs(strikes/current_price - 1.0))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(maturities, smoothed_ivs[:, atm_index], 'b-o', label='ATM Implied Vol')
        ax.plot(maturities, local_vols[:, atm_index], 'r-^', label='ATM Local Vol')
        
        # Add OTM volatilities
        otm_put_idx = max(0, np.argmin(np.abs(strikes/current_price - 0.9)))
        otm_call_idx = min(len(strikes)-1, np.argmin(np.abs(strikes/current_price - 1.1)))
        
        ax.plot(maturities, smoothed_ivs[:, otm_put_idx], 'b--', alpha=0.6, label='90% OTM Put Implied Vol')
        ax.plot(maturities, smoothed_ivs[:, otm_call_idx], 'b-.', alpha=0.6, label='110% OTM Call Implied Vol')
        
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Volatility')
        ax.set_title('Volatility Term Structure')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
    
    if display_mode == "Skew Analysis" or display_mode == "All Views":
        # Volatility skew visualization
        st.subheader("Volatility Skew Analysis")
        
        # Select maturities for skew analysis
        if len(maturities) >= 3:
            maturity_indices = [0, len(maturities)//2, -1]  # Short, medium, long-term
            maturity_labels = ['Short-term', 'Medium-term', 'Long-term']
        else:
            maturity_indices = list(range(len(maturities)))
            maturity_labels = [f"T={t:.2f}" for t in maturities]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, idx in enumerate(maturity_indices):
            if idx < 0:  # Handle negative index
                actual_idx = len(maturities) + idx
            else:
                actual_idx = idx
                
            ax.plot(strikes/current_price, smoothed_ivs[actual_idx, :],
                   f'C{i}-o', label=f'{maturity_labels[i]} Implied Vol (T={maturities[actual_idx]:.2f})')
            ax.plot(strikes/current_price, local_vols[actual_idx, :],
                   f'C{i}--', label=f'{maturity_labels[i]} Local Vol (T={maturities[actual_idx]:.2f})')
        
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Volatility')
        ax.set_title('Volatility Skew Across Maturities')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
    
    # Display analysis results
    display_vol_surface_analysis(analysis_results, maturities, strikes, smoothed_ivs, diagnostics, current_price)

def display_vol_surface_analysis(analysis_results, maturities, strikes, smoothed_ivs, diagnostics, current_price):
    """Display volatility surface analysis results"""
    st.subheader("Quantitative Analysis")
    
    # Organize results into columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Surface Characteristics")
        
        # ATM Term Structure
        atm_index = np.argmin(np.abs(strikes/current_price - 1.0))
        
        # Create DataFrame for ATM vols
        atm_vol_df = pd.DataFrame({
            'Maturity': maturities,
            'ATM Implied Vol': smoothed_ivs[:, atm_index],
            'ATM Local Vol': analysis_results['atm_local_vol_term']
        })
        
        st.markdown("#### ATM Volatility Term Structure")
        st.dataframe(atm_vol_df.style.format({
            'Maturity': '{:.2f}',
            'ATM Implied Vol': '{:.2%}',
            'ATM Local Vol': '{:.2%}'
        }))
        
        # Skew Analysis
        st.markdown("#### Volatility Skew Analysis")
        skew_df = pd.DataFrame(analysis_results['skew_analysis'])
        st.dataframe(skew_df.style.format({
            'maturity': '{:.2f}',
            'local_skew': '{:.4f}',
            'implied_skew': '{:.4f}'
        }))
    
    with col2:
        st.markdown("### Numerical Diagnostics")
        
        # Surface stability assessment
        st.markdown("#### Surface Stability")
        st.info(f"Stability Assessment: {analysis_results['surface_stability']['stability_assessment']}")
        st.metric("Smoothness Metric", f"{analysis_results['surface_stability']['smoothness_metric']:.4f}")
        
        # Implied vs Local Vol comparison
        st.markdown("#### IV vs LV Comparison")
        comp_df = pd.DataFrame([{
            'Metric': 'Mean Difference',
            'Value': analysis_results['vol_comparison']['mean_diff']
        }, {
            'Metric': 'Max Difference',
            'Value': analysis_results['vol_comparison']['max_diff']
        }, {
            'Metric': 'Min Difference',
            'Value': analysis_results['vol_comparison']['min_diff']
        }, {
            'Metric': 'RMS Difference',
            'Value': analysis_results['vol_comparison']['rms_diff']
        }])
        
        st.dataframe(comp_df.style.format({
            'Value': '{:.4f}'
        }))
        
        # Calculation diagnostics
        st.markdown("#### Calculation Diagnostics")
        diag_df = pd.DataFrame([{
            'Metric': 'Boundary Points',
            'Count': diagnostics['boundary_points']
        }, {
            'Metric': 'Denominator Fixes',
            'Count': diagnostics['denominator_fixes']
        }, {
            'Metric': 'Extreme Values Clipped',
            'Count': diagnostics['extreme_values_clipped']
        }])
        
        st.dataframe(diag_df)
    
    # Option to download data
    add_vol_surface_download_options(maturities, strikes, smoothed_ivs, analysis_results['atm_local_vol_term'], current_price, skew_df, atm_vol_df)

def add_vol_surface_download_options(maturities, strikes, smoothed_ivs, local_vols, current_price, skew_df, atm_vol_df):
    """Add download options for volatility surface data"""
    st.subheader("Download Results")

    try:
        # Try to import xlsxwriter
        import xlsxwriter
        excel_export_available = True
    except ImportError:
        excel_export_available = False

    # Create DataFrame collection for export
    export_data = {
        'Parameters': pd.DataFrame([
            {'Parameter': 'Spot Price', 'Value': current_price},
            # Add other parameters as needed
        ]),
        'Strikes': pd.DataFrame({'Strike': strikes, 'Moneyness': strikes/current_price}),
        'Maturities': pd.DataFrame({'Maturity': maturities}),
        'ImpliedVolSurface': pd.DataFrame(
            smoothed_ivs,
            index=[f'T={t:.2f}' for t in maturities],
            columns=[f'K={k:.1f}' for k in strikes]
        ),
        'LocalVolSurface': pd.DataFrame(
            np.outer(local_vols, np.ones(len(strikes))),  # Create a matrix from the vector
            index=[f'T={t:.2f}' for t in maturities],
            columns=[f'K={k:.1f}' for k in strikes]
        ),
        'ATM_TermStructure': atm_vol_df,
        'SkewAnalysis': skew_df
    }

    # CSV export option (always available)
    for name, df in export_data.items():
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        csv_buffer.seek(0)
        
        st.download_button(
            label=f"Download {name} as CSV",
            data=csv_buffer.getvalue().encode(),
            file_name=f"{name.lower()}.csv",
            mime="text/csv",
            key=f"csv_{name}"
        )

    # Excel export option (if xlsxwriter is available)
    if excel_export_available:
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            for name, df in export_data.items():
                df.to_excel(writer, sheet_name=name, index=name in ['ImpliedVolSurface', 'LocalVolSurface'])
        
        buffer.seek(0)
        
        st.download_button(
            label="Download All Data as Excel Workbook",
            data=buffer,
            file_name="volatility_surfaces.xlsx",
            mime="application/vnd.ms-excel"
        )
    else:
        st.info("Excel export requires the xlsxwriter package. Use the CSV downloads above or install xlsxwriter with 'pip install xlsxwriter'.")

def handle_var_tool(current_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """Handle the Value at Risk (VaR) calculator tool"""
    st.subheader("Value at Risk Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        confidence = st.slider("Confidence Level", 0.9, 0.99, 0.95, 0.01)
        horizon = st.slider("Risk Horizon (days)", 1, 30, 1) / 252  # Convert to years
    
    with col2:
        n_simulations = st.slider("Number of Simulations", 1000, 100000, 10000, 1000)
        strategy_type = st.selectbox("Position Type", ["Long Call", "Long Put", "Covered Call Writing", "Protective Put"])
    
    # Add enhanced VaR options
    use_t_dist = st.checkbox("Use Student's t-distribution (fat tails)", True,
                          help="Better captures extreme market events than normal distribution")
    if use_t_dist:
        degrees_of_freedom = st.slider("Degrees of Freedom", 3, 10, 5, 1,
                                     help="Lower values create fatter tails (3-5 for financial markets)")
    else:
        degrees_of_freedom = 5

    use_garch = st.checkbox("Use GARCH volatility modeling", False,
                          help="Models time-varying volatility instead of constant volatility")
    
    # Add stress testing option
    stress_test_tab = st.checkbox("Include Stress Testing", False,
                               help="Test portfolio against historical crisis scenarios")
    
    if st.button("Calculate VaR"):
        with st.spinner("Running VaR simulation..."):
            var_results = var_calculator(
                strategies=[strategy_type],
                quantities=[1],
                spot_price=current_price,
                strikes=[strike_price],
                maturities=[time_to_maturity],
                rates=risk_free_rate,
                vols=volatility,
                confidence=confidence,
                horizon=horizon,
                n_simulations=n_simulations,
                use_t_dist=use_t_dist,
                degrees_of_freedom=degrees_of_freedom,
                use_garch=use_garch
            )
            
            # Display VaR results
            display_var_results(var_results, confidence)
            
            # Add stress testing results if enabled
            if stress_test_tab:
                handle_stress_testing(current_price, strike_price, time_to_maturity, risk_free_rate, volatility)

def handle_risk_scenario_tool(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value):
    """Handle the Risk Scenario Analysis tool"""
    st.subheader("Strategy Scenario Analysis")
    
    strategy_list = ["Long Call", "Long Put", "Covered Call Writing", "Protective Put",
                  "Bull Call Spread", "Bear Put Spread", "Long Straddle", "Iron Condor"]
    selected_strategy = st.selectbox("Select Strategy for Analysis", strategy_list)
    
    # Add advanced options
    with st.expander("Advanced Analysis Options"):
        show_absolute_pnl = st.checkbox("Show Absolute P&L Values", True,
                                     help="If unchecked, shows P&L changes from current position value")
        
        visualization_mode = st.radio("Visualization Mode",
                                    ["Basic", "Comprehensive", "Risk Profile"],
                                    help="Basic: Key charts only\nComprehensive: All analysis charts\nRisk Profile: Summary view")
        
        num_scenarios = st.select_slider("Number of scenarios",
                                       options=[5, 7, 9, 11],
                                       value=7,
                                       help="More scenarios give finer granularity but may be harder to read")
    
    if st.button("Run Scenario Analysis"):
        with st.spinner("Running scenario analysis..."):
            # Get current option values for accurate P&L calculation
            current_call = black_scholes_calc(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'call')
            current_put = black_scholes_calc(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'put')
            
            # Run enhanced risk analysis
            risk_results = enhanced_risk_scenario_analysis(
                selected_strategy, current_price, strike_price, time_to_maturity,
                risk_free_rate, volatility, calculate_strategy_pnl,
                call_value=current_call, put_value=current_put
            )
            
            # Display analysis based on selected mode
            display_risk_scenario_analysis(risk_results, selected_strategy, current_price, strike_price, visualization_mode, show_absolute_pnl)

def display_risk_scenario_analysis(risk_results, selected_strategy, current_price, strike_price, visualization_mode, show_absolute_pnl):
    """Display risk scenario analysis visualizations"""
    if visualization_mode == "Basic" or visualization_mode == "Comprehensive":
        # Display price impact
        st.subheader("Price Impact Scenarios")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create more informative price impact chart
            price_fig = plt.figure(figsize=(12, 6))
            
            # Determine which P&L values to show based on user selection
            display_pnl = risk_results['price_impact']['absolute_pnl'] if show_absolute_pnl else risk_results['price_impact']['pnl_change']
            
     
            # Create color gradient based on values
            colors = ['#FF4136' if x < 0 else '#2ECC40' for x in display_pnl]
                
            # Create bars with enhanced styling - use uniform alpha
            x_labels = [f"${p:.0f}" for p in risk_results['price_impact']['scenarios']]
            bars = plt.bar(x_labels, display_pnl, color=colors, alpha=0.7, width=0.7)

            # If you want varying alpha values, set them individually after creating the bars
            for i, bar in enumerate(bars):
                # Calculate custom alpha value for this bar
                if len(display_pnl) > 0:  # Prevent division by zero
                    max_abs_value = max(abs(min(display_pnl)), abs(max(display_pnl)))
                    if max_abs_value > 0:  # Another safety check
                        custom_alpha = 0.6 + 0.4 * (abs(display_pnl[i]) / max_abs_value)
                        bar.set_alpha(custom_alpha)
            plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
            plt.xlabel('Price Scenarios', fontsize=12)
            plt.ylabel('P&L ($)' if show_absolute_pnl else 'P&L Change ($)', fontsize=12)
            plt.title(f'Price Impact on {selected_strategy} P&L', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # Add more informative labels
            plt.xticks(rotation=45, ha='right')
            current_price_index = np.argmin(abs(risk_results['price_impact']['scenarios'] - current_price))
            if current_price_index < len(x_labels):
                plt.axvline(x=current_price_index, color='yellow', linestyle='--', alpha=0.5)
                plt.text(current_price_index, plt.ylim()[0] * 0.9, 'Current',
                       rotation=90, va='bottom', color='yellow', alpha=0.7)
            
            # Add values on top of bars with better positioning
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2,
                       height + (0.01 * max(display_pnl) if height > 0 else 0.01 * min(display_pnl)),
                       f'${height:.2f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       color='white', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(price_fig)
        
        with col2:
            # Display price sensitivity metrics
            st.markdown("### Price Sensitivity")
            st.markdown(f"""
                <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px;">
                    <ul style="color: white; list-style-type: none; padding-left: 0;">
                        <li>• Max Gain: <span style="color: #2ECC40">${risk_results['price_impact']['max_gain']:.2f}</span></li>
                        <li>• Max Loss: <span style="color: #FF4136">${risk_results['price_impact']['max_loss']:.2f}</span></li>
                        <li>• P&L per 1% price move: ${risk_results['risk_profile']['price_sensitivity']:.2f}</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Show additional visualizations for comprehensive mode
        if visualization_mode == "Comprehensive":
            display_comprehensive_scenario_analysis(risk_results, selected_strategy, current_price, show_absolute_pnl, volatility)
        
        # Display extreme scenarios
        display_extreme_scenarios(risk_results)
    
    # Risk Profile Summary
    if visualization_mode == "Risk Profile" or visualization_mode == "Comprehensive":
        display_risk_profile_summary(risk_results, selected_strategy, current_price, strike_price)

def display_comprehensive_scenario_analysis(risk_results, selected_strategy, current_price, show_absolute_pnl, volatility):
    """Display comprehensive scenario analysis visualizations"""
    # Volatility impact
    st.subheader("Volatility Impact Scenarios")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create volatility impact chart
        vol_fig = plt.figure(figsize=(12, 6))
        
        # Determine which P&L values to show based on user selection
        display_pnl = risk_results['vol_impact']['absolute_pnl'] if show_absolute_pnl else risk_results['vol_impact']['pnl_change']
        
        # Create color gradient based on values
        colors = ['#FF4136' if x < 0 else '#2ECC40' for x in display_pnl]
        
        # Create bars with enhanced styling
        x_labels = [f"{v*100:.0f}%" for v in risk_results['vol_impact']['scenarios']]
        bars = plt.bar(x_labels, display_pnl, color=colors, width=0.7)
        
        plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        plt.xlabel('Volatility Scenarios', fontsize=12)
        plt.ylabel('P&L ($)' if show_absolute_pnl else 'P&L Change ($)', fontsize=12)
        plt.title(f'Volatility Impact on {selected_strategy} P&L', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
       
        # Add markers for current vol - use volatility instead of current_vol
        current_vol_index = np.argmin(np.abs(np.array(risk_results['vol_impact']['scenarios']) - volatility))
        plt.axvline(x=current_vol_index, color='yellow', linestyle='--', alpha=0.5)
        plt.text(current_vol_index, plt.ylim()[0] * 0.9, 'Current',
               rotation=90, va='bottom', color='yellow', alpha=0.7)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2,
                   height + (0.01 * max(display_pnl) if height > 0 else 0.01 * min(display_pnl)),
                   f'${height:.2f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   color='white', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(vol_fig)
    
    with col2:
        # Display vol sensitivity metrics
        st.markdown("### Volatility Sensitivity")
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px;">
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li>• Max Gain: <span style="color: #2ECC40">${risk_results['vol_impact']['max_gain']:.2f}</span></li>
                    <li>• Max Loss: <span style="color: #FF4136">${risk_results['vol_impact']['max_loss']:.2f}</span></li>
                    <li>• P&L per 1% vol change: ${risk_results['risk_profile']['vol_sensitivity']:.2f}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Time decay impact
    st.subheader("Time Decay Analysis")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        
        # Create time decay chart with correct orientation and behavior
        time_fig = plt.figure(figsize=(12, 6))

        # Determine which P&L values to show
        display_pnl = risk_results['time_decay']['absolute_pnl'] if show_absolute_pnl else risk_results['time_decay']['pnl_change']

        # Iron Condor typically benefits from time decay (theta positive)
        # Reverse the x-axis to show time passing left to right (days decreasing)
        days_to_expiry = risk_results['time_decay']['scenarios']  # These are in days
            
        # Sort x and y values by descending days (so days decrease from left to right)
        sorted_indices = np.argsort(days_to_expiry)[::-1]
        sorted_days = np.array(days_to_expiry)[sorted_indices]
        sorted_pnl = np.array(display_pnl)[sorted_indices]

        # Use line chart for time decay visualization
        plt.plot(sorted_days, sorted_pnl, 'o-',
               color='#FF851B', linewidth=2, markersize=8)

        plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        plt.xlabel('Days to Expiration', fontsize=12)
        plt.ylabel('P&L ($)' if show_absolute_pnl else 'P&L Change ($)', fontsize=12)
        plt.title(f'Time Decay Effect on {selected_strategy} P&L', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Add current time marker on the correct side of the graph
        current_time = time_to_maturity * 252  # Convert to days
        plt.axvline(x=current_time, color='yellow', linestyle='--', alpha=0.5)
        plt.text(current_time, plt.ylim()[0] * 0.9, 'Current',
               rotation=90, va='bottom', color='yellow', alpha=0.7)

        # Add value labels at key points - make sure we put them in readable positions
        for i, (x, y) in enumerate(zip(sorted_days, sorted_pnl)):
            if i % 2 == 0 or i == len(sorted_pnl) - 1:  # Label every other point to avoid crowding
                y_offset = 0.02 * max(abs(max(sorted_pnl)), abs(min(sorted_pnl)))
                plt.text(x, y + (y_offset if y > 0 else -y_offset),
                       f'${y:.2f}',
                       ha='center', va='bottom' if y > 0 else 'top',
                       color='white', fontweight='bold')

        # Ensure x-axis is properly formatted for days
        plt.xlim(max(sorted_days) * 1.05, min(sorted_days) * 0.9)  # Give some extra space on both ends
        plt.tight_layout()
        st.pyplot(time_fig)
    
    with col2:
        # Display time decay metrics
        st.markdown("### Time Decay Metrics")
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px;">
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li>• Daily Theta: <span style="${'color: #2ECC40' if risk_results['time_decay']['one_day_effect'] > 0 else 'color: #FF4136'}">${risk_results['time_decay']['one_day_effect']:.2f}</span></li>
                    <li>• Weekly Effect: <span style="${'color: #2ECC40' if risk_results['time_decay']['one_week_effect'] > 0 else 'color: #FF4136'}">${risk_results['time_decay']['one_week_effect']:.2f}</span></li>
                    <li>• Monthly Effect: <span style="${'color: #2ECC40' if risk_results['time_decay']['one_month_effect'] > 0 else 'color: #FF4136'}">${risk_results['time_decay']['one_month_effect']:.2f}</span></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def display_extreme_scenarios(risk_results):
    """Display extreme scenario analysis"""
    st.subheader("Market Stress Scenarios")
    
    # Process extreme scenario data
    scenario_names = list(risk_results['extreme_scenarios']['absolute_pnl'].keys())
    scenario_values = list(risk_results['extreme_scenarios']['absolute_pnl'].values())
    
    # Create DataFrame for visualization
    stress_df = pd.DataFrame({
        'Scenario': scenario_names,
        'P&L': scenario_values
    })
    
    # Sort by severity of impact
    stress_df = stress_df.sort_values('P&L')
    
    # Create bar chart of stress test results
    stress_fig = plt.figure(figsize=(12, 7))
    bars = plt.bar(
        stress_df["Scenario"],
        stress_df["P&L"],
        color=['#FF5555' if x < 0 else '#55CC55' for x in stress_df["P&L"]],
        width=0.7
    )
    plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    plt.ylabel('P&L ($)', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    plt.title('Stress Test Results', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars with better positioning
    for bar in bars:
        height = bar.get_height()
        y_pos = min(-0.7, height - 0.5) if height < 0 else max(0.3, height + 0.5)
        plt.text(
            bar.get_x() + bar.get_width()/2,
            y_pos,
            f'${height:.2f}',
            ha='center',
            va='bottom' if height >= 0 else 'top',
            color='white',
            fontweight='bold',
            fontsize=11
        )
    
    # Add current price marker
    current_index = stress_df[stress_df["Scenario"] == "current"].index
    if len(current_index) > 0:
        current_idx = current_index[0]
        plt.axvline(x=current_idx, color='yellow', linestyle='--', alpha=0.5)
        plt.text(current_idx, plt.ylim()[0] * 0.9, 'Current',
               rotation=90, va='bottom', color='yellow', alpha=0.7)
    
    plt.tight_layout(pad=2)
    st.pyplot(stress_fig)

def display_risk_profile_summary(risk_results, selected_strategy, current_price, strike_price):
    """Display risk profile summary for the selected strategy"""
    st.subheader(f"Risk Profile Summary for {selected_strategy}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current position metrics
        st.markdown("### Position Status")
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li>• Current Price: ${current_price:.2f}</li>
                    <li>• Strike Price: ${strike_price:.2f}</li>
                    <li>• Days to Expiry: {time_to_maturity*252:.0f}</li>
                    <li>• Volatility: {volatility*100:.1f}%</li>
                    <li>• Current P&L: ${risk_results['risk_profile']['current_pnl']:.2f}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Risk sensitivity summary
        st.markdown("### Risk Sensitivities")
        
        # Normalize sensitivities for visual comparison (0-100 scale)
        sensitivities = {
            'Price': risk_results['risk_profile']['price_sensitivity'],
            'Volatility': risk_results['risk_profile']['vol_sensitivity'],
            'Time': risk_results['risk_profile']['time_sensitivity']
        }
        
        max_sensitivity = max(sensitivities.values()) if any(sensitivities.values()) else 1
        normalized = {k: (v / max_sensitivity) * 100 for k, v in sensitivities.items()}
        
        # Create sensitivity visualization
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <p style="color: white;">Price Sensitivity: ${sensitivities['Price']:.2f} per 1%</p>
                <div style="background-color: #333; width: 100%; height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="background-color: #3498db; width: {normalized['Price']}%; height: 100%;"></div>
                </div>
                <p style="color: white; margin-top: 15px;">Volatility Sensitivity: ${sensitivities['Volatility']:.2f} per 1%</p>
                <div style="background-color: #333; width: 100%; height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="background-color: #9b59b6; width: {normalized['Volatility']}%; height: 100%;"></div>
                </div>
                <p style="color: white; margin-top: 15px;">Time Sensitivity: ${sensitivities['Time']:.2f} per day</p>
                <div style="background-color: #333; width: 100%; height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="background-color: #e67e22; width: {normalized['Time']}%; height: 100%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Risk-reward metrics
    st.subheader("Risk-Reward Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk metrics
        st.markdown("### Potential Outcomes")
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <ul style="color: white; list-style-type: none; padding-left: 0;">
                    <li>• Maximum Potential Gain: <span style="color: #2ECC40">${risk_results['risk_metrics']['max_potential_gain']:.2f}</span></li>
                    <li>• Maximum Potential Loss: <span style="color: #FF4136">${risk_results['risk_metrics']['max_potential_loss']:.2f}</span></li>
                    <li>• Worst-Case Scenario Loss: <span style="color: #FF4136">${risk_results['risk_metrics']['worst_case_loss']:.2f}</span></li>
                    <li>• Risk-Reward Ratio: {risk_results['risk_metrics']['risk_reward_ratio']:.2f}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Risk guidance based on the strategy
        risk_profile = get_strategy_risk_profile(selected_strategy)
        
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                <h4 style="color: white;">Strategy Guidance</h4>
                <div style="color: white;">
                    {risk_profile}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Add a comparison view of all impacts
    display_comparative_risk_analysis(risk_results, current_price, volatility)

def get_strategy_risk_profile(selected_strategy):
    """Get risk profile guidance for the selected strategy"""
    if "Call" in selected_strategy and "Long" in selected_strategy:
        return """
        <p><strong>Long Call Risk Profile:</strong></p>
        <ul>
            <li>Benefits from price increases and rising volatility</li>
            <li>Suffers from time decay and price decreases</li>
            <li>Maximum loss limited to premium paid</li>
            <li>Consider taking profits if large price increases occur</li>
        </ul>
        """
    elif "Put" in selected_strategy and "Long" in selected_strategy:
        return """
        <p><strong>Long Put Risk Profile:</strong></p>
        <ul>
            <li>Benefits from price decreases and rising volatility</li>
            <li>Suffers from time decay and price increases</li>
            <li>Maximum loss limited to premium paid</li>
            <li>Can be used as portfolio insurance</li>
        </ul>
        """
    elif "Covered Call" in selected_strategy:
        return """
        <p><strong>Covered Call Risk Profile:</strong></p>
        <ul>
            <li>Benefits from moderate price increases and time decay</li>
            <li>Suffers from large price increases (opportunity cost)</li>
            <li>Offers partial protection against price decreases</li>
            <li>Best for sideways or slightly bullish markets</li>
        </ul>
        """
    elif "Protective Put" in selected_strategy:
        return """
        <p><strong>Protective Put Risk Profile:</strong></p>
        <ul>
            <li>Benefits from price increases while limiting downside risk</li>
            <li>Suffers from time decay and high volatility premium</li>
            <li>Acts as insurance for long stock positions</li>
            <li>Consider rolling to further dates if protection still needed</li>
        </ul>
        """
    elif "Spread" in selected_strategy:
        return """
        <p><strong>Spread Strategy Risk Profile:</strong></p>
        <ul>
            <li>Benefits from price movement in expected direction</li>
            <li>Lower cost and risk than outright options</li>
            <li>Defined maximum profit and loss</li>
            <li>Best to exit if price moves strongly against position</li>
        </ul>
        """
    elif "Straddle" in selected_strategy:
        return """
        <p><strong>Straddle Risk Profile:</strong></p>
        <ul>
            <li>Benefits from large price movements in either direction</li>
            <li>Suffers from time decay if price remains stable</li>
            <li>Increased volatility is positive for this position</li>
            <li>Consider taking profits if large price move occurs</li>
        </ul>
        """
    else:
        return """
        <p><strong>Strategy Risk Profile:</strong></p>
        <ul>
            <li>Review the strategy payoff diagram for profit/loss zones</li>
            <li>Consider risk metrics and sensitivities when managing the position</li>
            <li>Monitor key risk factors that could impact position value</li>
        </ul>
        """

def display_comparative_risk_analysis(risk_results, current_price, volatility):
    """Display comparative risk analysis charts"""
    st.subheader("Comparative Risk Sensitivities")
    
    # Create a unified view of main sensitivities
    impacts = {
        'Price +10%': risk_results['price_impact']['pnl_change'][np.where(risk_results['price_impact']['scenarios'] >= current_price * 1.1)[0][0]] if any(risk_results['price_impact']['scenarios'] >= current_price * 1.1) else 0,
        'Price -10%': risk_results['price_impact']['pnl_change'][np.where(risk_results['price_impact']['scenarios'] <= current_price * 0.9)[0][-1]] if any(risk_results['price_impact']['scenarios'] <= current_price * 0.9) else 0,
        'Vol +50%': risk_results['vol_impact']['pnl_change'][np.where(risk_results['vol_impact']['scenarios'] >= volatility * 1.5)[0][0]] if any(risk_results['vol_impact']['scenarios'] >= volatility * 1.5) else 0,
        'Vol -50%': risk_results['vol_impact']['pnl_change'][np.where(risk_results['vol_impact']['scenarios'] <= volatility * 0.5)[0][-1]] if any(risk_results['vol_impact']['scenarios'] <= volatility * 0.5) else 0,
        '1 Week Passes': risk_results['time_decay']['one_week_effect'],
        'Market Crash': risk_results['extreme_scenarios']['pnl_change']['market_crash_2008'] if 'market_crash_2008' in risk_results['extreme_scenarios']['pnl_change'] else 0
    }
    
    # Create horizontal bar chart for comparative view
    impact_fig = plt.figure(figsize=(12, 7))
    factor_names = list(impacts.keys())
    impact_values = list(impacts.values())
    
    # Sort by absolute impact
    sorted_indices = np.argsort(np.abs(impact_values))[::-1]
    factor_names = [factor_names[i] for i in sorted_indices]
    impact_values = [impact_values[i] for i in sorted_indices]
    
    # Create horizontal bars
    bars = plt.barh(
        factor_names,
        impact_values,
        color=['#2ECC40' if x > 0 else '#FF4136' for x in impact_values],
        height=0.6
    )
    
    plt.axvline(x=0, color='white', linestyle='-', alpha=0.3)
    plt.xlabel('P&L Impact ($)', fontsize=12)
    plt.title('Comparative Risk Factor Impacts', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add values to bars
    for bar in bars:
        width = bar.get_width()
        x_pos = width + (0.01 * max(impact_values) if width > 0 else 0.01 * min(impact_values))
        plt.text(
            x_pos,
            bar.get_y() + bar.get_height()/2,
            f'${width:.2f}',
            va='center',
            color='white',
            fontweight='bold',
            fontsize=10
        )
    
    plt.tight_layout()
    st.pyplot(impact_fig)

def main():
    """Main application execution flow"""
    # Import all necessary libraries at the top level
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import warnings
    import io
    warnings.filterwarnings('ignore')
    
    try:
        # Setup page configuration and styling
        setup_page_config()
        add_custom_css()
        add_author_section()
        
        # Setup sidebar and get parameters
        model_type, current_price, strike_price, time_to_maturity, volatility, risk_free_rate, model_params = setup_sidebar()
        
        # Title and model selection display
        st.markdown(f"# 📈 {model_type} Option Pricing Model")
        
        # Calculate prices based on selected model
        with st.spinner("Calculating prices..."):
            price_info, call_value, put_value, paths_data = calculate_option_prices(
                model_type, current_price, strike_price, time_to_maturity,
                risk_free_rate, volatility, model_params
            )

        # Display option prices
        display_option_prices(price_info)
        
        # Add tabs for different functionalities
        main_tab, strategy_tab, quant_tab = st.tabs(["Basic Analysis", "Strategy Analysis", "Quant Research"])
        
        with main_tab:
            handle_basic_analysis_tab(model_type, current_price, strike_price, time_to_maturity,
                                    risk_free_rate, volatility, paths_data)
        
        with strategy_tab:
            handle_strategy_tab(current_price, strike_price, time_to_maturity,
                              risk_free_rate, volatility, call_value, put_value)
        
        with quant_tab:
            handle_quant_tab(current_price, strike_price, time_to_maturity,
                           risk_free_rate, volatility, call_value, put_value)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your inputs and try again.")
        if st.checkbox("Show detailed error trace", key="show_error"):
            st.exception(e)

if __name__ == "__main__":
    main()
