import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Options Pricing Models",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching your design
st.markdown("""
<style>
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
    /* Add more custom styling from previous version */
</style>
""", unsafe_allow_html=True)

# Model Selection in sidebar
st.sidebar.markdown("## Model Selection")
model_type = st.sidebar.selectbox(
    "",
    ["Black-Scholes", "Binomial", "Monte Carlo"],
    index=0
)

# Title changes based on model
st.markdown(f"# 📈 {model_type} Model")

# Author section
st.markdown("Created by:")
st.markdown("""
<a href="https://www.linkedin.com/in/navnoorbawa/" target="_blank" style="text-decoration: none; color: inherit;">
    <div style="display: flex; align-items: center; background-color: #1E2530; padding: 10px; border-radius: 5px; width: fit-content; cursor: pointer;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="margin-right: 10px;">
        <span style="color: #FFFFFF;">Navnoor Bawa</span>
    </div>
</a>
""", unsafe_allow_html=True)

# Common input parameters
input_data = pd.DataFrame({
    "Current Asset Price": [0],
    "Strike Price": ["100.0000"],
    "Time to Maturity (Years)": ["1.0000"],
    "Volatility (σ)": ["0.2000"],
    "Risk-Free Interest Rate": ["0.0500"]
}, index=[""])

st.table(input_data)

# Basic input parameters
current_price = st.sidebar.number_input("Current Asset Price", value=100.00, step=0.01, format="%.2f")
strike_price = st.sidebar.number_input("Strike Price", value=100.00, step=0.01, format="%.2f")
time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.00, step=0.01, format="%.2f")
volatility = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01, format="%.2f")
risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01, format="%.2f")

# Model-specific parameters
if model_type == "Binomial":
    steps = st.sidebar.slider("Number of Steps", 10, 1000, 100)
    option_style = st.sidebar.selectbox("Option Style", ["European", "American"])
elif model_type == "Monte Carlo":
    n_simulations = st.sidebar.slider("Number of Simulations", 1000, 50000, 10000)
    n_steps = st.sidebar.slider("Time Steps", 50, 500, 100)

# Model implementations (add your existing model code here)
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

# Calculate prices based on selected model
if model_type == "Black-Scholes":
    call_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                  risk_free_rate, volatility, 'call')
    put_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                 risk_free_rate, volatility, 'put')
    price_info = {"call": f"${call_value:.2f}", "put": f"${put_value:.2f}"}

elif model_type == "Binomial":
    call_value = binomial_calc(current_price, strike_price, time_to_maturity,
                             risk_free_rate, volatility, steps, 'call', option_style.lower())
    put_value = binomial_calc(current_price, strike_price, time_to_maturity,
                            risk_free_rate, volatility, steps, 'put', option_style.lower())
    price_info = {"call": f"${call_value:.2f}", "put": f"${put_value:.2f}"}

else:  # Monte Carlo
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

# Heatmap section (continuing with your existing heatmap code)
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Heatmap Parameters in sidebar
st.sidebar.markdown("## Heatmap Parameters")
min_spot = st.sidebar.number_input("Min Spot Price", value=80.00, step=0.01)
max_spot = st.sidebar.number_input("Max Spot Price", value=120.00, step=0.01)

# Volatility sliders
min_vol = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.00, 0.10)
max_vol = st.sidebar.slider("Max Volatility for Heatmap", 0.01, 1.00, 0.30)

# Generate heatmap data
spot_prices = np.linspace(min_spot, max_spot, 10)
volatilities = np.linspace(min_vol, max_vol, 10)
call_prices = np.zeros((len(volatilities), len(spot_prices)))
put_prices = np.zeros((len(volatilities), len(spot_prices)))

for i, vol in enumerate(volatilities):
    for j, spot in enumerate(spot_prices):
        if model_type == "Black-Scholes":
            call_prices[i, j] = black_scholes_calc(spot, strike_price, time_to_maturity, risk_free_rate, vol, 'call')
            put_prices[i, j] = black_scholes_calc(spot, strike_price, time_to_maturity, risk_free_rate, vol, 'put')
        elif model_type == "Binomial":
            call_prices[i, j] = binomial_calc(spot, strike_price, time_to_maturity, risk_free_rate, vol, steps, 'call', option_style.lower())
            put_prices[i, j] = binomial_calc(spot, strike_price, time_to_maturity, risk_free_rate, vol, steps, 'put', option_style.lower())
        else:  # Monte Carlo
            call_price, _, _ = monte_carlo_calc(spot, strike_price, time_to_maturity, risk_free_rate, vol, n_simulations, n_steps, 'call')
            put_price, _, _ = monte_carlo_calc(spot, strike_price, time_to_maturity, risk_free_rate, vol, n_simulations, n_steps, 'put')
            call_prices[i, j] = call_price
            put_prices[i, j] = put_price

# Create heatmap visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Price Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        call_prices,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        ax=ax1,
        xticklabels=[f'{p:.2f}' for p in spot_prices],
        yticklabels=[f'{v:.2f}' for v in volatilities]
    )
    ax1.set_xlabel('Spot Price')
    ax1.set_ylabel('Volatility')
    plt.title('CALL')
    st.pyplot(fig1)

with col2:
    st.subheader("Put Price Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        put_prices,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        ax=ax2,
        xticklabels=[f'{p:.2f}' for p in spot_prices],
        yticklabels=[f'{v:.2f}' for v in volatilities]
    )
    ax2.set_xlabel('Spot Price')
    ax2.set_ylabel('Volatility')
    plt.title('PUT')
    st.pyplot(fig2)

# Add Monte Carlo specific visualizations if selected
if model_type == "Monte Carlo":
    st.subheader("Monte Carlo Simulation Paths")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, time_to_maturity, n_steps), call_paths[:100].T, alpha=0.1)
    plt.plot(np.linspace(0, time_to_maturity, n_steps), np.mean(call_paths, axis=0), 'r', linewidth=2)
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.title('Monte Carlo Simulation Paths (first 100 paths)')
    st.pyplot(fig)
