import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf
from scipy.stats import norm
import math
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Option Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .option-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .greeks-header {
        background: linear-gradient(45deg, #ff6b6b, #ffa500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .sidebar .stRadio > div {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black-Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Tree Model'

class BlackScholesCalculator:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to expiration (in years)
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        
    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def put_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
    
    # Greeks calculations
    def delta_call(self):
        return norm.cdf(self.d1())
    
    def delta_put(self):
        return norm.cdf(self.d1()) - 1
    
    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta_call(self):
        d1 = self.d1()
        d2 = self.d2()
        return (-(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) 
                - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) / 365
    
    def theta_put(self):
        d1 = self.d1()
        d2 = self.d2()
        return (-(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) 
                + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) / 365
    
    def vega(self):
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T) / 100
    
    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2()) / 100
    
    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) / 100

class MonteCarloOptionPricing:
    def __init__(self, S, K, T, r, sigma, num_simulations=10000):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_simulations = num_simulations
        self.price_paths = None
        
    def simulate_price_paths(self):
        dt = self.T / 252  # Daily time step
        num_steps = int(self.T * 252)
        
        # Generate random price paths
        Z = np.random.standard_normal((num_steps, self.num_simulations))
        price_paths = np.zeros((num_steps + 1, self.num_simulations))
        price_paths[0] = self.S
        
        for t in range(1, num_steps + 1):
            price_paths[t] = price_paths[t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z[t-1]
            )
        
        self.price_paths = price_paths
        return price_paths
    
    def call_price(self):
        if self.price_paths is None:
            self.simulate_price_paths()
        
        final_prices = self.price_paths[-1]
        payoffs = np.maximum(final_prices - self.K, 0)
        return np.exp(-self.r * self.T) * np.mean(payoffs)
    
    def put_price(self):
        if self.price_paths is None:
            self.simulate_price_paths()
        
        final_prices = self.price_paths[-1]
        payoffs = np.maximum(self.K - final_prices, 0)
        return np.exp(-self.r * self.T) * np.mean(payoffs)

class BinomialTreeOptionPricing:
    def __init__(self, S, K, T, r, sigma, num_steps=100):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.num_steps = num_steps
        
    def calculate_parameters(self):
        dt = self.T / self.num_steps
        u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.r * dt) - d) / (u - d)  # Risk-neutral probability
        return u, d, p, dt
    
    def call_price(self):
        u, d, p, dt = self.calculate_parameters()
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(self.num_steps + 1)
        for i in range(self.num_steps + 1):
            asset_prices[i] = self.S * (u ** (self.num_steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.maximum(asset_prices - self.K, 0)
        
        # Step backwards through tree
        for j in range(self.num_steps - 1, -1, -1):
            for i in range(j + 1):
                option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
        
        return option_values[0]
    
    def put_price(self):
        u, d, p, dt = self.calculate_parameters()
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(self.num_steps + 1)
        for i in range(self.num_steps + 1):
            asset_prices[i] = self.S * (u ** (self.num_steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.maximum(self.K - asset_prices, 0)
        
        # Step backwards through tree
        for j in range(self.num_steps - 1, -1, -1):
            for i in range(j + 1):
                option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
        
        return option_values[0]

@st.cache_data
def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            data = stock.history(period="5d")
        return float(data['Close'].iloc[-1]) if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching current price for {ticker}: {str(e)}")
        return None

def calculate_historical_volatility(data, window=252):
    """Calculate historical volatility from price data"""
    returns = np.log(data['Close'] / data['Close'].shift(1))
    return returns.rolling(window=window).std() * np.sqrt(252)

def create_price_chart(data, ticker):
    """Create an interactive price chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_payoff_diagram(S, K_list, option_type='call'):
    """Create option payoff diagram"""
    fig = go.Figure()
    
    spot_range = np.linspace(min(K_list) * 0.7, max(K_list) * 1.3, 100)
    
    for K in K_list:
        if option_type == 'call':
            payoff = np.maximum(spot_range - K, 0)
            name = f'Call K={K}'
        else:
            payoff = np.maximum(K - spot_range, 0)
            name = f'Put K={K}'
        
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=payoff,
            mode='lines',
            name=name,
            line=dict(width=2)
        ))
    
    # Add current stock price line
    fig.add_vline(x=S, line_dash="dash", line_color="red", annotation_text=f"Current Price: ${S:.2f}")
    
    fig.update_layout(
        title=f'{option_type.capitalize()} Option Payoff Diagram',
        xaxis_title='Stock Price at Expiration',
        yaxis_title='Payoff',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_greeks_heatmap(S_range, T_range, K, r, sigma, greek_type='delta'):
    """Create Greeks heatmap"""
    greek_values = np.zeros((len(T_range), len(S_range)))
    
    for i, T in enumerate(T_range):
        for j, S in enumerate(S_range):
            if T > 0:
                bs = BlackScholesCalculator(S, K, T, r, sigma)
                if greek_type == 'delta':
                    greek_values[i, j] = bs.delta_call()
                elif greek_type == 'gamma':
                    greek_values[i, j] = bs.gamma()
                elif greek_type == 'theta':
                    greek_values[i, j] = bs.theta_call()
                elif greek_type == 'vega':
                    greek_values[i, j] = bs.vega()
    
    fig = go.Figure(data=go.Heatmap(
        z=greek_values,
        x=S_range,
        y=T_range,
        colorscale='RdBu',
        colorbar=dict(title=greek_type.capitalize())
    ))
    
    fig.update_layout(
        title=f'{greek_type.capitalize()} Heatmap',
        xaxis_title='Stock Price',
        yaxis_title='Time to Expiration (Years)',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_volatility_smile(K_range, S, T, r, market_prices, option_type='call'):
    """Create implied volatility smile"""
    implied_vols = []
    
    for K in K_range:
        # This is a simplified approach - in reality, you'd solve for implied vol
        # For demonstration, we'll create a synthetic smile
        moneyness = K / S
        if option_type == 'call':
            # Synthetic volatility smile
            implied_vol = 0.2 + 0.1 * (moneyness - 1)**2
        else:
            implied_vol = 0.2 + 0.15 * (moneyness - 1)**2
        
        implied_vols.append(implied_vol)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=K_range,
        y=implied_vols,
        mode='lines+markers',
        name='Implied Volatility',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_vline(x=S, line_dash="dash", line_color="blue", annotation_text=f"ATM: ${S:.2f}")
    
    fig.update_layout(
        title='Implied Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility',
        template='plotly_white',
        height=400
    )
    
    return fig

# Main App
st.markdown('<h1 class="main-header">🚀 Advanced Option Pricing Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## ⚙️ Configuration")

pricing_method = st.sidebar.selectbox(
    '🎯 Select Pricing Model',
    options=[model.value for model in OPTION_PRICING_MODEL],
    help="Choose the mathematical model for option pricing"
)

st.sidebar.markdown("---")

# Input parameters
st.sidebar.markdown("### 📊 Market Parameters")

ticker = st.sidebar.text_input('🏢 Ticker Symbol', 'AAPL', help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)")

# Get current price and data
current_price = get_current_price(ticker)
stock_data = get_stock_data(ticker)

if current_price is not None:
    st.sidebar.success(f"Current Price: ${current_price:.2f}")
    default_strike = round(current_price, 2)
    min_strike = max(0.1, round(current_price * 0.5, 2))
    max_strike = round(current_price * 2, 2)
else:
    st.sidebar.error("Unable to fetch current price")
    default_strike = 100.0
    min_strike = 50.0
    max_strike = 200.0

strike_price = st.sidebar.number_input(
    '🎯 Strike Price ($)',
    min_value=min_strike,
    max_value=max_strike,
    value=default_strike,
    step=0.01,
    help="The price at which the option can be exercised"
)

risk_free_rate = st.sidebar.slider(
    '📈 Risk-Free Rate (%)',
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
    help="Current risk-free interest rate"
) / 100

# Calculate historical volatility if data is available
if stock_data is not None and not stock_data.empty:
    hist_vol = calculate_historical_volatility(stock_data).iloc[-1]
    default_vol = min(max(hist_vol * 100, 10), 100) if not np.isnan(hist_vol) else 25.0
else:
    default_vol = 25.0

volatility = st.sidebar.slider(
    '📊 Volatility (%)',
    min_value=1.0,
    max_value=150.0,
    value=float(default_vol),
    step=0.5,
    help="Annual volatility of the underlying asset"
) / 100

expiration_date = st.sidebar.date_input(
    '📅 Expiration Date',
    min_value=datetime.today() + timedelta(days=1),
    value=datetime.today() + timedelta(days=30),
    help="Option expiration date"
)

days_to_expiry = (expiration_date - datetime.today().date()).days
time_to_expiry = days_to_expiry / 365.0

# Model-specific parameters
if pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    st.sidebar.markdown("### 🎲 Monte Carlo Parameters")
    num_simulations = st.sidebar.selectbox(
        'Number of Simulations',
        options=[1000, 5000, 10000, 50000, 100000],
        index=2,
        help="More simulations = higher accuracy but slower computation"
    )

elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
    st.sidebar.markdown("### 🌳 Binomial Tree Parameters")
    num_steps = st.sidebar.selectbox(
        'Number of Steps',
        options=[50, 100, 200, 500, 1000],
        index=2,
        help="More steps = higher accuracy but slower computation"
    )

st.sidebar.markdown("---")

# Advanced features toggle
show_greeks = st.sidebar.checkbox('📈 Show Greeks Analysis', value=True)
show_payoff = st.sidebar.checkbox('📊 Show Payoff Diagram', value=True)
show_vol_smile = st.sidebar.checkbox('😊 Show Volatility Smile', value=False)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if stock_data is not None and not stock_data.empty:
        st.subheader(f"📈 {ticker} Stock Price Chart")
        price_chart = create_price_chart(stock_data, ticker)
        st.plotly_chart(price_chart, use_container_width=True)
    
    # Calculate option prices
    if st.button('🚀 Calculate Option Prices', type='primary'):
        with st.spinner('Calculating option prices...'):
            try:
                if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
                    bs = BlackScholesCalculator(current_price, strike_price, time_to_expiry, risk_free_rate, volatility)
                    call_price = bs.call_price()
                    put_price = bs.put_price()
                    
                    # Calculate Greeks
                    greeks = {
                        'Call Delta': bs.delta_call(),
                        'Put Delta': bs.delta_put(),
                        'Gamma': bs.gamma(),
                        'Call Theta': bs.theta_call(),
                        'Put Theta': bs.theta_put(),
                        'Vega': bs.vega(),
                        'Call Rho': bs.rho_call(),
                        'Put Rho': bs.rho_put()
                    }
                
                elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
                    mc = MonteCarloOptionPricing(current_price, strike_price, time_to_expiry, risk_free_rate, volatility, num_simulations)
                    call_price = mc.call_price()
                    put_price = mc.put_price()
                    
                    # Approximate Greeks using finite differences
                    bs = BlackScholesCalculator(current_price, strike_price, time_to_expiry, risk_free_rate, volatility)
                    greeks = {
                        'Call Delta': bs.delta_call(),
                        'Put Delta': bs.delta_put(),
                        'Gamma': bs.gamma(),
                        'Call Theta': bs.theta_call(),
                        'Put Theta': bs.theta_put(),
                        'Vega': bs.vega(),
                        'Call Rho': bs.rho_call(),
                        'Put Rho': bs.rho_put()
                    }
                
                elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
                    bt = BinomialTreeOptionPricing(current_price, strike_price, time_to_expiry, risk_free_rate, volatility, num_steps)
                    call_price = bt.call_price()
                    put_price = bt.put_price()
                    
                    # Approximate Greeks
                    bs = BlackScholesCalculator(current_price, strike_price, time_to_expiry, risk_free_rate, volatility)
                    greeks = {
                        'Call Delta': bs.delta_call(),
                        'Put Delta': bs.delta_put(),
                        'Gamma': bs.gamma(),
                        'Call Theta': bs.theta_call(),
                        'Put Theta': bs.theta_put(),
                        'Vega': bs.vega(),
                        'Call Rho': bs.rho_call(),
                        'Put Rho': bs.rho_put()
                    }
                
                # Display results
                st.markdown("## 💰 Option Prices")
                
                col_call, col_put = st.columns(2)
                
                with col_call:
                    st.markdown("""
                    <div class="option-card">
                        <h3 style="color: #28a745; margin-bottom: 1rem;">🟢 Call Option</h3>
                        <h2 style="color: #28a745; margin: 0;">${:.2f}</h2>
                    </div>
                    """.format(call_price), unsafe_allow_html=True)
                
                with col_put:
                    st.markdown("""
                    <div class="option-card">
                        <h3 style="color: #dc3545; margin-bottom: 1rem;">🔴 Put Option</h3>
                        <h2 style="color: #dc3545; margin: 0;">${:.2f}</h2>
                    </div>
                    """.format(put_price), unsafe_allow_html=True)
                
                                # Greeks Analysis
                if show_greeks:
                    st.markdown('<div class="greeks-header">🎯 Greeks Analysis</div>', unsafe_allow_html=True)
                    
                    # Create custom styled Greeks display
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0;">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center; backdrop-filter: blur(10px);">
                                <h4 style="color: #3498db; margin: 0;">Call Delta</h4>
                                <h3 style="color: white; margin: 0.5rem 0;">{:.4f}</h3>
                                <p style="color: #bdc3c7; font-size: 0.8rem; margin: 0;">Price sensitivity</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center; backdrop-filter: blur(10px);">
                                <h4 style="color: #e74c3c; margin: 0;">Put Delta</h4>
                                <h3 style="color: white; margin: 0.5rem 0;">{:.4f}</h3>
                                <p style="color: #bdc3c7; font-size: 0.8rem; margin: 0;">Price sensitivity</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center; backdrop-filter: blur(10px);">
                                <h4 style="color: #f39c12; margin: 0;">Gamma</h4>
                                <h3 style="color: white; margin: 0.5rem 0;">{:.4f}</h3>
                                <p style="color: #bdc3c7; font-size: 0.8rem; margin: 0;">Delta sensitivity</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center; backdrop-filter: blur(10px);">
                                <h4 style="color: #9b59b6; margin: 0;">Vega</h4>
                                <h3 style="color: white; margin: 0.5rem 0;">{:.4f}</h3>
                                <p style="color: #bdc3c7; font-size: 0.8rem; margin: 0;">Volatility sensitivity</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center; backdrop-filter: blur(10px);">
                                <h4 style="color: #1abc9c; margin: 0;">Call Theta</h4>
                                <h3 style="color: white; margin: 0.5rem 0;">{:.4f}</h3>
                                <p style="color: #bdc3c7; font-size: 0.8rem; margin: 0;">Time decay/day</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center; backdrop-filter: blur(10px);">
                                <h4 style="color: #e67e22; margin: 0;">Put Theta</h4>
                                <h3 style="color: white; margin: 0.5rem 0;">{:.4f}</h3>
                                <p style="color: #bdc3c7; font-size: 0.8rem; margin: 0;">Time decay/day</p>
                            </div>
                        </div>
                    </div>
                    """.format(
                        greeks['Call Delta'],
                        greeks['Put Delta'], 
                        greeks['Gamma'],
                        greeks['Vega'],
                        greeks['Call Theta'],
                        greeks['Put Theta']
                    ), unsafe_allow_html=True)
                
                # Payoff Diagram
                if show_payoff:
                    st.markdown("## 📊 Payoff Diagrams")
                    
                    payoff_col1, payoff_col2 = st.columns(2)
                    
                    with payoff_col1:
                        call_payoff_fig = create_payoff_diagram(current_price, [strike_price], 'call')
                        st.plotly_chart(call_payoff_fig, use_container_width=True)
                    
                    with payoff_col2:
                        put_payoff_fig = create_payoff_diagram(current_price, [strike_price], 'put')
                        st.plotly_chart(put_payoff_fig, use_container_width=True)
                
                # Volatility Smile
                if show_vol_smile:
                    st.markdown("## 😊 Volatility Smile")
                    strike_range = np.linspace(current_price * 0.8, current_price * 1.2, 20)
                    vol_smile_fig = create_volatility_smile(strike_range, current_price, time_to_expiry, risk_free_rate, [], 'call')
                    st.plotly_chart(vol_smile_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating option prices: {str(e)}")

with col2:
    st.markdown("### 📋 Summary")
    
    # Input summary
    st.markdown(f"""
    **Current Stock Price:** ${current_price:.2f}  
    **Strike Price:** ${strike_price:.2f}  
    **Days to Expiry:** {days_to_expiry}  
    **Risk-Free Rate:** {risk_free_rate*100:.1f}%  
    **Volatility:** {volatility*100:.1f}%  
    **Pricing Model:** {pricing_method}
    """)
    
    # Moneyness indicator
    if current_price is not None:
        moneyness = current_price / strike_price
        if moneyness > 1.05:
            st.success("📈 Call ITM / Put OTM")
        elif moneyness < 0.95:
            st.error("📉 Call OTM / Put ITM")
        else:
            st.warning("⚖️ Near ATM")
    
    # Market data
    # Market data
if stock_data is not None and not stock_data.empty:
    st.markdown("### 📊 Market Data")
    
    # Calculate basic stats
    returns = stock_data['Close'].pct_change()
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: white; margin: 0; font-size: 0.9rem;">52W High</h4>
                <h3 style="color: #fff; margin: 0.5rem 0;">${:.2f}</h3>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: white; margin: 0; font-size: 0.9rem;">52W Low</h4>
                <h3 style="color: #fff; margin: 0.5rem 0;">${:.2f}</h3>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: white; margin: 0; font-size: 0.9rem;">Avg Volume</h4>
                <h3 style="color: #fff; margin: 0.5rem 0;">{:.0f}</h3>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: white; margin: 0; font-size: 0.9rem;">Historical Vol</h4>
                <h3 style="color: #fff; margin: 0.5rem 0;">{:.1f}%</h3>
            </div>
        </div>
    </div>
    """.format(
        stock_data['High'].max(),
        stock_data['Low'].min(),
        stock_data['Volume'].mean(),
        volatility*100
    ), unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>⚠️ This tool is for educational purposes only. Not financial advice.</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)