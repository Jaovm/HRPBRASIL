import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Função para baixar dados históricos
def get_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data.dropna(axis=1, how='all')

# Função para calcular retornos diários
def get_returns(data):
    returns = data.pct_change().dropna()
    return returns.dropna(axis=1, thresh=int(returns.shape[0] * 0.8))

# Função para calcular a volatilidade de cada ativo
def calculate_volatility(returns):
    return returns.std()

# Função para otimizar os pesos usando Risk Parity
def risk_parity_allocation(returns):
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)
    init_guess = np.repeat(1/num_assets, num_assets)
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    def risk_contributions(weights, cov_matrix):
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_risk = cov_matrix @ weights
        risk_contributions = weights * marginal_risk / portfolio_volatility
        return np.std(risk_contributions)
    
    result = minimize(risk_contributions, init_guess, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return pd.Series(result.x, index=returns.columns)

# Interface no Streamlit
st.title('Otimização de Carteira com Risk Parity')

default_tickers = ['ITUB3.SA', 'B3SA3.SA', 'WEGE3.SA', 'PETR4.SA', 'VALE3.SA']
tickers = st.text_input('Digite os tickers separados por vírgula:', ', '.join(default_tickers)).split(', ')
start_date = st.text_input("Data de início (YYYY-MM-DD)", "2015-01-01")
end_date = st.text_input("Data de fim (YYYY-MM-DD)", pd.to_datetime("today").strftime('%Y-%m-%d'))

if st.button('Calcular alocação Risk Parity'):
    data = get_stock_data(tickers, start_date, end_date)
    st.write("### Dados Brutos")
    st.write(data.tail())
    
    returns = get_returns(data)
    st.write("### Retornos Calculados")
    st.write(returns.head())
    
    if returns.shape[1] < 2:
        st.warning("Poucos ativos com dados disponíveis. Tente adicionar mais tickers ou ampliar o período de análise.")
    else:
        weights = risk_parity_allocation(returns)
        st.write('### Pesos da Carteira Otimizada')
        st.write(weights)
