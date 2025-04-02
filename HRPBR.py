import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

# Função para baixar dados históricos
def get_stock_data(tickers, start='2019-01-01'):
    data = yf.download(tickers, start=start, end='2025-04-01')['Adj Close']
    data = data.dropna(axis=1, how='all')  # Remove ativos sem histórico suficiente
    return data

# Função para calcular a matriz de correlação
def get_correlation_matrix(returns):
    return returns.corr()

# Função para calcular a matriz de covariância shrinkage
def get_covariance_matrix(returns):
    returns = returns.dropna()  # Remove NaN
    if returns.shape[0] < 2 or returns.shape[1] < 2:  # Garante que há dados suficientes
        raise ValueError("Dados insuficientes para calcular a matriz de covariância.")
    
    lw = LedoitWolf()
    cov_matrix = lw.fit(returns).covariance_
    return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

# Função para alocação Hierarchical Risk Parity
def hrp_allocation(returns):
    cov_matrix = get_covariance_matrix(returns)
    corr_matrix = get_correlation_matrix(returns)
    dist_matrix = np.sqrt((1 - corr_matrix) / 2)
    linkage = sch.linkage(squareform(dist_matrix), method='ward')
    sorted_indices = sch.leaves_list(linkage)
    ordered_cov = cov_matrix.iloc[sorted_indices, sorted_indices]
    inv_variance = 1 / np.diag(ordered_cov)
    weights = inv_variance / np.sum(inv_variance)
    return pd.Series(weights, index=returns.columns[sorted_indices])

# Interface no Streamlit
st.title('Otimização de Carteira com Hierarchical Risk Parity')

# Seleção de ativos
default_tickers = ['ITUB3.SA', 'B3SA3.SA', 'WEGE3.SA', 'PETR4.SA', 'VALE3.SA']
tickers = st.text_input('Digite os tickers separados por vírgula:', ', '.join(default_tickers)).split(', ')

# Baixando dados
if st.button('Calcular alocação HRP'):
    data = get_stock_data(tickers)
    returns = data.pct_change().dropna()
    
    if returns.shape[0] < 2 or returns.shape[1] < 2:
        st.warning("Dados insuficientes para calcular a matriz de covariância. Tente adicionar mais ativos ou aumentar o período de análise.")
    else:
        weights = hrp_allocation(returns)
        st.write('### Pesos da Carteira Otimizada')
        st.write(weights)
        
        # Plotando o dendrograma
        corr_matrix = get_correlation_matrix(returns)
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        linkage = sch.linkage(squareform(dist_matrix), method='ward')
        
        plt.figure(figsize=(10, 5))
        sch.dendrogram(linkage, labels=returns.columns, leaf_rotation=90)
        plt.title('Dendrograma de Clustering')
        st.pyplot(plt)
