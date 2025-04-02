import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf

# Função para obter os dados históricos
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

# Função para calcular a matriz de correlação e a distância
def get_correlation_distance(data):
    returns = data.pct_change().dropna()
    correlation_matrix = returns.corr()
    distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
    return correlation_matrix, distance_matrix

# Função para aplicar HRP
def hierarchical_risk_parity(data):
    _, distance_matrix = get_correlation_distance(data)
    dist_linkage = sch.linkage(ssd.squareform(distance_matrix), method='ward')
    dendro = sch.dendrogram(dist_linkage, no_plot=True)
    sorted_indices = dendro['leaves']
    inv_volatility = 1 / data.pct_change().std()
    weights = inv_volatility / inv_volatility.sum()
    sorted_weights = weights.iloc[sorted_indices]
    sorted_weights /= sorted_weights.sum()
    return sorted_weights.sort_index()

# Interface do Streamlit
st.title("Hierarchical Risk Parity para Ações Brasileiras")

tickers = st.text_area("Insira os tickers separados por espaço", "B3SA3 BBAS3 ITUB3 PETR4 VALE3 WEGE3")
tickers = tickers.split()

start_date = st.date_input("Data de início", pd.to_datetime("2020-01-01"))
end_date = st.date_input("Data de fim", pd.to_datetime("2024-01-01"))

if st.button("Analisar Portfólio"):
    data = get_data(tickers, start_date, end_date)
    hrp_weights = hierarchical_risk_parity(data)
    
    st.subheader("Pesos da carteira HRP")
    st.dataframe(hrp_weights)
    
    # Plot do dendrograma
    _, distance_matrix = get_correlation_distance(data)
    dist_linkage = sch.linkage(ssd.squareform(distance_matrix), method='ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    sch.dendrogram(dist_linkage, labels=tickers, ax=ax)
    st.pyplot(fig)
