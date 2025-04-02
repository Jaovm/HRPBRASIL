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
    tickers = [t if t.endswith(".SA") else t + ".SA" for t in tickers]
    data = yf.download(tickers, start=start, end=end)
    if data.empty:
        raise ValueError("Nenhum dado foi baixado. Verifique os tickers e as datas.")
    if 'Adj Close' in data.columns:
        data = data['Adj Close']
    elif 'Close' in data.columns:
        data = data['Close']
    else:
        raise KeyError("Os dados não contêm 'Adj Close' ou 'Close'. Verifique os tickers.")
    data = data.dropna(axis=1, how='all')  # Remove ativos sem dados
    if data.shape[1] == 0:
        raise ValueError("Nenhum ativo contém dados suficientes para a análise.")
    return data

# Função para calcular a matriz de correlação e a distância
def get_correlation_distance(data):
    returns = data.pct_change().dropna()
    if returns.shape[1] < 2:
        raise ValueError("Dados insuficientes para calcular correlações.")
    correlation_matrix = returns.corr()
    distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
    np.fill_diagonal(distance_matrix.values, 0)  # Garantindo que a diagonal seja zero
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Garantindo simetria
    if np.isnan(distance_matrix).any():
        raise ValueError("A matriz de distância contém valores inválidos. Verifique os dados de entrada.")
    return correlation_matrix, distance_matrix

# Função para aplicar HRP
def hierarchical_risk_parity(data):
    _, distance_matrix = get_correlation_distance(data)
    if np.all(distance_matrix == 0):
        raise ValueError("A matriz de distância está vazia ou inválida.")
    dist_linkage = sch.linkage(ssd.squareform(distance_matrix, checks=False), method='ward')
    dendro = sch.dendrogram(dist_linkage, no_plot=True)
    sorted_indices = dendro['leaves']
    inv_volatility = 1 / data.pct_change().std()
    weights = inv_volatility / inv_volatility.sum()
    sorted_weights = weights.iloc[sorted_indices]
    sorted_weights /= sorted_weights.sum()
    return sorted_weights.sort_index()

# Simulação de carteiras aleatórias para comparação
def simulate_random_portfolios(data, num_portfolios):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    results = np.zeros((3, num_portfolios))
    all_weights = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(data.columns))
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        all_weights.append(weights)
    
    max_sharpe_idx = np.argmax(results[2])
    min_risk_idx = np.argmin(results[1])
    best_sharpe_weights = all_weights[max_sharpe_idx]
    lowest_risk_weights = all_weights[min_risk_idx]
    
    return results, best_sharpe_weights, lowest_risk_weights

# Interface do Streamlit
st.title("Hierarchical Risk Parity para Ações Brasileiras")

tickers = st.text_area("Insira os tickers separados por espaço", "B3SA3 BBAS3 ITUB3 PETR4 VALE3 WEGE3")
tickers = tickers.split()

start_date = st.date_input("Data de início", pd.to_datetime("2020-01-01"))
end_date = st.date_input("Data de fim", pd.to_datetime("2024-01-01"))
num_simulations = st.number_input("Número de simulações de portfólio", min_value=1000, max_value=50000, value=10000, step=1000)

if st.button("Analisar Portfólio"):
    try:
        data = get_data(tickers, start_date, end_date)
        hrp_weights = hierarchical_risk_parity(data)
        
        st.subheader("Pesos da carteira HRP")
        st.dataframe(hrp_weights)
        
        _, distance_matrix = get_correlation_distance(data)
        dist_linkage = sch.linkage(ssd.squareform(distance_matrix, checks=False), method='ward')
        fig, ax = plt.subplots(figsize=(10, 5))
        sch.dendrogram(dist_linkage, labels=data.columns, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Simulação de Portfólios Aleatórios")
        results, best_sharpe_weights, lowest_risk_weights = simulate_random_portfolios(data, num_simulations)
        
        fig, ax = plt.subplots()
        scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.7)
        ax.set_xlabel('Volatilidade')
        ax.set_ylabel('Retorno Esperado')
        fig.colorbar(scatter, label='Sharpe Ratio')
        st.pyplot(fig)
        
        st.subheader("Carteira com Melhor Sharpe")
        st.dataframe(pd.Series(best_sharpe_weights, index=data.columns, name="Pesos"))
        
        st.subheader("Carteira com Menor Risco")
        st.dataframe(pd.Series(lowest_risk_weights, index=data.columns, name="Pesos"))
        
    except Exception as e:
        st.error(f"Erro ao processar a análise: {e}")
