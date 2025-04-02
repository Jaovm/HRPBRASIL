import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import riskfolio as rp
import matplotlib.pyplot as plt

# Função para obter dados
@st.cache_data
def get_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)
    
    if df.empty:
        st.error("Erro ao obter os dados. Verifique os tickers e a conexão com a internet.")
        return None
    
    if 'Adj Close' not in df:
        st.error("Dados retornados sem a coluna 'Adj Close'. Verifique os tickers e tente novamente.")
        return None
    
    return df['Adj Close']

# Interface do usuário
st.title('Otimização de Portfólio com Hierarchical Risk Parity (HRP)')

# Entrada de dados
tickers_default = "AGRO3.SA BBAS3.SA BBSE3.SA BPAC11.SA EGIE3.SA ITUB3.SA PRIO3.SA PSSA3.SA SAPR3.SA SBSP3.SA VIVT3.SA WEGE3.SA TOTS3.SA B3SA3.SA TAEE3.SA"
tickers = st.text_area('Digite os tickers separados por espaço', tickers_default).split()
data_inicio = st.date_input("Data de início", value=pd.to_datetime("2019-01-01"))
data_fim = st.date_input("Data de fim", value=pd.to_datetime("2024-01-01"))
n_portfolios = st.number_input("Número de simulações", min_value=100, max_value=100000, value=5000, step=100)

if st.button('Otimizar Portfólio'):
    df = get_data(tickers, data_inicio, data_fim)
    if df is None:
        st.stop()
    
    returns = df.pct_change().dropna()
    
    # Otimização HRP
    port = rp.HCPortfolio(returns=returns)
    weights = port.optimization(model='HRP', codependence='pearson', rm='MV', linkage='ward', leaf_order=True)
    
    st.subheader("Pesos da Carteira Otimizada (HRP)")
    st.write(weights)
    
    # Simulação de carteiras aleatórias para maximizar Sharpe
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    results = np.zeros((3, n_portfolios))
    weights_record = []
    
    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(len(tickers)))
        weights_record.append(w)
        port_return = np.dot(w, mean_returns)
        port_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe_ratio = port_return / port_volatility
        results[0, i] = port_return
        results[1, i] = port_volatility
        results[2, i] = sharpe_ratio
    
    max_sharpe_idx = np.argmax(results[2])
    best_weights = weights_record[max_sharpe_idx]
    best_allocation = pd.Series(best_weights, index=tickers)
    
    st.subheader("Melhor Carteira pelo Índice de Sharpe")
    st.write(best_allocation)
    
    # Plot Fronteira Eficiente
    fig, ax = plt.subplots()
    scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
    ax.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], c='red', marker='*', s=200, label='Melhor Sharpe')
    ax.set_xlabel('Volatilidade')
    ax.set_ylabel('Retorno Esperado')
    ax.set_title('Fronteira Eficiente')
    ax.legend()
    fig.colorbar(scatter, label='Sharpe Ratio')
    st.pyplot(fig)


