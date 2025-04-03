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
    
    # Verifica quais colunas estão disponíveis
    st.write("Colunas disponíveis nos dados:", df.columns)
    
    # Se 'Adj Close' não estiver presente, verifica se 'Close' está disponível
    if 'Adj Close' in df:
        return df['Adj Close']
    elif 'Close' in df:
        return df['Close']
    else:
        st.error("Dados retornados sem as colunas esperadas ('Adj Close' ou 'Close'). Verifique os tickers e tente novamente.")
        return None

# Interface do usuário
st.title('Otimização de Portfólio com Hierarchical Risk Parity (HRP)')

# Entrada de dados
tickers_default = "AGRO3.SA BBAS3.SA BBSE3.SA BPAC11.SA EGIE3.SA ITUB3.SA PRIO3.SA PSSA3.SA SAPR3.SA SBSP3.SA VIVT3.SA WEGE3.SA TOTS3.SA B3SA3.SA TAEE3.SA"
tickers = st.text_area('Digite os tickers separados por espaço', tickers_default).split()
data_inicio = st.date_input("Data de início", value=pd.to_datetime("2018-01-01"))
data_fim = st.date_input("Data de fim", value=pd.to_datetime("2025-04-01"))
n_portfolios = st.number_input("Número de simulações", min_value=100, max_value=10000000, value=10000, step=1000)
limite_min_alocacao = st.slider("Limite mínimo de alocação por ativo (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
limite_max_alocacao = st.slider("Limite máximo de alocação por ativo (%)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)

if st.button('Otimizar Portfólio'):
    df = get_data(tickers, data_inicio, data_fim)
    if df is None:
        st.stop()
    
    returns = df.pct_change().dropna()
    
    # Tratamento de dados para evitar erros na otimização
    if returns.isnull().values.any():
        st.error("Os dados contêm valores nulos. Verifique os tickers e o período selecionado.")
        st.stop()
    
    returns = returns.dropna(axis=1, how='all')  # Remove ativos sem dados suficientes
    
    if returns.shape[1] < 2:
        st.error("Não há dados suficientes para a otimização. Verifique os tickers.")
        st.stop()
    
    st.write("Dados de retorno disponíveis:", returns.describe())
    
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
        w = np.random.uniform(limite_min_alocacao, limite_max_alocacao, len(tickers))
        w /= w.sum()
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
    
    # Backtest das carteiras HRP, Melhor Sharpe e Ibovespa
    ibov = get_data(["^BVSP"], data_inicio, data_fim)
    if ibov is not None:
        ibov_returns = ibov.pct_change().dropna()
        hrp_returns = (returns * weights.values.flatten()).sum(axis=1)
        sharpe_returns = (returns * best_weights).sum(axis=1)
        
        # Criar um DataFrame com os retornos acumulados
        backtest = pd.DataFrame({
            'HRP': (1 + hrp_returns).cumprod(),
            'Melhor Sharpe': (1 + sharpe_returns).cumprod()
        })
        
        # Adicionar o Ibovespa ao backtest apenas se ele não estiver vazio
        if not ibov_returns.empty:
            backtest['Ibovespa'] = (1 + ibov_returns).cumprod()
        
        # Alinhar os índices para evitar problemas de tamanhos diferentes
        backtest = backtest.dropna()
        
        st.subheader("Backtest das Estratégias")
        st.line_chart(backtest)
        
        # Cálculo de métricas de desempenho
        cagr = lambda x: (x.iloc[-1] / x.iloc[0]) ** (1 / len(x.index.year.unique())) - 1
        vol_anu = lambda x: x.pct_change().std() * np.sqrt(252)
        
        metrics = pd.DataFrame({
            'CAGR': backtest.apply(cagr),
            'Volatilidade Anual': backtest.apply(vol_anu)
        })
        
        st.subheader("Métricas das Estratégias")
        st.write(metrics)
