import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuracao da pagina
st.set_page_config(
    page_title="E-commerce Analytics - Big Data & IA",
    page_icon="",
    layout="wide"
)

# CSS personalizado com cores melhoradas e tema escuro consistente
st.markdown("""
<style>
    /* Configuração global para tema escuro */
    .stApp {
        background-color: #1a1a1a;
        color: white;
    }
    
    /* Títulos principais */
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .sub-header {
        color: #e2e8f0;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Caixas de métricas com cores contrastantes */
    .metric-box {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        margin: 0.5rem 0;
        border: 2px solid #718096;
    }
    .metric-box h3 {
        color: #f7fafc;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .metric-box h2 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Caixas de informação */
    .info-box {
        background: linear-gradient(135deg, #2b6cb0 0%, #2c5282 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        border: 2px solid #4299e1;
    }
    .info-box h4 {
        color: #f7fafc;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .info-box p {
        color: white;
        margin: 0.5rem 0;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Caixas de sucesso */
    .success-box {
        background: linear-gradient(135deg, #22543d 0%, #2f855a 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        border: 2px solid #68d391;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Caixas de aviso */
    .warning-box {
        background: linear-gradient(135deg, #c05621 0%, #dd6b20 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        border: 2px solid #f6ad55;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* CSS FORÇADO para caixas de seleção - tema escuro */
    .stSelectbox > div > div {
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #718096 !important;
        background-color: #4a5568 !important;
    }
    
    /* Forçar fundo escuro na caixa principal do selectbox */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        color: white !important;
    }
    
    /* Forçar fundo escuro no container interno */
    div[data-testid="stSelectbox"] > div > div > div {
        background-color: #2d3748 !important;
        color: white !important;
    }
    
    /* Forçar fundo escuro no texto selecionado */
    div[data-testid="stSelectbox"] > div > div > div > div {
        background-color: #2d3748 !important;
        color: white !important;
    }
    
    /* Forçar fundo escuro no dropdown quando aberto */
    div[data-testid="stSelectbox"] > div > div > div > div > div {
        background-color: #2d3748 !important;
        color: white !important;
    }
    
    /* Forçar fundo escuro nas opções do dropdown */
    div[data-testid="stSelectbox"] > div > div > div > div > div > div {
        background-color: #2d3748 !important;
        color: white !important;
    }
    
    /* CSS adicional para garantir que funcione */
    .stSelectbox div[role="combobox"] {
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        color: white !important;
    }
    
    .stSelectbox div[role="combobox"]:hover {
        background-color: #4a5568 !important;
        border-color: #718096 !important;
    }
    
    /* Sliders com tema escuro */
    .stSlider > div > div > div {
        background-color: #4a5568 !important;
    }
    
    /* Melhorar contraste dos textos */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
    }
    
    .stMarkdown p {
        color: #e2e8f0 !important;
    }
    
    /* Labels dos widgets */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        text-transform: capitalize;
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1rem;
    }
    
    /* Melhorar sidebar */
    .css-1d391kg {
        background-color: #2d3748 !important;
    }
    
    /* Melhorar tabelas */
    .stDataFrame {
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 8px;
        color: white !important;
    }
    
    /* Botões */
    .stButton > button {
        background-color: #e53e3e !important;
        color: white !important;
        border: 2px solid #c53030 !important;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #c53030 !important;
        border-color: #e53e3e !important;
    }
    
    /* Títulos das seções */
    .stMarkdown h1 {
        color: #ffffff !important;
        border-bottom: 3px solid #4a5568;
        padding-bottom: 10px;
    }
    .stMarkdown h2 {
        color: #e2e8f0 !important;
        border-bottom: 2px solid #718096;
        padding-bottom: 8px;
    }
    .stMarkdown h3 {
        color: #e2e8f0 !important;
    }
    
    /* Radio buttons do sidebar */
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
    
    /* Título do sidebar */
    .css-1d391kg h1 {
        color: #ffffff !important;
    }
    
    /* Valores dos sliders */
    .stSlider > div > div > div > div {
        color: #ffffff !important;
    }
    
    /* CSS adicional para forçar o tema escuro nos selectboxes */
    .stSelectbox [data-baseweb="select"] {
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        background-color: #4a5568 !important;
        border-color: #718096 !important;
    }
    
    /* Forçar cor do texto selecionado */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #2d3748 !important;
        color: white !important;
    }
    
    /* Forçar cor das opções do dropdown */
    .stSelectbox [data-baseweb="select"] [role="option"] {
        background-color: #2d3748 !important;
        color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] [role="option"]:hover {
        background-color: #4a5568 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Carrega os dados de e-commerce
    try:
        df_clientes = pd.read_csv('data/clientes.csv')
        df_produtos = pd.read_csv('data/produtos.csv')
        df_vendas = pd.read_csv('data/vendas.csv')
        
        # Converter datas de forma segura
        df_clientes['data_cadastro'] = pd.to_datetime(df_clientes['data_cadastro'], errors='coerce')
        df_vendas['data_venda'] = pd.to_datetime(df_vendas['data_venda'], errors='coerce')
        
        return df_clientes, df_produtos, df_vendas
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None

@st.cache_resource
def train_sales_model(df_clientes, df_produtos, df_vendas):
    # Treina modelo simples de previsao de vendas
    try:
        # Merge dos dados
        df_analise = df_vendas.merge(df_clientes, on='cliente_id')
        df_analise = df_analise.merge(df_produtos[['produto_id', 'avaliacao']], on='produto_id')
        
        # Features para o modelo
        features = ['idade', 'renda_mensal', 'preco', 'avaliacao', 'mes', 'dia_semana']
        X = df_analise[features]
        y = df_analise['valor_total']
        
        # Treinar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        modelo = RandomForestRegressor(n_estimators=50, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        # Calcular metricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return modelo, r2, rmse, features
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {e}")
        return None, 0, 0, []

def main():
    # Titulo principal
    st.markdown('<h1 class="main-header"> E-commerce Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Big Data & Inteligencia Artificial</h3>', unsafe_allow_html=True)
    
    # Carregar dados
    with st.spinner('Carregando dados...'):
        df_clientes, df_produtos, df_vendas = load_data()
    
    if df_clientes is None:
        st.error("Nao foi possivel carregar os dados. Verifique se os arquivos existem.")
        return
    
    # Menu lateral simples
    st.sidebar.title(" Menu")
    st.sidebar.markdown("---")
    opcao = st.sidebar.radio(
        "Escolha uma opcao:",
        [" Visao Geral", " Analise de Vendas", " Clientes", " IA - Previsoes", " Graficos"]
    )
    
    if opcao == " Visao Geral":
        show_overview(df_clientes, df_produtos, df_vendas)
    elif opcao == " Analise de Vendas":
        show_sales_analysis(df_vendas, df_produtos)
    elif opcao == " Clientes":
        show_customers(df_clientes)
    elif opcao == " IA - Previsoes":
        show_ai_predictions(df_clientes, df_produtos, df_vendas)
    elif opcao == " Graficos":
        show_charts(df_vendas, df_produtos)

def show_overview(df_clientes, df_produtos, df_vendas):
    st.header(" Visao Geral do Negocio")
    
    # Metricas principais em caixas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3> Clientes</h3>
            <h2>{len(df_clientes):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3> Produtos</h3>
            <h2>{len(df_produtos):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3> Vendas</h3>
            <h2>{len(df_vendas):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        receita_total = df_vendas['valor_total'].sum()
        st.markdown(f"""
        <div class="metric-box">
            <h3> Receita</h3>
            <h2>R$ {receita_total:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Informacoes importantes
    st.markdown("""
    <div class="info-box">
        <h4> Resumo do Negocio</h4>
        <p> <strong>Ticket Medio:</strong> R$ {:.2f}</p>
        <p> <strong>Melhor Categoria:</strong> {}</p>
        <p> <strong>Metodo de Pagamento Preferido:</strong> {}</p>
        <p> <strong>Periodo de Maior Venda:</strong> {}</p>
    </div>
    """.format(
        df_vendas['valor_total'].mean(),
        df_produtos['categoria'].value_counts().index[0],
        df_vendas['metodo_pagamento'].value_counts().index[0],
        df_vendas['data_venda'].dt.month.value_counts().index[0]
    ), unsafe_allow_html=True)
    
    # Grafico simples de vendas por mes
    st.subheader(" Vendas por Mes")
    df_vendas['mes_ano'] = df_vendas['data_venda'].dt.to_period('M')
    vendas_mes = df_vendas.groupby('mes_ano')['valor_total'].sum().reset_index()
    vendas_mes['mes_ano'] = vendas_mes['mes_ano'].astype(str)
    
    fig = px.line(vendas_mes, x='mes_ano', y='valor_total', 
                 title='Evolucao das Vendas',
                 labels={'valor_total': 'Receita (R$)', 'mes_ano': 'Mes'},
                 color_discrete_sequence=['#4299e1'])
    fig.update_layout(
        plot_bgcolor='#2d3748',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff',
        title_font_color='#ffffff',
        title_font_size=18,
        xaxis=dict(
            color='#e2e8f0', 
            gridcolor='#4a5568',
            linecolor='#718096',
            title_font_color='#ffffff'
        ),
        yaxis=dict(
            color='#e2e8f0', 
            gridcolor='#4a5568',
            linecolor='#718096',
            title_font_color='#ffffff'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def show_sales_analysis(df_vendas, df_produtos):
    st.header(" Analise de Vendas")
    
    # Top produtos
    st.subheader(" Top 10 Produtos Mais Vendidos")
    df_analise = df_vendas.merge(df_produtos, on='produto_id')
    top_produtos = df_analise.groupby(['produto_id', 'nome', 'categoria'])['valor_total'].sum().reset_index()
    top_produtos = top_produtos.sort_values('valor_total', ascending=False).head(10)
    
    # Tabela simples
    for i, (_, produto) in enumerate(top_produtos.iterrows(), 1):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4c51bf 0%, #553c9a 100%); 
                    color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
                    border: 2px solid #a5b4fc; box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    font-weight: 600;">
            <strong>{i}.</strong> {produto['nome']} - {produto['categoria']}<br>
             R$ {produto['valor_total']:,.2f}
        </div>
        """, unsafe_allow_html=True)
    
    # Categorias mais vendidas
    st.subheader(" Categorias Mais Vendidas")
    receita_categoria = df_analise.groupby('categoria')['valor_total'].sum().reset_index()
    receita_categoria = receita_categoria.sort_values('valor_total', ascending=False)
    
    fig = px.bar(receita_categoria, x='categoria', y='valor_total',
                title='Receita por Categoria',
                labels={'valor_total': 'Receita (R$)', 'categoria': 'Categoria'},
                color_discrete_sequence=['#38a169', '#2f855a', '#22543d', '#4c51bf', '#553c9a'])
    fig.update_layout(
        xaxis_tickangle=45,
        plot_bgcolor='#2d3748',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff',
        title_font_color='#ffffff',
        title_font_size=18,
        xaxis=dict(
            color='#e2e8f0', 
            gridcolor='#4a5568',
            linecolor='#718096',
            title_font_color='#ffffff'
        ),
        yaxis=dict(
            color='#e2e8f0', 
            gridcolor='#4a5568',
            linecolor='#718096',
            title_font_color='#ffffff'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def show_customers(df_clientes):
    st.header(" Analise de Clientes")
    
    # Estatisticas basicas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Idade dos Clientes")
        fig = px.histogram(df_clientes, x='idade', nbins=20, 
                          title='Distribuicao de Idades',
                          labels={'idade': 'Idade', 'count': 'Quantidade'},
                          color_discrete_sequence=['#4299e1'])
        fig.update_layout(
            plot_bgcolor='#2d3748',
            paper_bgcolor='#1a1a1a',
            font_color='#ffffff',
            title_font_color='#ffffff',
            title_font_size=16,
            xaxis=dict(
                color='#e2e8f0', 
                gridcolor='#4a5568',
                linecolor='#718096',
                title_font_color='#ffffff'
            ),
            yaxis=dict(
                color='#e2e8f0', 
                gridcolor='#4a5568',
                linecolor='#718096',
                title_font_color='#ffffff'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Renda dos Clientes")
        fig = px.histogram(df_clientes, x='renda_mensal', nbins=20,
                          title='Distribuicao de Renda',
                          labels={'renda_mensal': 'Renda Mensal (R$)', 'count': 'Quantidade'},
                          color_discrete_sequence=['#38a169'])
        fig.update_layout(
            plot_bgcolor='#2d3748',
            paper_bgcolor='#1a1a1a',
            font_color='#ffffff',
            title_font_color='#ffffff',
            title_font_size=16,
            xaxis=dict(
                color='#e2e8f0', 
                gridcolor='#4a5568',
                linecolor='#718096',
                title_font_color='#ffffff'
            ),
            yaxis=dict(
                color='#e2e8f0', 
                gridcolor='#4a5568',
                linecolor='#718096',
                title_font_color='#ffffff'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tipos de clientes
    st.subheader(" Tipos de Clientes")
    tipo_clientes = df_clientes['tipo_cliente'].value_counts()
    fig = px.pie(values=tipo_clientes.values, names=tipo_clientes.index,
                title='Distribuicao por Tipo de Cliente',
                color_discrete_sequence=['#f6ad55', '#ed8936', '#4c51bf'])
    fig.update_layout(
        plot_bgcolor='#2d3748',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff',
        title_font_color='#ffffff',
        title_font_size=16
    )
    st.plotly_chart(fig, use_container_width=True)

def show_ai_predictions(df_clientes, df_produtos, df_vendas):
    st.header(" Inteligencia Artificial - Previsoes")
    
    # Treinar modelo
    with st.spinner('Treinando modelo de IA...'):
        modelo, r2, rmse, features = train_sales_model(df_clientes, df_produtos, df_vendas)
    
    if modelo is None:
        st.error("Erro ao treinar o modelo de IA.")
        return
    
    # Mostrar performance do modelo
    st.subheader(" Performance do Modelo")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
            <h4>Precisao (R²)</h4>
            <h2>{r2:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="warning-box">
            <h4>Erro Medio (RMSE)</h4>
            <h2>R$ {rmse:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Interface de predicao
    st.subheader(" Prever Valor de Venda")
    st.write("Preencha os dados abaixo para prever o valor de uma venda:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("** Dados do Cliente**")
        idade = st.slider("Idade", 18, 80, 35, key="idade_slider")
        renda = st.slider("Renda Mensal (R$)", 1000, 50000, 5000, key="renda_slider")
        preco = st.slider("Preco do Produto (R$)", 10, 2000, 100, key="preco_slider")
    
    with col2:
        st.write("** Dados do Produto**")
        avaliacao = st.slider("Avaliacao", 1.0, 5.0, 4.0, 0.1, key="avaliacao_slider")
        
        # Mês com nomes
        meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
        mes_selecionado = st.selectbox("Mes", meses, index=5, key="mes_select")
        mes = meses.index(mes_selecionado) + 1  # Converter para número (1-12)
        
        # Dia da semana com nomes
        dias_semana = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 
                      'Sexta-feira', 'Sábado', 'Domingo']
        dia_selecionado = st.selectbox("Dia da Semana", dias_semana, index=0, key="dia_select")
        dia_semana = dias_semana.index(dia_selecionado)  # Converter para número (0-6)
    
    if st.button(" Prever Venda", type="primary"):
        # Fazer predicao
        entrada = [[idade, renda, preco, avaliacao, mes, dia_semana]]
        valor_predito = modelo.predict(entrada)[0]
        
        if valor_predito > 500:
            st.markdown(f"""
            <div class="success-box">
                <h3> Venda de Alto Valor!</h3>
                <h2>Valor Previsto: R$ {valor_predito:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        elif valor_predito > 200:
            st.markdown(f"""
            <div class="info-box">
                <h3> Venda de Valor Medio</h3>
                <h2>Valor Previsto: R$ {valor_predito:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h3> Venda de Baixo Valor</h3>
                <h2>Valor Previsto: R$ {valor_predito:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

def show_charts(df_vendas, df_produtos):
    st.header(" Graficos e Visualizacoes")
    
    # Vendas por dia da semana
    st.subheader(" Vendas por Dia da Semana")
    df_vendas['dia_semana_nome'] = df_vendas['data_venda'].dt.day_name()
    vendas_dia = df_vendas.groupby('dia_semana_nome')['valor_total'].sum().reset_index()
    
    fig = px.bar(vendas_dia, x='dia_semana_nome', y='valor_total',
                title='Vendas por Dia da Semana',
                labels={'valor_total': 'Receita (R$)', 'dia_semana_nome': 'Dia'},
                color_discrete_sequence=['#4299e1', '#38a169', '#f6ad55', '#e53e3e', '#4c51bf', '#553c9a', '#ed8936'])
    fig.update_layout(
        plot_bgcolor='#2d3748',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff',
        title_font_color='#ffffff',
        title_font_size=16,
        xaxis=dict(
            color='#e2e8f0', 
            gridcolor='#4a5568',
            linecolor='#718096',
            title_font_color='#ffffff'
        ),
        yaxis=dict(
            color='#e2e8f0', 
            gridcolor='#4a5568',
            linecolor='#718096',
            title_font_color='#ffffff'
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Metodos de pagamento
    st.subheader(" Metodos de Pagamento")
    vendas_pagamento = df_vendas.groupby('metodo_pagamento')['valor_total'].sum().reset_index()
    
    fig = px.pie(vendas_pagamento, values='valor_total', names='metodo_pagamento',
                title='Vendas por Metodo de Pagamento',
                color_discrete_sequence=['#4299e1', '#38a169', '#f6ad55', '#e53e3e'])
    fig.update_layout(
        plot_bgcolor='#2d3748',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff',
        title_font_color='#ffffff',
        title_font_size=16
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Avaliacoes
    st.subheader(" Avaliacoes dos Produtos")
    fig = px.histogram(df_produtos, x='avaliacao', nbins=20,
                      title='Distribuicao das Avaliacoes',
                      labels={'avaliacao': 'Avaliacao', 'count': 'Quantidade'},
                      color_discrete_sequence=['#f6ad55'])
    fig.update_layout(
        plot_bgcolor='#2d3748',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff',
        title_font_color='#ffffff',
        title_font_size=16,
        xaxis=dict(
            color='#e2e8f0', 
            gridcolor='#4a5568',
            linecolor='#718096',
            title_font_color='#ffffff'
        ),
        yaxis=dict(
            color='#e2e8f0', 
            gridcolor='#4a5568',
            linecolor='#718096',
            title_font_color='#ffffff'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
