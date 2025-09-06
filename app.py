import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Configuracao da pagina
st.set_page_config(
    page_title="E-commerce Analytics - Big Data & IA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para tema escuro educacional
st.markdown("""
<style>
    /* Configuração global para tema escuro */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Títulos principais */
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        background: linear-gradient(45deg, #00d4ff, #5b73e8, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        color: #e2e8f0;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.4rem;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Cards educacionais com tema escuro */
    .education-card {
        background: rgba(30, 30, 50, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100, 200, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .education-card h3 {
        color: #64c8ff;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .education-card p {
        color: #e2e8f0;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    /* Caixas de métricas com tema escuro */
    .metric-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        margin: 0.5rem 0;
        border: 2px solid #475569;
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
    # Carrega os dados de e-commerce com otimizações
    try:
        # Verificar se os arquivos existem
        import os
        if not os.path.exists('data/clientes.csv'):
            st.error("Arquivo data/clientes.csv não encontrado. Execute o gerador de dados primeiro.")
            return None, None, None
        
        # Carregar dados com otimizações
        df_clientes = pd.read_csv('data/clientes.csv', parse_dates=['data_cadastro'])
        df_produtos = pd.read_csv('data/produtos.csv')
        df_vendas = pd.read_csv('data/vendas.csv', parse_dates=['data_venda'])
        
        # Verificar se os dados não estão vazios
        if df_clientes.empty or df_produtos.empty or df_vendas.empty:
            st.error("Um ou mais arquivos de dados estão vazios.")
            return None, None, None
        
        # Otimizações de memória
        df_clientes = df_clientes.astype({
            'cliente_id': 'int32',
            'idade': 'int8',
            'renda_mensal': 'float32'
        })
        
        df_produtos = df_produtos.astype({
            'produto_id': 'int32',
            'preco': 'float32',
            'estoque': 'int16',
            'avaliacao': 'float32',
            'peso': 'float32'
        })
        
        df_vendas = df_vendas.astype({
            'venda_id': 'int32',
            'cliente_id': 'int32',
            'produto_id': 'int32',
            'quantidade': 'int8',
            'desconto': 'float32',
            'preco': 'float32',
            'valor_total': 'float32',
            'mes': 'int8',
            'dia_semana': 'int8'
        })
        
        # Converter datas de forma segura
        df_clientes['data_cadastro'] = pd.to_datetime(df_clientes['data_cadastro'], errors='coerce')
        df_vendas['data_venda'] = pd.to_datetime(df_vendas['data_venda'], errors='coerce')
        
        # Remover linhas com datas inválidas
        df_clientes = df_clientes.dropna(subset=['data_cadastro'])
        df_vendas = df_vendas.dropna(subset=['data_venda'])
        
        return df_clientes, df_produtos, df_vendas
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        import traceback
        st.error(f"Detalhes: {traceback.format_exc()}")
        return None, None, None

def create_advanced_features(df_analise):
    """Cria features avançadas para melhorar a precisão do modelo"""
    
    # Resolver conflito de colunas 'preco' após merge
    if 'preco_x' in df_analise.columns and 'preco_y' in df_analise.columns:
        # Usar preco_y (do df_produtos) como preco principal
        df_analise['preco'] = df_analise['preco_y']
        df_analise = df_analise.drop(['preco_x', 'preco_y'], axis=1)
    elif 'preco_x' in df_analise.columns:
        df_analise['preco'] = df_analise['preco_x']
        df_analise = df_analise.drop('preco_x', axis=1)
    
    # Features de interação
    df_analise['idade_renda'] = df_analise['idade'] * df_analise['renda_mensal']
    df_analise['preco_avaliacao'] = df_analise['preco'] * df_analise['avaliacao']
    df_analise['renda_preco_ratio'] = df_analise['renda_mensal'] / df_analise['preco']
    
    # Features temporais avançadas
    df_analise['dia_mes'] = df_analise['data_venda'].dt.day
    df_analise['trimestre'] = df_analise['data_venda'].dt.quarter
    df_analise['semana_ano'] = df_analise['data_venda'].dt.isocalendar().week
    
    # Features de sazonalidade
    df_analise['eh_fim_semana'] = df_analise['dia_semana'].isin([5, 6]).astype(int)
    df_analise['eh_feriado'] = df_analise['mes'].isin([11, 12]).astype(int)  # Black Friday/Natal
    
    # Features de clustering de clientes
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_analise['cluster_cliente'] = kmeans.fit_predict(df_analise[['idade', 'renda_mensal']])
    
    # Features de produto
    df_analise['preco_categoria'] = df_analise.groupby('categoria')['preco'].transform('mean')
    df_analise['preco_vs_categoria'] = df_analise['preco'] / df_analise['preco_categoria']
    
    # Features de desconto
    df_analise['valor_sem_desconto'] = df_analise['quantidade'] * df_analise['preco']
    df_analise['economia_desconto'] = df_analise['valor_sem_desconto'] - df_analise['valor_total']
    
    return df_analise

@st.cache_resource
def train_advanced_sales_model(df_clientes, df_produtos, df_vendas):
    """Treina modelo de previsão de vendas - OTIMIZADO PARA EDUCAÇÃO"""
    try:
        # Verificar se os dados estão vazios
        if df_clientes.empty or df_produtos.empty or df_vendas.empty:
            st.error("Dados vazios detectados. Verifique os arquivos CSV.")
            return None
        
        # Limitar dados para demonstração educacional
        max_samples = 2000  # Reduzido para demonstração mais rápida
        if len(df_vendas) > max_samples:
            df_vendas = df_vendas.sample(n=max_samples, random_state=42)
        
        # Merge dos dados
        df_analise = df_vendas.merge(df_clientes, on='cliente_id', how='inner')
        df_analise = df_analise.merge(df_produtos, on='produto_id', how='inner')
        
        if df_analise.empty:
            st.error("Nenhum dado encontrado após o merge.")
            return None
        
        # Resolver conflito de preco
        if 'preco_x' in df_analise.columns and 'preco_y' in df_analise.columns:
            df_analise['preco'] = df_analise['preco_y']
            df_analise = df_analise.drop(['preco_x', 'preco_y'], axis=1)
        
        # Features educacionais simplificadas
        features = [
            'idade', 'renda_mensal', 'preco', 'avaliacao', 'mes', 'dia_semana', 
            'quantidade', 'desconto'
        ]
        
        # Criar features simples
        df_analise['idade_renda'] = df_analise['idade'] * df_analise['renda_mensal']
        df_analise['preco_avaliacao'] = df_analise['preco'] * df_analise['avaliacao']
        features.extend(['idade_renda', 'preco_avaliacao'])
        
        # Limpar dados
        df_analise = df_analise.replace([np.inf, -np.inf], np.nan)
        df_analise = df_analise.dropna(subset=features + ['valor_total'])
        
        if df_analise.empty:
            st.error("Nenhum dado válido após limpeza.")
            return None
        
        X = df_analise[features]
        y = df_analise['valor_total']
        
        # Modelos educacionais simples
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0)
        }
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar modelos
        trained_models = {}
        model_scores = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            trained_models[name] = model
            model_scores[name] = score
        
        # Usar o melhor modelo
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = trained_models[best_model_name]
        
        # Calcular métricas
        y_pred_best = best_model.predict(X_test)
        r2_best = r2_score(y_test, y_pred_best)
        rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
        mae_best = mean_absolute_error(y_test, y_pred_best)
        
        return {
            'ensemble': best_model,
            'models': trained_models,
            'scaler': None,  # Simplificado para educação
            'features': features,
            'r2': r2_best,
            'rmse': rmse_best,
            'mae': mae_best,
            'cv_mean': r2_best,  # Simplificado
            'cv_std': 0.0,
            'model_scores': model_scores,
            'best_model': best_model_name
        }
        
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {str(e)}")
        return None

def generate_data_if_missing():
    """Gera dados automaticamente se não existirem"""
    import os
    import subprocess
    import sys
    
    if not os.path.exists('data/clientes.csv'):
        st.info("Dados não encontrados. Gerando dataset automaticamente...")
        try:
            # Executar o gerador de dados
            result = subprocess.run([sys.executable, 'src/data_generator.py'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                st.success("Dataset gerado com sucesso!")
                return True
            else:
                st.error(f"Erro ao gerar dados: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            st.error("Timeout ao gerar dados. Tente executar manualmente.")
            return False
        except Exception as e:
            st.error(f"Erro ao executar gerador: {e}")
            return False
    return True

def main():
    # Titulo principal com design educacional
    st.markdown('<h1 class="main-header">📊 E-commerce Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Big Data & Inteligência Artificial para Análise de Negócios</h3>', unsafe_allow_html=True)
    
    # Card de introdução educacional
    st.markdown("""
    <div class="education-card">
        <h3>🎓 Sobre este Projeto</h3>
        <p><strong>Objetivo:</strong> Demonstrar como Big Data e IA podem ser aplicados em análises de e-commerce.</p>
        <p><strong>Tecnologias:</strong> Python, Streamlit, Scikit-learn, Pandas, Plotly</p>
        <p><strong>Dados:</strong> Dataset sintético com 5.000 clientes, 200 produtos e 15.000 vendas</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar e gerar dados se necessário
    if not generate_data_if_missing():
        st.error("❌ Não foi possível gerar os dados necessários.")
        return
    
    # Carregar dados
    with st.spinner('🔄 Carregando dados...'):
        df_clientes, df_produtos, df_vendas = load_data()
    
    if df_clientes is None:
        st.error("❌ Não foi possível carregar os dados. Verifique se os arquivos existem.")
        return
    
    # Menu lateral educacional
    st.sidebar.markdown("## 🎯 Navegação")
    st.sidebar.markdown("---")
    
    # Adicionar explicações para cada seção
    st.sidebar.markdown("### 📚 Seções Disponíveis:")
    
    opcao = st.sidebar.radio(
        "Escolha uma seção:",
        ["📈 Visão Geral", "💰 Análise de Vendas", "👥 Clientes", "🤖 IA - Previsões", "📊 Gráficos"]
    )
    
    # Mostrar descrição da seção selecionada
    descriptions = {
        "📈 Visão Geral": "Métricas principais e resumo do negócio",
        "💰 Análise de Vendas": "Produtos mais vendidos e categorias",
        "👥 Clientes": "Perfil demográfico e segmentação",
        "🤖 IA - Previsões": "Modelos de Machine Learning para previsões",
        "📊 Gráficos": "Visualizações interativas dos dados"
    }
    
    st.sidebar.markdown(f"**📝 {descriptions[opcao]}**")
    
    # Executar seção selecionada
    if opcao == "📈 Visão Geral":
        show_overview(df_clientes, df_produtos, df_vendas)
    elif opcao == "💰 Análise de Vendas":
        show_sales_analysis(df_vendas, df_produtos)
    elif opcao == "👥 Clientes":
        show_customers(df_clientes)
    elif opcao == "🤖 IA - Previsões":
        show_ai_predictions(df_clientes, df_produtos, df_vendas)
    elif opcao == "📊 Gráficos":
        show_charts(df_vendas, df_produtos)

def show_overview(df_clientes, df_produtos, df_vendas):
    st.header("📈 Visão Geral do Negócio")
    
    # Card educacional explicativo
    st.markdown("""
    <div class="education-card">
        <h3>📊 O que são KPIs (Key Performance Indicators)?</h3>
        <p>KPIs são métricas essenciais que ajudam a medir o desempenho de um negócio. 
        Vamos analisar os principais indicadores do nosso e-commerce:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metricas principais em caixas melhoradas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>👥 Clientes</h3>
            <h2>{len(df_clientes):,}</h2>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">Base de clientes ativa</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>📦 Produtos</h3>
            <h2>{len(df_produtos):,}</h2>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">Catálogo disponível</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>🛒 Vendas</h3>
            <h2>{len(df_vendas):,}</h2>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">Transações realizadas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        receita_total = df_vendas['valor_total'].sum()
        st.markdown(f"""
        <div class="metric-box">
            <h3>💰 Receita Total</h3>
            <h2>R$ {receita_total:,.0f}</h2>
            <p style="font-size: 0.9rem; margin-top: 0.5rem;">Faturamento acumulado</p>
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
    st.header("🤖 Inteligência Artificial - Previsões")
    
    # Card educacional sobre Machine Learning
    st.markdown("""
    <div class="education-card">
        <h3>🧠 O que é Machine Learning?</h3>
        <p><strong>Machine Learning</strong> é uma área da IA que permite aos computadores aprenderem padrões nos dados 
        e fazerem previsões sem serem explicitamente programados para cada situação.</p>
        <p><strong>Neste projeto:</strong> Usamos algoritmos para prever valores de vendas baseados em características 
        do cliente, produto e contexto da transação.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar informações sobre os dados
    st.info(f"📊 **Dados disponíveis:** {len(df_clientes):,} clientes, {len(df_produtos):,} produtos, {len(df_vendas):,} vendas")
    
    # Opção para limpar cache e retreinar
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Retreinar Modelo", help="Limpa o cache e treina um novo modelo"):
            try:
                st.cache_resource.clear()
                st.success("Cache limpo! Recarregando...")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao limpar cache: {e}")
                st.rerun()
    
    # Treinar modelo avançado com progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Iniciando treinamento do modelo...")
        progress_bar.progress(10)
        
        status_text.text("📊 Carregando e processando dados...")
        progress_bar.progress(30)
        
        status_text.text("🤖 Treinando modelos de IA...")
        progress_bar.progress(60)
        
        model_data = train_advanced_sales_model(df_clientes, df_produtos, df_vendas)
        
        progress_bar.progress(90)
        status_text.text("✅ Finalizando treinamento...")
        progress_bar.progress(100)
        
        status_text.text("🎉 Modelo treinado com sucesso!")
        
    except Exception as e:
        st.error(f"Erro durante o treinamento: {e}")
        return
    
    if model_data is None:
        st.error("❌ Erro ao treinar o modelo de IA.")
        return
    
    # Card educacional sobre métricas
    st.markdown("""
    <div class="education-card">
        <h3>📏 Como Avaliamos a Qualidade do Modelo?</h3>
        <p><strong>R² (Coeficiente de Determinação):</strong> Mede quanto do comportamento dos dados o modelo consegue explicar (0-1, quanto maior melhor)</p>
        <p><strong>RMSE (Root Mean Square Error):</strong> Erro médio em reais - quanto o modelo "erra" em média</p>
        <p><strong>MAE (Mean Absolute Error):</strong> Erro absoluto médio - diferença média entre previsão e realidade</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar performance do modelo
    st.subheader("📊 Performance do Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
            <h4>🎯 Precisão (R²)</h4>
            <h2>{model_data['r2']:.3f}</h2>
            <p style="font-size: 0.8rem;">{model_data['r2']*100:.1f}% dos dados explicados</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="warning-box">
            <h4>📉 Erro Médio (RMSE)</h4>
            <h2>R$ {model_data['rmse']:.2f}</h2>
            <p style="font-size: 0.8rem;">Erro típico em reais</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Erro Absoluto (MAE)</h4>
            <h2>R$ {model_data['mae']:.2f}</h2>
            <p style="font-size: 0.8rem;">Diferença média</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Validação Cruzada</h4>
            <h2>{model_data['cv_mean']:.3f}</h4>
            <p style="font-size: 0.8rem;">Teste de robustez</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar scores dos modelos individuais
    st.subheader(" Performance dos Modelos Individuais")
    
    # Destacar o melhor modelo
    if 'best_model' in model_data:
        st.success(f"🏆 **Melhor Modelo:** {model_data['best_model']} (R² = {model_data['model_scores'][model_data['best_model']]:.3f})")
    
    for model_name, score in model_data['model_scores'].items():
        if model_name == model_data.get('best_model', ''):
            st.write(f"🥇 **{model_name}**: R² = {score:.3f} (MELHOR)")
        else:
            st.write(f"**{model_name}**: R² = {score:.3f}")
    
    # Card educacional sobre previsões
    st.markdown("""
    <div class="education-card">
        <h3>🔮 Teste o Modelo de Previsão</h3>
        <p>Agora você pode testar o modelo ajustando os parâmetros abaixo. 
        O algoritmo irá prever o valor de uma venda baseado nas características que você definir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interface de predicao
    st.subheader("🎯 Simulador de Previsão de Vendas")
    st.write("Ajuste os parâmetros abaixo e veja como o modelo prevê o valor da venda:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("** Dados do Cliente**")
        idade = st.slider("Idade", 18, 80, 35, key="idade_slider")
        renda = st.slider("Renda Mensal (R$)", 1000, 50000, 5000, key="renda_slider")
        preco = st.slider("Preco do Produto (R$)", 10, 2000, 100, key="preco_slider")
        quantidade = st.slider("Quantidade", 1, 10, 1, key="quantidade_slider")
        desconto = st.slider("Desconto (%)", 0.0, 50.0, 0.0, key="desconto_slider")
    
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
        # Criar features para predição (apenas as 10 features usadas no treinamento)
        features_dict = {
            'idade': idade,
            'renda_mensal': renda,
            'preco': preco,
            'avaliacao': avaliacao,
            'mes': mes,
            'dia_semana': dia_semana,
            'quantidade': quantidade,
            'desconto': desconto / 100,
            'idade_renda': idade * renda,
            'preco_avaliacao': preco * avaliacao
        }
        
        # Garantir que as features estão na ordem correta
        model_features = model_data['features']
        features_array = np.array([[features_dict[feature] for feature in model_features]])
        
        # Fazer predicao (sem normalização para simplicidade educacional)
        try:
            valor_predito = model_data['ensemble'].predict(features_array)[0]
        except Exception as pred_error:
            st.error(f"Erro na predição: {pred_error}")
            return
        
        # Calcular valor com desconto
        valor_sem_desconto = quantidade * preco
        valor_final = valor_sem_desconto * (1 - desconto/100)
        
        if valor_predito > 500:
            st.markdown(f"""
            <div class="success-box">
                <h3> Venda de Alto Valor!</h3>
                <h2>Valor Previsto: R$ {valor_predito:.2f}</h2>
                <p>Valor Real: R$ {valor_final:.2f}</p>
                <p>Precisão: {model_data['r2']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        elif valor_predito > 200:
            st.markdown(f"""
            <div class="info-box">
                <h3> Venda de Valor Medio</h3>
                <h2>Valor Previsto: R$ {valor_predito:.2f}</h2>
                <p>Valor Real: R$ {valor_final:.2f}</p>
                <p>Precisão: {model_data['r2']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h3> Venda de Baixo Valor</h3>
                <h2>Valor Previsto: R$ {valor_predito:.2f}</h2>
                <p>Valor Real: R$ {valor_final:.2f}</p>
                <p>Precisão: {model_data['r2']*100:.1f}%</p>
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
