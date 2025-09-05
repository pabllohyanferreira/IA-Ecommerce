import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def criar_dataset_ecommerce():
    # Cria um dataset realista de e-commerce
    np.random.seed(42)
    
    # Parametros do dataset
    n_clientes = 5000
    n_produtos = 200
    n_vendas = 15000
    
    print("Gerando dataset de E-commerce...")
    
    # 1. Dataset de Clientes
    print("Criando dados de clientes...")
    clientes = {
        'cliente_id': range(1, n_clientes + 1),
        'idade': np.random.normal(35, 12, n_clientes).astype(int),
        'genero': np.random.choice(['M', 'F'], n_clientes, p=[0.48, 0.52]),
        'cidade': np.random.choice([
            'Sao Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 
            'Brasilia', 'Fortaleza', 'Manaus', 'Curitiba', 'Recife', 'Porto Alegre'
        ], n_clientes),
        'renda_mensal': np.random.lognormal(8, 0.5, n_clientes),
        'tipo_cliente': np.random.choice(['Novo', 'Regular', 'VIP'], n_clientes, p=[0.3, 0.6, 0.1]),
        'data_cadastro': pd.date_range('2020-01-01', '2024-01-01', periods=n_clientes)
    }
    
    df_clientes = pd.DataFrame(clientes)
    df_clientes['renda_mensal'] = np.clip(df_clientes['renda_mensal'], 1000, 50000)
    
    # 2. Dataset de Produtos
    print("Criando dados de produtos...")
    categorias = [
        'Eletronicos', 'Roupas', 'Casa e Jardim', 'Esportes', 'Livros',
        'Beleza', 'Alimentacao', 'Automotivo', 'Brinquedos', 'Saude'
    ]
    
    produtos = {
        'produto_id': range(1, n_produtos + 1),
        'nome': [f'Produto_{i}' for i in range(1, n_produtos + 1)],
        'categoria': np.random.choice(categorias, n_produtos),
        'preco': np.random.lognormal(4, 0.8, n_produtos),
        'estoque': np.random.poisson(50, n_produtos),
        'avaliacao': np.random.normal(4.2, 0.8, n_produtos),
        'peso': np.random.normal(1.5, 0.8, n_produtos)
    }
    
    df_produtos = pd.DataFrame(produtos)
    df_produtos['preco'] = np.clip(df_produtos['preco'], 10, 2000)
    df_produtos['avaliacao'] = np.clip(df_produtos['avaliacao'], 1, 5)
    df_produtos['peso'] = np.clip(df_produtos['peso'], 0.1, 10)
    
    # 3. Dataset de Vendas
    print("Criando dados de vendas...")
    
    # Gerar datas de vendas (ultimos 2 anos)
    data_inicio = datetime(2022, 1, 1)
    data_fim = datetime(2024, 1, 1)
    datas_vendas = pd.date_range(data_inicio, data_fim, periods=n_vendas)
    
    vendas = {
        'venda_id': range(1, n_vendas + 1),
        'cliente_id': np.random.choice(df_clientes['cliente_id'], n_vendas),
        'produto_id': np.random.choice(df_produtos['produto_id'], n_vendas),
        'data_venda': datas_vendas,
        'quantidade': np.random.poisson(2, n_vendas) + 1,
        'desconto': np.random.choice([0, 0.05, 0.1, 0.15, 0.2], n_vendas, p=[0.4, 0.3, 0.2, 0.08, 0.02]),
        'metodo_pagamento': np.random.choice(['Cartao', 'PIX', 'Boleto', 'PayPal'], n_vendas, p=[0.5, 0.3, 0.15, 0.05]),
        'avaliacao_compra': np.random.choice([1, 2, 3, 4, 5], n_vendas, p=[0.02, 0.05, 0.15, 0.35, 0.43])
    }
    
    df_vendas = pd.DataFrame(vendas)
    
    # Calcular valor total da venda
    df_vendas = df_vendas.merge(df_produtos[['produto_id', 'preco']], on='produto_id')
    df_vendas['valor_total'] = df_vendas['quantidade'] * df_vendas['preco'] * (1 - df_vendas['desconto'])
    
    # Adicionar sazonalidade
    df_vendas['mes'] = df_vendas['data_venda'].dt.month
    df_vendas['dia_semana'] = df_vendas['data_venda'].dt.dayofweek
    
    # Aumentar vendas em novembro/dezembro (Black Friday/Natal)
    black_friday_mask = df_vendas['mes'].isin([11, 12])
    df_vendas.loc[black_friday_mask, 'valor_total'] *= 1.3
    
    # Aumentar vendas nos fins de semana
    fim_semana_mask = df_vendas['dia_semana'].isin([5, 6])
    df_vendas.loc[fim_semana_mask, 'valor_total'] *= 1.1
    
    print("Dataset criado com sucesso!")
    
    return df_clientes, df_produtos, df_vendas

def analisar_dataset(df_clientes, df_produtos, df_vendas):
    # Realiza analise basica do dataset
    print("\n" + "="*60)
    print("ANALISE DO DATASET DE E-COMMERCE")
    print("="*60)
    
    print(f"\nClientes: {len(df_clientes):,}")
    print(f"Produtos: {len(df_produtos):,}")
    print(f"Vendas: {len(df_vendas):,}")
    
    print(f"\nReceita Total: R$ {df_vendas['valor_total'].sum():,.2f}")
    print(f"Ticket Medio: R$ {df_vendas['valor_total'].mean():.2f}")
    print(f"Vendas por Mes: {len(df_vendas) / 24:.0f}")
    
    print(f"\nTop 5 Categorias:")
    top_categorias = df_produtos['categoria'].value_counts().head()
    for cat, count in top_categorias.items():
        print(f"   {cat}: {count} produtos")
    
    print(f"\nAvaliacao Media dos Produtos: {df_produtos['avaliacao'].mean():.2f}")
    print(f"Avaliacao Media das Compras: {df_vendas['avaliacao_compra'].mean():.2f}")
    
    return True

def salvar_datasets(df_clientes, df_produtos, df_vendas):
    # Salva os datasets em arquivos CSV
    os.makedirs('data', exist_ok=True)
    
    print("\nSalvando datasets...")
    
    df_clientes.to_csv('data/clientes.csv', index=False)
    df_produtos.to_csv('data/produtos.csv', index=False)
    df_vendas.to_csv('data/vendas.csv', index=False)
    
    print("Datasets salvos em:")
    print("   data/clientes.csv")
    print("   data/produtos.csv")
    print("   data/vendas.csv")

def main():
    print("GERADOR DE DATASET DE E-COMMERCE")
    print("="*50)
    
    # Criar datasets
    df_clientes, df_produtos, df_vendas = criar_dataset_ecommerce()
    
    # Analisar datasets
    analisar_dataset(df_clientes, df_produtos, df_vendas)
    
    # Salvar datasets
    salvar_datasets(df_clientes, df_produtos, df_vendas)
    
    print("\nDataset de E-commerce criado com sucesso!")
    print("Execute 'streamlit run app.py' para visualizar a aplicacao.")

if __name__ == "__main__":
    main()
