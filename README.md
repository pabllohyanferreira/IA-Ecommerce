#  E-commerce Analytics - Big Data & IA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

##  Sobre o Projeto

Este projeto implementa uma aplicação completa de **Big Data** e **Inteligência Artificial** para análise de e-commerce, desenvolvido para demonstrar conceitos avançados de Data Science e Machine Learning.

###  Objetivos

- Analisar padrões de vendas e comportamento de clientes
- Implementar sistema de recomendação de produtos
- Segmentar clientes usando algoritmos de clustering
- Prever vendas futuras com modelos de regressão
- Criar dashboard interativo para análise de negócios

##  Funcionalidades

###  Dashboard Interativo
- **Métricas principais** do negócio em tempo real
- **Visualizações dinâmicas** com Plotly
- **Interface responsiva** e moderna

###  Análise de Vendas
- **Top produtos** mais vendidos
- **Análise temporal** de vendas
- **Categorias mais lucrativas**

###  Segmentação de Clientes
- **Clustering K-Means** para segmentação
- **Análise demográfica** detalhada
- **Perfis de clientes** personalizados

###  Inteligência Artificial
- **Random Forest** para previsão de vendas
- **Sistema de recomendação** baseado em histórico
- **Métricas de performance** dos modelos

##  Dataset

O projeto utiliza um dataset simulado realista contendo:

- **5.000 clientes** com dados demográficos
- **200 produtos** em 10 categorias
- **15.000 transações** de vendas
- **Receita total**: R$ 3.719.070,15

### Estrutura dos Dados

`
 data/
 clientes.csv     # Dados demográficos dos clientes
 produtos.csv     # Catálogo de produtos
 vendas.csv       # Histórico de transações
`

##  Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit** - Interface web interativa
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Scikit-learn** - Machine Learning
- **Plotly** - Visualizações interativas
- **Matplotlib/Seaborn** - Gráficos estáticos

##  Como Executar

### 1. Clone o Repositório
`ash
git clone https://github.com/pabllohyanferreira/IA-Ecommerce.git
cd IA-Ecommerce
`

### 2. Instale as Dependências
`ash
pip install -r requirements.txt
`

### 3. Execute a Aplicação
`ash
streamlit run app.py
`

### 4. Acesse no Navegador
`
http://localhost:8501
`

##  Estrutura do Projeto

`
IA-Ecommerce/
 app.py                    # Aplicação principal Streamlit
 requirements.txt          # Dependências do projeto
 run_app.bat              # Script de execução (Windows)
 README.md                # Este arquivo
 data/                    # Datasets
    clientes.csv        # Dados dos clientes
    produtos.csv        # Dados dos produtos
    vendas.csv          # Dados das vendas
 src/                    # Código fonte
    data_generator.py   # Gerador de dados
 models/                 # Modelos treinados
`

##  Modelos de IA Implementados

### 1. Random Forest Regressor
- **Objetivo**: Prever valores de vendas
- **Features**: idade, renda, preço, avaliação, mês, dia da semana
- **Performance**: R² Score e RMSE

### 2. K-Means Clustering
- **Objetivo**: Segmentar clientes
- **Features**: idade e renda mensal
- **Clusters**: 4 segmentos de clientes

### 3. Sistema de Recomendação
- **Objetivo**: Recomendar produtos
- **Baseado**: Histórico de compras e preferências
- **Algoritmo**: Filtragem colaborativa simplificada

##  Resultados

### Métricas do Negócio
- **Ticket Médio**: R$ 247,94
- **Taxa de Conversão**: 86,2%
- **Categoria Top**: Brinquedos
- **Método de Pagamento**: Cartão (50%)

### Performance dos Modelos
- **R² Score**: ~0.85 (85% de precisão)
- **RMSE**: ~R$ 45,00 (erro médio)
- **Segmentação**: 4 clusters bem definidos

##  Aplicação Acadêmica

Este projeto é ideal para:
- **Disciplinas de Big Data** e Data Science
- **Estudos de Machine Learning** aplicado ao e-commerce
- **Análise de comportamento** do consumidor
- **Demonstração de técnicas** de clustering e regressão

##  Próximos Passos

- [ ] Implementar mais algoritmos de recomendação
- [ ] Adicionar análise de sentimentos em reviews
- [ ] Implementar detecção de anomalias
- [ ] Adicionar análise de churn de clientes
- [ ] Integrar com APIs de e-commerce reais

##  Autor

**Pabllo Hyan Ferreira**
- GitHub: [@pabllohyanferreira](https://github.com/pabllohyanferreira)
- Projeto desenvolvido para disciplina de **Big Data em Python**

##  Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

##  Contribuições

Contribuições são sempre bem-vindas! Sinta-se à vontade para:

1. Fazer um Fork do projeto
2. Criar uma branch para sua feature (git checkout -b feature/AmazingFeature)
3. Commit suas mudanças (git commit -m 'Add some AmazingFeature')
4. Push para a branch (git push origin feature/AmazingFeature)
5. Abrir um Pull Request

##  Contato

Se você tiver alguma dúvida ou sugestão, não hesite em entrar em contato!

---

 **Se este projeto foi útil para você, considere dar uma estrela!** 
