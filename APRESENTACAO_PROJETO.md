# E-commerce Analytics - Big Data & IA
## Apresentação para Sala de Aula

---

## 1. INTRODUÇÃO AO PROJETO

### O que é este projeto?
- **Aplicação web** que analisa dados de uma loja online
- **Dashboard interativo** para visualizar informações de negócio
- **Inteligência Artificial** para prever vendas e segmentar clientes

### Por que é importante?
- **Big Data**: Processa milhares de registros de vendas
- **IA**: Aprende padrões e faz previsões
- **Negócio**: Ajuda a tomar decisões baseadas em dados

---

## 2. DADOS DO PROJETO

### Quantos dados temos?
- **5.000 clientes** com informações pessoais
- **200 produtos** em 10 categorias diferentes
- **15.000 vendas** realizadas
- **Receita total**: R$ 3.719.070,15

### Tipos de informações:
- **Clientes**: idade, gênero, cidade, renda, tipo de cliente
- **Produtos**: nome, categoria, preço, estoque, avaliação
- **Vendas**: data, quantidade, valor, método de pagamento

---

## 3. FUNCIONALIDADES PRINCIPAIS

### 3.1 VISÃO GERAL DO NEGÓCIO
**O que faz:**
- Mostra números importantes da empresa
- Calcula métricas como ticket médio
- Identifica melhores categorias e métodos de pagamento

**Exemplo prático:**
- "Nossa empresa tem 5.000 clientes"
- "O ticket médio é R$ 247,94"
- "A categoria mais vendida é Brinquedos"

### 3.2 ANÁLISE DE VENDAS
**O que faz:**
- Lista os 10 produtos mais vendidos
- Mostra receita por categoria
- Identifica padrões de vendas

**Exemplo prático:**
- "Produto X vendeu R$ 50.000"
- "Categoria Eletrônicos gera mais receita"
- "Vendas aumentam em novembro/dezembro"

### 3.3 ANÁLISE DE CLIENTES
**O que faz:**
- Mostra distribuição de idades dos clientes
- Analisa faixas de renda
- Categoriza tipos de clientes (Novo, Regular, VIP)

**Exemplo prático:**
- "60% dos clientes têm entre 25-45 anos"
- "Maioria ganha entre R$ 3.000-8.000"
- "10% são clientes VIP"

### 3.4 INTELIGÊNCIA ARTIFICIAL - PREVISÕES
**O que faz:**
- **Treina um modelo** com dados históricos
- **Preve o valor** de uma venda futura
- **Mostra a precisão** do modelo

**Como funciona:**
1. Usuário preenche dados (idade, renda, preço do produto, etc.)
2. IA analisa padrões similares no histórico
3. Calcula valor provável da venda
4. Mostra resultado com nível de confiança

**Exemplo prático:**
- Cliente: 35 anos, renda R$ 5.000
- Produto: R$ 100, avaliação 4.5
- **IA prevê**: "Venda de R$ 280,00"

### 3.5 GRÁFICOS E VISUALIZAÇÕES
**O que faz:**
- Cria gráficos interativos
- Mostra vendas por dia da semana
- Analisa métodos de pagamento
- Distribui avaliações dos produtos

**Exemplo prático:**
- "Sábado é o dia com mais vendas"
- "Cartão é usado em 50% das compras"
- "Produtos têm avaliação média de 4.2"

---

## 4. TECNOLOGIAS UTILIZADAS

### Linguagem de Programação
- **Python**: Linguagem principal do projeto

### Bibliotecas de Dados
- **Pandas**: Manipula e organiza os dados
- **NumPy**: Faz cálculos matemáticos

### Machine Learning
- **Scikit-learn**: Algoritmos de IA
- **Random Forest**: Para previsões
- **K-Means**: Para segmentação

### Interface Web
- **Streamlit**: Cria a interface web
- **Plotly**: Gera gráficos interativos

---

## 5. COMO FUNCIONA A INTELIGÊNCIA ARTIFICIAL

### 5.1 TREINAMENTO DO MODELO
1. **Pega dados históricos** de 15.000 vendas
2. **Identifica padrões**: "Clientes jovens compram mais"
3. **Aprende regras**: "Produtos caros + clientes ricos = vendas altas"
4. **Cria modelo** que pode prever vendas futuras

### 5.2 PREVISÃO DE VENDAS
**Entrada (o que o usuário informa):**
- Idade do cliente
- Renda mensal
- Preço do produto
- Avaliação do produto
- Mês da venda
- Dia da semana

**Saída (o que a IA calcula):**
- Valor provável da venda
- Nível de confiança (R² = 0.595 = 59,5% de precisão)

### 5.3 SEGMENTAÇÃO DE CLIENTES
- **Agrupa clientes** similares
- **Identifica perfis**: "Clientes jovens com baixa renda"
- **Personaliza estratégias** para cada grupo

---

## 6. EXEMPLO PRÁTICO DE USO

### Cenário: "Quero saber se uma venda vai dar certo"

1. **Acesso o sistema** no navegador
2. **Vou para "IA - Previsões"**
3. **Preencho os dados:**
   - Cliente: 28 anos, renda R$ 4.000
   - Produto: R$ 150, avaliação 4.8
   - Data: Junho, Segunda-feira
4. **Clico em "Prever Venda"**
5. **Sistema responde**: "Venda de Alto Valor! R$ 420,00"

### Por que isso é útil?
- **Vendedor**: Sabe se vale a pena investir tempo
- **Gerente**: Planeja metas e estratégias
- **Empresa**: Otimiza recursos e estoque

---

## 7. RESULTADOS E MÉTRICAS

### Performance do Modelo de IA
- **Precisão (R²)**: 59,5% - Modelo razoável
- **Erro Médio (RMSE)**: R$ 151,80 - Erro aceitável

### Métricas do Negócio
- **Ticket Médio**: R$ 247,94
- **Melhor Categoria**: Brinquedos
- **Método Preferido**: Cartão (50%)
- **Período de Pico**: Novembro/Dezembro

---

## 8. BENEFÍCIOS DO PROJETO

### Para o Negócio
- **Tomada de decisão** baseada em dados
- **Identificação de oportunidades** de vendas
- **Otimização de recursos** e estoque
- **Melhoria do atendimento** ao cliente

### Para o Aprendizado
- **Prática com Big Data** real
- **Aplicação de Machine Learning**
- **Desenvolvimento de dashboard**
- **Análise de negócios**

---

## 9. DEMONSTRAÇÃO PRÁTICA

### Passo a passo para mostrar na sala:

1. **Abrir a aplicação** no navegador
2. **Mostrar "Visão Geral"** - números principais
3. **Navegar para "Análise de Vendas"** - produtos top
4. **Ir para "Clientes"** - gráficos demográficos
5. **Demonstrar "IA - Previsões"** - preencher dados e prever
6. **Mostrar "Gráficos"** - visualizações interativas

### Pontos importantes para destacar:
- **Interface intuitiva** e fácil de usar
- **Dados em tempo real** e atualizados
- **Previsões baseadas** em padrões reais
- **Visualizações claras** e informativas

---

## 10. CONCLUSÃO

### O que aprendemos?
- **Big Data** não é só quantidade, é valor
- **IA** pode ajudar em decisões de negócio
- **Visualizações** tornam dados compreensíveis
- **Python** é uma ferramenta poderosa para análise

### Próximos passos possíveis:
- Adicionar mais algoritmos de IA
- Integrar com dados reais de e-commerce
- Implementar alertas automáticos
- Criar relatórios personalizados

---

## PERGUNTAS FREQUENTES

**P: Por que usar IA para prever vendas?**
R: Porque ajuda a tomar decisões melhores e mais rápidas.

**P: Os dados são reais?**
R: Não, são dados simulados mas realistas para demonstração.

**P: Posso usar isso em uma empresa real?**
R: Sim, adaptando para dados reais e necessidades específicas.

**P: É difícil de usar?**
R: Não, a interface é intuitiva e não requer conhecimento técnico.

---

*Projeto desenvolvido por Pablo Hyan Ferreira para disciplina de Big Data em Python* 