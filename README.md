# 🌍 Expectativa de Vida Global — 2016 a 2025

Pipeline completo de Ciência de Dados aplicado à saúde pública, construído sobre dados da OMS para prever a expectativa de vida de países a partir de indicadores socioeconômicos e de saúde.

---

## 📋 Contexto

A expectativa de vida ao nascer é um dos principais indicadores de desenvolvimento humano utilizado pela OMS, Banco Mundial e PNUD. Ela sintetiza, em um único número, a qualidade dos sistemas de saúde, a renda per capita, a escolaridade e as condições sanitárias de uma população.

**Pergunta central do projeto:**
> Quais fatores socioeconômicos e de saúde têm maior influência sobre a expectativa de vida de um país, e como podemos usá-los para prever anos futuros com precisão adequada?

---

## 📁 Estrutura do Repositório

```
.
├── Aula18_Expectativa_Vida.ipynb        # Notebook principal
├── expectativa_vida_2016_2025.csv       # Dataset bruto (1.945 linhas × 22 colunas)
├── df_clean.csv                         # Dataset após Data Munging (1.930 linhas)
├── requirements.txt
└── README.md
```

---

## 🗃️ Dataset

| Atributo | Valor |
|---|---|
| Fonte | OMS / dados sintéticos 2016–2025 |
| Registros brutos | 1.945 linhas |
| Registros após limpeza | 1.930 linhas (15 duplicatas removidas) |
| Colunas | 22 (21 features + 1 target) |
| Países | 193 entidades únicas |
| Período | 2016 a 2025 (10 anos) |
| Target | `Life Expectancy` (anos) |

**Domínios das variáveis:**

| Domínio | Exemplos |
|---|---|
| Identificação | Country, Year, Status (Developed/Developing) |
| Saúde | Adult Mortality, HIV/AIDS, Hepatitis B, Polio, DTP, BMI, Measles |
| Socioeconômico | GDP, Population, Schooling, Income Composition, % Expenditure |

---

## ⚙️ Pipeline — 8 Etapas

### Etapa 1 · Análise Exploratória (EDA)
- Inspeção estrutural: `shape`, `dtypes`, `describe()`, missings por coluna
- Distribuição da variável-alvo (assimétrica à esquerda)
- Evolução temporal e top 10 países por expectativa média
- Respostas às perguntas orientadoras Q1–Q4 (distribuição, missings, skewness, comparação Developing vs Developed)
- Teste de Mann-Whitney: diferença **estatisticamente significativa** entre grupos (p < 10⁻¹¹⁴), medianas: Developed 81.9 anos vs Developing 72.1 anos

### Etapa 2 · Data Munging
Problemas reais identificados e tratados:

| Problema | Tratamento |
|---|---|
| Vírgulas decimais (`"67,55"`) | `str.replace(',', '.')` + `pd.to_numeric` |
| Placeholders `"-"` em colunas numéricas | Substituição por `NaN` |
| `Year` em múltiplos formatos (`2016`, `01/01/2016`, `2016-01-01`) | Extração com `str.extract(r'(\d{4})')` |
| `Status` inconsistente (`DEVELOPING`, ` Developed `) | `str.strip().str.title()` → 2 categorias |
| `Adult Mortality` negativo (4 registros) | Substituição por `NaN` |
| 15 linhas duplicadas | `drop_duplicates()` |
| `life_expectancy` fora de [35, 100] (10 registros) | Substituição por `NaN` |
| Vacinas fora de [0, 100%] | Substituição por `NaN` |
| `bmi` fora de [5, 80] e `income_composition` fora de [0, 1] | Substituição por `NaN` |
| Valores ausentes remanescentes | Imputação: mediana por país → por Status → global |

**Resultado:** 0 missings restantes, 1.930 linhas limpas, tipos corretos, `df_clean.csv` salvo.

### Etapa 3 · Engenharia de Recursos

| Feature Nova | Origem | Motivação |
|---|---|---|
| `log_gdp`, `log_population`, `log_measles`, `log_percentage_expenditure` | Colunas brutas | Reduz skewness; relação log-linear com saúde |
| `vaccine_coverage_avg` | HepB + Polio + DTP | Índice sintético de imunização |
| `disease_burden` | HIV/AIDS + Measles | Pressão agregada de doenças infecciosas |
| `thinness_avg` | thinness 1–19 + 5–9 anos | Evita multicolinearidade (r ≈ 0.98) |
| `mortality_le_ratio` | Adult Mortality / Life Expectancy | Criada conforme especificação; removida na Etapa 4 (data leakage) |
| `status_enc` | Status | Encoding binário (Developed = 1) |
| `year_norm` | Year | Tendência temporal normalizada 0–1 |

### Etapa 4 · Correlação e Multicolinearidade (VIF)
- Heatmap de correlação com 20 features + target
- Top 10 correlações com `life_expectancy`
- Cálculo do VIF para todas as features
- Remoção de `mortality_le_ratio` (VIF = 23.2, data leakage direto do target) e `infant_deaths` (r ≈ 0.99 com `underfive_deaths`)
- Após remoção: todas as features com VIF < 6 ✅

### Etapa 5 · Modelagem Estatística (OLS)
Três versões do modelo OLS:

| Versão | Descrição | R² | Cond. No. |
|---|---|---|---|
| v1 | 9 features em escala original | 0.662 | 2.041 ❌ |
| v2 | Remove variáveis com p > 0.05 | 0.662 | — |
| v3 | StandardScaler aplicado | 0.662 | **3.7** ✅ |

Diagnóstico de resíduos (grid 2×2): Resíduos vs Fitted · Q-Q Plot · Histograma · Scale-Location.

Durbin-Watson ≈ 0.90 indica autocorrelação esperada em dados de painel (mesmo país repetido por ano) — limitação inerente ao OLS clássico nesse tipo de dado.

### Etapa 6 · Modelagem Preditiva
6 algoritmos comparados com KFold (k=5) usando **Pipeline** (StandardScaler encapsulado nos modelos lineares, eliminando data leakage de escala). Métricas: CV R², Test R², RMSE, MAE, MAPE.

| Modelo | CV R² | Test R² | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|---|
| **Random Forest** | **0.951** | **0.963** | **1.64** | **1.11** | **1.53** |
| XGBoost | 0.950 | 0.960 | 1.70 | 1.17 | 1.61 |
| Gradient Boosting | 0.933 | 0.941 | 2.06 | 1.56 | 2.15 |
| Linear Regression | 0.734 | 0.754 | 4.21 | 3.29 | 4.60 |
| Ridge (α=1) | 0.735 | 0.754 | 4.21 | 3.29 | 4.60 |
| Lasso (α=0.1) | 0.734 | 0.754 | 4.21 | 3.28 | 4.59 |

**Melhor modelo:** Random Forest — Test R² = 0.963, MAPE = 1.53% (erro médio de ~1.1 ano sobre ~72 anos previstos).

### Etapa 6.1 · Validação Temporal
Treinamento em 2016–2023, teste em 2024–2025 — simula uso real do modelo para prever dados futuros não vistos no treino.

| Modelo | Test R² | RMSE | MAPE (%) |
|---|---|---|---|
| **Random Forest** | **0.907** | **2.59** | **2.51** |
| XGBoost | 0.899 | 2.70 | 2.71 |
| Gradient Boosting | 0.882 | 2.92 | 3.06 |
| Linear Regression | 0.679 | 4.82 | 5.28 |

O Random Forest mantém R² 0.907 em dados de anos futuros — confirma capacidade de **generalização real**, não apenas memorização do conjunto de treino.

### Etapa 7 · Avaliação do Modelo
- Gráfico Real vs. Previsto (diagonal de referência)
- Erro médio absoluto por grupo: Developed (MAE ≈ 1.56) vs. Developing (MAE ≈ 1.46)
- Curva de aprendizado: convergência entre treino e validação indica boa generalização sem overfitting

### Etapa 8 · Previsão com Novos Dados
Cenário hipotético: país em desenvolvimento médio, ano 2027.

| Cenário | Previsão | Δ vs Base |
|---|---|---|
| Base | 69.12 anos | — |
| +20% gastos em saúde | 68.51 anos | −0.61 |
| HIV/AIDS → 0.1 | 69.16 anos | +0.04 |

IC 95% via Bootstrap (500 amostras): **[66.25 · 71.18] anos**

> Os deltas pequenos são esperados em Random Forest aplicado a uma observação isolada. Para uso de política pública, o modelo é mais adequado para **comparar grupos de países** do que para simular intervenções individuais.

---

## 📊 Resultados Principais

- **15 duplicatas e 10 registros implausíveis** detectados e removidos sistematicamente no munging.
- O OLS (R² ≈ 0.66) revelou que **mortalidade adulta, escolaridade, composição de renda e HIV/AIDS** são os principais preditores lineares da expectativa de vida.
- O uso de **Pipeline** eliminando data leakage de escala elevou o Test R² do Random Forest para **0.963**.
- A **validação temporal** (2016–2023 → 2024–2025) confirma R² 0.907 em dados futuros — modelo generaliza além do período de treino.
- A padronização reduz o Cond. No. de 2.041 para 3.7 — eliminando multicolinearidade numérica do OLS sem alterar o R².

---

## 🛠️ Tecnologias

```
Python 3.12
pandas · numpy · scipy
matplotlib · seaborn
statsmodels
scikit-learn
xgboost
```

---

## ▶️ Como Executar

```bash
# Clone o repositório
git clone https://github.com/<seu-usuario>/life_expectancy_project.git
cd life_expectancy_project

# Instale as dependências
pip install -r requirements.txt

# Abra o notebook
jupyter lab Aula18_Expectativa_Vida.ipynb
```

> O arquivo `expectativa_vida_2016_2025.csv` deve estar no mesmo diretório do notebook.

---

## ⚠️ Observações

- O modelo tem finalidade **preditiva e exploratória**, não causal.
- A importância das variáveis não implica causalidade — fatores como escolaridade e renda são correlacionados entre si e com a expectativa de vida.
- As simulações de cenários são **apoio inicial à decisão**, não substituem análise de custo-benefício e viabilidade operacional.
- O Durbin-Watson baixo (≈ 0.90) indica autocorrelação nos resíduos do OLS, esperada em dados de painel. Para inferência causal rigorosa, modelos de efeitos fixos por país seriam mais adequados.

---

## 📚 Referências

- World Health Organization — Global Health Observatory
- Business Case: Ciência de Dados Aplicada — Expectativa de Vida Global 2016–2025
- Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
- Statsmodels documentation: [https://www.statsmodels.org](https://www.statsmodels.org)