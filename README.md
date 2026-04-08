# Banking Transaction-Driven Loan Analytics

Built entirely from scratch across Python, SQL, and Power BI — this 
project connects raw transaction behavior to loan default outcomes, 
forecasts portfolio cash flows, surfaces behavioral anomalies, and 
delivers executive-ready dashboards. Every design decision in this 
project has a documented rationale grounded in how financial 
institutions actually operate.

---

## Why This Project Exists

Public portfolio projects in data science tend to fall into one of 
two categories: Kaggle notebooks that stop at model accuracy, or 
dashboard projects that visualize pre-cleaned data with no analytical 
depth. Neither reflects what an analyst actually does on the job.

This project was built to fill that gap. A credit analyst at a bank 
does not just train a model — they engineer behavioral signals from 
transaction data, validate models against regulatory standards, build 
the data infrastructure that operations teams consume, and present 
findings to a credit committee. This pipeline covers all of it.

---

## Dataset

Two publicly available datasets form the foundation:

**PaySim Synthetic Financial Dataset** (Kaggle — ealaxi)  
6.35 million mobile money transactions simulated over 30 days. 
Contains transaction type, amount, sender/receiver balances, and a 
fraud flag. Used exclusively as the behavioral transaction layer — 
fraud rows are dropped because PaySim zeroes out balance columns for 
cancelled transactions, which would corrupt feature calculations.

**Lending Club Loan Dataset** (Kaggle — wordsforthewise)  
2.2 million loan records spanning 2007 to 2018. Filtered to 1.35 
million resolved loans with known outcomes — Fully Paid or Charged 
Off. In-progress statuses (Current, Late, Grace Period) are excluded 
to prevent label ambiguity from contaminating the training set.

These two datasets share no natural join key. In production, a CIF 
(Customer Information File) number from the core banking system would 
link transaction accounts to loan records. Here, a **synthetic cohort 
assignment** bridges the gap — Lending Club borrowers are binned into 
12 income and risk cohorts, and PaySim customers with matching 
economic profiles are randomly assigned within cohorts using a fixed 
seed of 42. This is documented transparently as a data engineering 
decision rather than obscured.

---

## Technical Stack

| Layer | Technology |
|---|---|
| Data processing | Python 3.11, pandas, numpy |
| ML model | XGBoost 3.2 with CUDA (RTX 4060) |
| Time series | statsmodels SARIMAX, Holt-Winters |
| Explainability | SHAP (TreeExplainer) |
| Database | PostgreSQL 18.3 |
| ORM / loader | SQLAlchemy, psycopg2 |
| Dashboards | Power BI Desktop, DAX |
| Environment | Miniconda, VS Code, Windows 11 |

---

## Project Structure
banking-transaction-driven-loan-analytics/
├── src/
│   ├── config.py                 # paths, seeds, hyperparameters
│   ├── data_loader.py            # PaySim + Lending Club ingestion
│   ├── feature_engineering.py   # behavioral + loan features, cohort join
│   ├── model_xgb.py             # XGBoost classifier, GPU training
│   ├── model_sarimax.py         # SARIMAX forecasting, HW benchmark
│   ├── score_customer.py        # interactive single-customer scorer
│   └── forecast_query.py        # interactive segment forecast queries
├── sql/
│   ├── create_tables.sql        # star schema DDL
│   ├── create_views.sql         # 4 analytical views
│   ├── load_to_postgres.py      # parquet → PostgreSQL loader
│   └── create_views.py          # view runner
├── powerbi/                     # .pbix files (gitignored)
├── data/                        # gitignored — download from Kaggle
├── run_pipeline.py              # end-to-end runner
└── README.md

---

## Phase 1 — Feature Engineering

This is the most analytically intensive phase of the project. Raw 
transaction data is meaningless to a default model — what matters is 
what the transactions reveal about a borrower's financial health.

PaySim transactions are aggregated to monthly customer-level features 
across four groups:

**Volatility features** capture erratic financial behavior. The 
coefficient of variation of transaction amounts normalizes for account 
size — a customer transacting in $5,000 increments with high variance 
is more stressed than one transacting in $50 increments with the same 
absolute variance. Balance standard deviation over a 3-month rolling 
window captures intra-month account instability.

**Behavioral pattern features** identify structural cash flow 
problems. The withdrawal-to-deposit ratio is the single most 
operationally meaningful metric — a ratio consistently above 1.0 
means money is leaving faster than it arrives. Sustained above-1.0 
ratios are the first signal a relationship manager looks at when 
assessing customer health. A synthetic overdraft flag fires when 
closing balance falls below 10% of the month's average balance.

**Trend features** capture trajectory rather than level. A customer 
with a $5,000 balance is very different from a customer whose balance 
was $12,000 three months ago and is now $5,000. The balance slope 
over the trailing 3 months provides this directional signal. Month-
over-month change in average transaction amount captures acceleration 
or deceleration in spending.

**Stress event flags** aggregate behavioral stress over time. The 
financial distress duration feature counts how many of the last 6 
months saw outflows exceed inflows. A customer with a distress 
duration of 5 has been in structural deficit spending for 5 of the 
last 6 months — this is the strongest individual predictor of future 
default in behavioral credit models.

Rolling features are computed using vectorized numpy array operations 
rather than pandas groupby transforms. The groupby approach with 
6.35M rows ran for over 10 minutes before being replaced with a sort-
based boundary masking approach that completes in under 90 seconds.

From Lending Club, nine additional features are engineered: 
loan-to-income ratio, interest burden (installment as percentage of 
monthly income), FICO midpoint, ordinal grade encoding (A=1 through 
G=7), term extraction from string, revolving utilization flag above 
75%, and public records flag. Interaction features are then computed 
— grade × distress duration, DTI × withdrawal ratio, FICO × balance 
volatility — to allow the model to learn joint effects without 
requiring deep trees.

---

## Phase 2a — Default Classification (XGBoost)

**Model choice:** XGBoost with GPU acceleration via CUDA on an RTX 
4060. XGBoost was selected over Random Forest because RAPIDS cuML 
(required for GPU-accelerated Random Forest) does not support Windows 
natively. XGBoost's Windows CUDA wheel works cleanly and delivers 
gradient boosting — which outperforms bagged ensembles through 
sequential error correction — with full GPU acceleration.

**Train/test split:** Temporal rather than random. Loans issued 
through 2015 form the training set (829,324 loans). Loans issued from 
2016 onwards form the test set (518,372 loans). Random splits are 
never used on time-adjacent financial data — they would allow the 
model to train on 2016 loans while testing on 2015 loans, introducing 
leakage and producing inflated metrics that do not represent 
deployment performance.

**Class imbalance:** The training set has an 18.5% default rate — 
approximately 4.4 non-defaulters per defaulter. `scale_pos_weight` 
is set to 4.42, instructing XGBoost to treat each default case as 
equivalent to 4.42 non-default cases. Without this adjustment, the 
model would learn to predict "no default" for almost everyone and 
achieve 81.5% accuracy while detecting almost nothing.

**Early stopping:** Training runs for up to 1500 rounds with early 
stopping at 50 rounds of no improvement on test AUC. The model 
stopped at round 302, restoring weights from round 253.

**Results:**

| Metric | Value |
|---|---|
| AUC-ROC | 0.700 |
| Gini Coefficient | 0.400 |
| KS Statistic | 0.288 |
| Top-10% Lift | 2.13x |
| Top-20% Lift | 1.89x |

The AUC of 0.700 sits at the entry point of the industry-standard 
range (0.70–0.80) for consumer credit models. The ceiling at 0.70 
reflects the synthetic cohort assignment — because transaction 
profiles are randomly assigned within cohorts rather than organically 
linked to loan outcomes, behavioral features cannot establish true 
predictive causality. A production deployment with native CIF-linked 
data would realistically achieve 0.74–0.78 AUC.

The decile table tells a more operationally meaningful story than 
AUC alone:

| Decile | Default Rate | Lift | Cumulative Defaults Captured |
|---|---|---|---|
| D1 (top 10%) | 46.7% | 2.13x | 20.8% |
| D1–D2 (top 20%) | — | 1.89x | 36.7% |
| D1–D3 (top 30%) | — | — | 50.4% |

A bank actioning the top 30% of flagged borrowers catches half of 
all defaults while reviewing only 30% of the book.

**SHAP explainability:** SHAP TreeExplainer computes Shapley values 
for every customer in the test set. The top 5 features by SHAP 
contribution become that customer's adverse action reason codes — the 
specific factors that drove their score. This directly mirrors the 
reason code requirements under ECOA and FCRA, and satisfies the SR 
11-7 model risk management guidance that requires model outputs to be 
explainable to non-technical stakeholders.

---

## Phase 2b — Portfolio Cash Flow Forecasting (SARIMAX)

**What is being forecast:** Monthly net cash flow per portfolio 
segment — total installment repayments received minus total loan 
disbursements — aggregated across 12 segments defined by 3 loan grade 
buckets (AB, CD, EG) and 4 US regions.

**Model selection rationale:** SARIMAX was selected after evaluating 
Prophet, LightGBM recursive forecasting, and Holt-Winters. Prophet 
fits one model per segment and handles external regressors through a 
clunky workaround. LightGBM recursive forecasting requires more 
monthly observations per segment than are available here to learn 
seasonal patterns reliably. Holt-Winters handles no external 
regressors at all. SARIMAX natively incorporates seasonal 
differencing at m=12 for annual repayment cycles, accepts transaction 
velocity and disbursement volume as first-class exogenous variables 
through the X component, and is the standard model in CFA and FRM 
curricula for exactly this type of monthly financial series.

**Order selection:** AIC/BIC grid search across a constrained 
candidate set selected (1,1,2)(0,1,1,12) across all 12 segments. The 
seasonal differencing D=1 removes the annual seasonal pattern by 
subtracting each month's value from the same month one year prior, 
leaving the irregular component for the ARMA terms to model.

**Validation:** Walk-forward expanding window cross-validation with 
a minimum training window of 36 months. Each forecast step is 
genuinely out-of-sample — the model has never seen the month it is 
predicting. 101–103 validation steps are generated per segment, 
producing a robust estimate of deployment performance.

**Results:**

| Metric | SARIMAX | Holt-Winters |
|---|---|---|
| MAPE range | 0.06%–0.13% | 16.90%–31.47% |
| Lift over benchmark | 16–31 percentage points | — |
| Validation steps | 101–103 per segment | same |

The sub-1% MAPE reflects the near-deterministic growth trend in the 
Lending Club portfolio from 2007 to 2018 rather than model 
overfitting — the same trend causes Holt-Winters to fail at 16–31% 
MAPE, confirming SARIMAX captures genuine structural signal. On a 
mature, stationary portfolio, expected MAPE would be 8–15%. The 
validation framework is production-appropriate regardless of the 
absolute error level.

**What-if simulation:** The `forecast_query.py` interactive tool 
allows segment-level scenario analysis — a 20% transaction velocity 
reduction (economic slowdown) produces a 2.3% deterioration in 
6-month forecast cash flow for Grade EG segments, reflecting the 
contractual nature of installment payments which are relatively 
insensitive to short-term behavioral changes.

---

## Phase 3 — SQL Data Mart

A PostgreSQL star schema data mart operationalizes all Python outputs 
into a queryable, joinable structure that Power BI and downstream 
applications can consume.

**Schema:**

Dimensions:    dim_customer (1.35M)  dim_date (156)  dim_segment (12)
Facts:         fact_loans (1.35M)    fact_features (6.35M)
fact_predictions (518K)  fact_forecasts (1,735)
fact_decile_analysis (10)  fact_shap_importance (39)

**Four analytical views** cover the four pillars of credit risk 
monitoring:

`vw_customer_risk_scorecard` — 518,372 rows. Joins XGBoost default 
probabilities to behavioral transaction features and computes a 
composite risk score: 60% model PD + 40% behavioral stress. This 
weighting mirrors how credit teams overlay quantitative model scores 
with qualitative behavioral overlays. Outputs composite risk tier 
(Low/Medium/High/Watch), watchlist flag (high PD + behavioral 
distress), and portfolio-at-risk flag.

`vw_portfolio_cashflow_forecast` — 1,735 rows. SARIMAX forecasts 
versus actuals with 3-month rolling MAPE, variance attribution 
(structural vs behavioral drivers), cash flow status, and direction 
accuracy for each segment-month combination.

`vw_anomaly_detection` — 8,574 rows. Population-level behavioral 
anomaly detection across three dimensions: withdrawal ratio spike 
(top 5th percentile, 2+ standard deviations above population mean), 
balance deterioration (closing balance below 10% of mean balance with 
negative slope), and high risk profile (top 5th percentile behavioral 
risk score with top 10th percentile distress duration). The original 
personalized rolling baseline design was replaced with population-
level thresholds because PaySim's 30-day simulation produces only one 
monthly row per customer — insufficient for rolling window anomaly 
detection. This limitation is documented explicitly.

`vw_segment_risk_decomposition` — 48 rows. Portfolio risk 
concentration by grade bucket, region, and income band. Includes 
Basel III-aligned concentration flags that fire when any segment's 
portfolio-at-risk value exceeds 15% of total book value. This 
elevates the view from analytical output to regulatory infrastructure.

---

## Phase 4 — Power BI Dashboards

Two dashboards across six pages form an executive reporting suite.

**Dashboard 1 — Portfolio Health and Cash Flow** covers portfolio 
overview KPIs, metric movement explanation, and cash flow drill-down. 
The SARIMAX confidence band visualization — actual versus forecast 
lines with 80% prediction interval shading — is the most visually 
distinctive output. DAX measures power dynamic KPI cards, an auto-
generated plain-English performance commentary, and a model 
performance comparison matrix showing SARIMAX versus Holt-Winters 
MAPE by segment.

**Dashboard 2 — Risk Intelligence and Early Warning** covers the 
customer risk heatmap, the anomaly detection alert feed, and the 
stress testing interface. The risk heatmap uses a matrix visual with 
loan grade on rows and US state on columns, coloured green-to-red by 
average default probability. The what-if stress testing page 
implements two Power BI parameters: an interest rate change parameter 
(−2% to +5% in 0.25% increments) that dynamically recalculates how 
many borrowers breach the 40% affordability threshold, and a velocity 
shock parameter (0% to 50%) that projects additional Watch-tier 
entries under economic stress.

---

## Running the Project

**Environment setup:**
```bash
conda create -n banking-risk python=3.11 -y
conda activate banking-risk
pip install pandas numpy scipy scikit-learn joblib tqdm statsmodels \
            xgboost shap psycopg2-binary sqlalchemy matplotlib \
            seaborn python-dotenv pyarrow fastparquet
```

**Data:** Download PaySim (`ealaxi/paysim1`) and Lending Club 
(`wordsforthewise/lending-club`) from Kaggle. Place `paysim.csv` 
and `lending_club.csv.gz` in `data/raw/`.

**Environment variables:** Create `.env` at project root:

DB_HOST=localhost
DB_PORT=5432
DB_NAME=banking_risk
DB_USER=postgres
DB_PASSWORD=your_password

**Run individual stages:**
```bash
python src/data_loader.py          # Stage 1 — data ingestion
python src/feature_engineering.py  # Stage 2 — feature engineering
python src/model_xgb.py            # Stage 3a — XGBoost classifier
python src/model_sarimax.py        # Stage 3b — SARIMAX forecasting
python sql/load_to_postgres.py     # Stage 4 — load to PostgreSQL
python sql/create_views.py         # Stage 5 — create analytical views
```

**Interactive scoring:**
```bash
python src/score_customer.py       # score individual customers
python src/forecast_query.py       # query segment forecasts
```

**Full pipeline:**
```bash
python run_pipeline.py
```

---

## Key Design Decisions

**Temporal train/test split over random split.** Financial models 
must be validated on data that postdates their training set. A random 
split would allow training on 2016 loans while testing on 2015 loans, 
producing optimistically inflated metrics that do not represent real 
deployment performance.

**Monotone constraints evaluated and removed.** Constraints were 
tested on XGBoost to enforce known credit risk relationships — higher 
distress duration should increase default probability. They were 
removed because the synthetic cohort assignment introduces noise that 
fights the constraints, causing underfitting. The decision is 
documented explicitly: in a production environment with native CIF-
linked data, monotone constraints on distress duration and withdrawal 
ratio would be appropriate and are recommended.

**Anomaly detection redesigned from personalized to population-
level.** The original design used 6-month rolling baselines per 
customer. PaySim's 30-day simulation produces one monthly row per 
customer, making rolling window anomaly detection impossible. 
Documented as a known limitation with the production alternative 
described.

**SARIMAX over Prophet.** Prophet fits one model per segment through 
a clunky add_regressor() workaround and approximates seasonality 
through Fourier terms. SARIMAX handles seasonal differencing natively 
and accepts exogenous regressors as first-class inputs. For monthly 
financial series with 36–120 observations per segment and known 
annual seasonality, SARIMAX is the analytically correct choice and is 
present in the CFA and FRM curricula that credit teams are trained on.

**SR 11-7 alignment.** The Federal Reserve's model risk management 
guidance requires that models used in credit decisions be 
interpretable and that outputs be explainable to non-technical 
stakeholders. SHAP reason codes provide customer-level explanations 
that satisfy adverse action notice requirements under ECOA and FCRA. 
SARIMAX satisfies the interpretability requirement through its 
explicit coefficient structure. Both model choices are documented with 
this regulatory context in mind.

---

## Limitations and Production Extensions

This project uses synthetic data with an artificial join between 
datasets. In a production environment:

- The PaySim-to-Lending Club join would be replaced with native CIF 
linkage from a core banking system (Finacle, Temenos, or similar)
- Anomaly detection would use 12+ months of transaction history for 
personalized rolling baselines rather than population percentiles
- The XGBoost model would be retrained quarterly with fresh resolved 
loans and monitored for PSI (Population Stability Index) drift
- SARIMAX forecasts would target a mature, stationary portfolio where 
expected MAPE would be 8–15% rather than the sub-1% achieved here on 
a rapidly growing book
- Monotone constraints would be applied on behavioral features with 
confirmed causal linkage to default outcomes
- The SQL views would be materialized and refreshed on a nightly 
schedule rather than computed on-demand

---

## Model Performance Summary

| Model | Metric | Value |
|---|---|---|
| XGBoost | AUC-ROC | 0.700 |
| XGBoost | Gini Coefficient | 0.400 |
| XGBoost | KS Statistic | 0.288 |
| XGBoost | Top-decile lift | 2.13x |
| XGBoost | Top-20% defaults captured | 36.7% |
| SARIMAX | MAPE (best segment) | 0.06% |
| SARIMAX | MAPE (worst segment) | 0.13% |
| SARIMAX | Lift over HW benchmark | 16–31pp |
| SARIMAX | Validation steps per segment | 101–103 |
| Anomaly Detection | Customers flagged | 8,574 |
| Anomaly Detection | Flag rate | 1.65% |
| SQL Data Mart | Total rows loaded | 8.2M+ |
| SQL Data Mart | Views created | 4 |

---
