-- ============================================================
-- BANKING RISK ANALYTICS — TABLE DEFINITIONS
-- Run once to set up the schema
-- ============================================================

-- ── Drop existing tables in dependency order ─────────────────
DROP TABLE IF EXISTS fact_predictions CASCADE;
DROP TABLE IF EXISTS fact_loans CASCADE;
DROP TABLE IF EXISTS fact_transactions CASCADE;
DROP TABLE IF EXISTS fact_forecasts CASCADE;
DROP TABLE IF EXISTS fact_features CASCADE;
DROP TABLE IF EXISTS fact_decile_analysis CASCADE;
DROP TABLE IF EXISTS fact_shap_importance CASCADE;
DROP TABLE IF EXISTS dim_customer CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_segment CASCADE;

-- ── Dimension: Customer ───────────────────────────────────────
CREATE TABLE dim_customer (
    customer_id         VARCHAR(12)     PRIMARY KEY,
    income_band         VARCHAR(10),
    risk_tier_cohort    VARCHAR(10),
    paysim_customer_id  VARCHAR(12),
    cohort              VARCHAR(20),
    region              VARCHAR(20),
    addr_state          VARCHAR(5)
);

-- ── Dimension: Date ───────────────────────────────────────────
CREATE TABLE dim_date (
    month_key           DATE            PRIMARY KEY,
    year                INTEGER,
    month               INTEGER,
    quarter             INTEGER,
    is_month_end        BOOLEAN,
    is_quarter_end      BOOLEAN,
    month_label         VARCHAR(10)     -- e.g. 'Jan-2015'
);

-- ── Dimension: Segment ───────────────────────────────────────
CREATE TABLE dim_segment (
    segment_id          VARCHAR(30)     PRIMARY KEY,
    grade_bucket        VARCHAR(10),
    region              VARCHAR(20)
);

-- ── Fact: Loans ───────────────────────────────────────────────
CREATE TABLE fact_loans (
    customer_id         VARCHAR(12)     PRIMARY KEY,
    loan_amnt           NUMERIC(12,2),
    term_months         INTEGER,
    int_rate            NUMERIC(6,3),
    installment         NUMERIC(10,2),
    grade               VARCHAR(2),
    grade_encoded       INTEGER,
    sub_grade           VARCHAR(3),
    annual_inc          NUMERIC(14,2),
    monthly_income      NUMERIC(12,2),
    loan_status         VARCHAR(80),
    default_flag        INTEGER,
    dti                 NUMERIC(8,3),
    issue_date          DATE,
    issue_year          INTEGER,
    issue_month         INTEGER,
    addr_state          VARCHAR(5),
    fico_mid            NUMERIC(6,1),
    open_acc            INTEGER,
    pub_rec             INTEGER,
    has_pub_rec         INTEGER,
    revol_util          NUMERIC(8,3),
    revol_bal           NUMERIC(14,2),
    high_revol_util_flag INTEGER,
    loan_to_income      NUMERIC(10,6),
    interest_burden     NUMERIC(10,6)
);

-- ── Fact: Transaction Features (monthly per customer) ─────────
CREATE TABLE fact_features (
    customer_id                 VARCHAR(12),
    month_key                   VARCHAR(10),
    txn_count                   NUMERIC(10,2),
    mean_amount                 NUMERIC(14,4),
    std_amount                  NUMERIC(14,4),
    cv_amount                   NUMERIC(12,6),
    total_outflow               NUMERIC(14,2),
    total_inflow                NUMERIC(14,2),
    cashout_amount              NUMERIC(14,2),
    closing_balance             NUMERIC(14,2),
    mean_balance                NUMERIC(14,2),
    std_balance                 NUMERIC(14,2),
    withdrawal_deposit_ratio    NUMERIC(12,6),
    cashout_proportion          NUMERIC(12,6),
    max_withdrawal_to_balance   NUMERIC(12,6),
    overdraft_flag              INTEGER,
    balance_vol_3m              NUMERIC(14,4),
    balance_slope_3m            NUMERIC(14,4),
    txn_count_lag1              NUMERIC(10,2),
    mean_amount_mom_change      NUMERIC(12,6),
    distress_duration_6m        NUMERIC(6,2),
    behavioral_risk_score       NUMERIC(8,6),
    PRIMARY KEY (customer_id, month_key)
);

-- ── Fact: Model Predictions ───────────────────────────────────
CREATE TABLE fact_predictions (
    customer_id             VARCHAR(12)     PRIMARY KEY,
    prediction_month        DATE,
    loan_amnt               NUMERIC(12,2),
    grade                   VARCHAR(2),
    addr_state              VARCHAR(5),
    default_flag            INTEGER,
    default_probability     NUMERIC(8,6),
    risk_tier               VARCHAR(12),
    model_name              VARCHAR(30)
);

-- ── Fact: SARIMAX Forecasts ───────────────────────────────────
CREATE TABLE fact_forecasts (
    segment_id              VARCHAR(30),
    month_key               DATE,
    actual_cash_flow        NUMERIC(18,2),
    forecasted_cash_flow    NUMERIC(18,2),
    upper_bound             NUMERIC(18,2),
    lower_bound             NUMERIC(18,2),
    is_forecast             BOOLEAN,
    sarimax_mape            NUMERIC(12,6),
    sarimax_mae             NUMERIC(18,2),
    sarimax_bias            NUMERIC(18,2),
    hw_mape                 NUMERIC(12,6),
    sarimax_lift_pp         NUMERIC(12,6),
    best_order              VARCHAR(20),
    best_seasonal_order     VARCHAR(20),
    n_validation_steps      INTEGER,
    PRIMARY KEY (segment_id, month_key)
);

-- ── Fact: Decile Analysis ─────────────────────────────────────
CREATE TABLE fact_decile_analysis (
    decile                  VARCHAR(5)      PRIMARY KEY,
    total                   INTEGER,
    defaults                INTEGER,
    default_rate            NUMERIC(8,6),
    cum_defaults_pct        NUMERIC(8,4),
    lift                    NUMERIC(8,4),
    mean_score              NUMERIC(8,6)
);

-- ── Fact: SHAP Feature Importance ────────────────────────────
CREATE TABLE fact_shap_importance (
    feature                 VARCHAR(50)     PRIMARY KEY,
    shap_importance         NUMERIC(12,8),
    gain_importance         NUMERIC(14,4)
);

COMMENT ON TABLE fact_loans IS
    'Resolved Lending Club loans with engineered features. 
     One row per borrower.';
COMMENT ON TABLE fact_predictions IS
    'XGBoost default probability scores for test set borrowers.
     Includes risk tier and calibrated probability.';
COMMENT ON TABLE fact_forecasts IS
    'SARIMAX 6-month forward cash flow forecasts by portfolio segment.
     Includes Holt-Winters benchmark comparison.';
COMMENT ON TABLE fact_features IS
    'Monthly transaction behavioral features from PaySim.
     One row per (paysim_customer, month).';
COMMENT ON TABLE fact_decile_analysis IS
    'XGBoost model lift by score decile.
     D1 = highest risk decile.';

SELECT 'Tables created successfully' AS status;