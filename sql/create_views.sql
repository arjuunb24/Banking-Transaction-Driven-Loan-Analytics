-- ============================================================
-- BANKING RISK ANALYTICS — THE FOUR ANALYTICAL VIEWS
-- Run after load_to_postgres.py completes
-- ============================================================


-- ════════════════════════════════════════════════════════════
-- VIEW 1: CUSTOMER RISK SCORECARD
-- Coverage: KPI Tracking + Risk Segmentation
-- Answers: Who are my riskiest borrowers and why?
-- ════════════════════════════════════════════════════════════

DROP VIEW IF EXISTS vw_customer_risk_scorecard CASCADE;

CREATE VIEW vw_customer_risk_scorecard AS

WITH behavioral_stats AS (
    -- Aggregate transaction features to customer level
    -- Takes the most recent month's behavioral snapshot per customer
    SELECT
        f.customer_id,
        AVG(f.withdrawal_deposit_ratio)     AS avg_withdrawal_deposit_ratio,
        AVG(f.distress_duration_6m)         AS avg_distress_duration,
        AVG(f.balance_vol_3m)               AS avg_balance_volatility,
        AVG(f.overdraft_flag)               AS overdraft_frequency,
        AVG(f.cashout_proportion)           AS avg_cashout_proportion,
        AVG(f.closing_balance)              AS avg_closing_balance,
        AVG(f.mean_balance)                 AS avg_mean_balance,
        SUM(f.overdraft_flag)               AS total_overdraft_months,
        MAX(f.behavioral_risk_score)        AS max_behavioral_risk_score,
        AVG(f.behavioral_risk_score)        AS avg_behavioral_risk_score,
        COUNT(*)                            AS months_on_book_txn
    FROM fact_features f
    GROUP BY f.customer_id
),

composite_score AS (
    -- Compute composite risk score:
    -- 60% model default probability + 40% behavioral stress score
    -- This mirrors how banks overlay model scores with qualitative overlays
    SELECT
        p.customer_id,
        p.default_probability,
        p.risk_tier,
        p.grade,
        p.loan_amnt,
        p.default_flag,
        p.prediction_month,
        b.avg_withdrawal_deposit_ratio,
        b.avg_distress_duration,
        b.avg_balance_volatility,
        b.overdraft_frequency,
        b.avg_cashout_proportion,
        b.avg_closing_balance,
        b.total_overdraft_months,
        b.avg_behavioral_risk_score,

        -- Behavioral stress score normalized 0-100
        -- Higher = more stressed
        ROUND(
            LEAST(b.avg_behavioral_risk_score * 100, 100)
        ::NUMERIC, 2) AS behavioral_stress_score,

        -- Composite score: 60% PD + 40% behavioral
        ROUND((
            0.60 * p.default_probability +
            0.40 * LEAST(b.avg_behavioral_risk_score, 1.0)
        )::NUMERIC, 6) AS composite_risk_score,

        -- Composite risk tier based on composite score
        CASE
            WHEN (0.60 * p.default_probability +
                  0.40 * LEAST(b.avg_behavioral_risk_score, 1.0)) >= 0.60
                THEN 'Watch'
            WHEN (0.60 * p.default_probability +
                  0.40 * LEAST(b.avg_behavioral_risk_score, 1.0)) >= 0.45
                THEN 'High'
            WHEN (0.60 * p.default_probability +
                  0.40 * LEAST(b.avg_behavioral_risk_score, 1.0)) >= 0.25
                THEN 'Medium'
            ELSE 'Low'
        END AS composite_risk_tier

    FROM fact_predictions p
    LEFT JOIN behavioral_stats b
        ON p.customer_id = b.customer_id
)

SELECT
    c.customer_id,
    c.prediction_month,

    -- Loan attributes
    l.grade,
    l.grade_encoded,
    l.loan_amnt,
    l.int_rate,
    l.installment,
    l.dti,
    l.annual_inc,
    l.term_months,
    l.fico_mid,
    l.loan_to_income,
    l.interest_burden,
    l.revol_util,
    l.has_pub_rec,
    l.issue_year,
    l.addr_state,

    -- Model output
    c.default_probability,
    c.risk_tier                     AS model_risk_tier,
    c.default_flag                  AS actual_outcome,

    -- Behavioral features
    c.avg_withdrawal_deposit_ratio,
    c.avg_distress_duration,
    c.avg_balance_volatility,
    c.overdraft_frequency,
    c.total_overdraft_months,
    c.avg_closing_balance,
    c.avg_cashout_proportion,

    -- Composite scoring
    c.behavioral_stress_score,
    c.composite_risk_score,
    c.composite_risk_tier,

    -- Watchlist flag:
    -- Customers with high model score AND behavioral stress
    CASE
        WHEN c.default_probability >= 0.60
         AND c.avg_distress_duration >= 3
            THEN TRUE
        ELSE FALSE
    END AS watchlist_flag,

    -- Portfolio-at-risk flag
    CASE
        WHEN c.composite_risk_tier IN ('High', 'Watch')
            THEN TRUE
        ELSE FALSE
    END AS portfolio_at_risk_flag

FROM composite_score c
LEFT JOIN fact_loans l
    ON c.customer_id = l.customer_id;

COMMENT ON VIEW vw_customer_risk_scorecard IS
    'Customer-level risk scorecard combining XGBoost PD scores
     with behavioral transaction features. Composite score =
     60% model PD + 40% behavioral stress. Feeds Power BI
     risk heatmap and watchlist dashboard.';


-- ════════════════════════════════════════════════════════════
-- VIEW 2: PORTFOLIO CASH FLOW VS FORECAST
-- Coverage: KPI Tracking + Metric Movement Explanation
-- Answers: How accurate are our forecasts and why do errors occur?
-- ════════════════════════════════════════════════════════════

DROP VIEW IF EXISTS vw_portfolio_cashflow_forecast CASCADE;

CREATE VIEW vw_portfolio_cashflow_forecast AS

WITH forecast_base AS (
    SELECT
        f.segment_id,
        f.month_key,
        f.actual_cash_flow,
        f.forecasted_cash_flow,
        f.upper_bound,
        f.lower_bound,
        f.is_forecast,
        f.sarimax_mape,
        f.hw_mape,
        f.sarimax_lift_pp,
        f.sarimax_bias,
        f.best_order,
        f.best_seasonal_order,
        f.n_validation_steps,

        -- Segment decomposition
        s.grade_bucket,
        s.region,

        -- Forecast error (historical periods only)
        CASE
            WHEN NOT f.is_forecast AND f.actual_cash_flow IS NOT NULL
            THEN f.actual_cash_flow - f.forecasted_cash_flow
            ELSE NULL
        END AS forecast_error,

        -- Forecast error percentage
        CASE
            WHEN NOT f.is_forecast
             AND f.actual_cash_flow IS NOT NULL
             AND ABS(f.actual_cash_flow) > 0
            THEN ROUND(
                ((f.actual_cash_flow - f.forecasted_cash_flow) /
                 ABS(f.actual_cash_flow) * 100)::NUMERIC, 4
            )
            ELSE NULL
        END AS forecast_error_pct,

        -- Confidence interval width (forecast periods only)
        CASE
            WHEN f.is_forecast
            THEN f.upper_bound - f.lower_bound
            ELSE NULL
        END AS confidence_interval_width

    FROM fact_forecasts f
    LEFT JOIN dim_segment s ON f.segment_id = s.segment_id
),

rolling_mape AS (
    -- 3-month rolling MAPE per segment
    -- Shows whether forecast accuracy is improving or deteriorating
    SELECT
        segment_id,
        month_key,
        ROUND(AVG(ABS(forecast_error_pct)) OVER (
            PARTITION BY segment_id
            ORDER BY month_key
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        )::NUMERIC, 4) AS rolling_3m_mape
    FROM forecast_base
    WHERE NOT is_forecast
),

variance_attribution AS (
    -- Attribute forecast errors to behavioral vs structural drivers
    -- Behavioral driver: months with above-median withdrawal activity
    -- Structural driver: months with above-average disbursements
    SELECT
        fb.segment_id,
        fb.month_key,
        CASE
            WHEN ABS(fb.forecast_error_pct) > 5
             AND fb.forecast_error_pct IS NOT NULL
            THEN
                CASE
                    -- High absolute error with large disbursements
                    -- = structural driver (loan volume surge)
                    WHEN fb.actual_cash_flow < fb.forecasted_cash_flow
                        THEN 'Structural — Disbursement Surge'
                    -- High absolute error with positive actuals
                    -- = behavioral driver (repayment acceleration)
                    WHEN fb.actual_cash_flow > fb.forecasted_cash_flow
                        THEN 'Behavioral — Repayment Acceleration'
                    ELSE 'Within Normal Variance'
                END
            ELSE 'Within Normal Variance'
        END AS variance_driver
    FROM forecast_base fb
    WHERE NOT fb.is_forecast
)

SELECT
    fb.segment_id,
    fb.grade_bucket,
    fb.region,
    fb.month_key,
    fb.is_forecast,

    -- Cash flow values
    fb.actual_cash_flow,
    fb.forecasted_cash_flow,
    fb.upper_bound,
    fb.lower_bound,
    fb.confidence_interval_width,

    -- Error metrics
    fb.forecast_error,
    fb.forecast_error_pct,
    rm.rolling_3m_mape,

    -- Model quality metrics
    ROUND(fb.sarimax_mape::NUMERIC, 4) AS sarimax_mape,
    ROUND(fb.hw_mape::NUMERIC, 4)      AS hw_mape,
    ROUND(fb.sarimax_lift_pp::NUMERIC, 4) AS model_lift_pp,
    fb.sarimax_bias,
    fb.best_order,
    fb.n_validation_steps,

    -- Variance attribution
    va.variance_driver,

    -- Cash flow status
    CASE
        WHEN fb.forecasted_cash_flow >= 0 THEN 'Surplus'
        ELSE 'Deficit'
    END AS cash_flow_status,

    -- Forecast vs actual direction match
    CASE
        WHEN NOT fb.is_forecast
         AND fb.actual_cash_flow IS NOT NULL
        THEN
            CASE
                WHEN SIGN(fb.actual_cash_flow) =
                     SIGN(fb.forecasted_cash_flow)
                THEN 'Direction Correct'
                ELSE 'Direction Wrong'
            END
        ELSE NULL
    END AS direction_accuracy

FROM forecast_base fb
LEFT JOIN rolling_mape rm
    ON fb.segment_id = rm.segment_id
    AND fb.month_key = rm.month_key
LEFT JOIN variance_attribution va
    ON fb.segment_id = va.segment_id
    AND fb.month_key = va.month_key;

COMMENT ON VIEW vw_portfolio_cashflow_forecast IS
    'SARIMAX cash flow forecasts vs actuals with error attribution.
     Includes 3-month rolling MAPE and variance driver classification.
     Feeds Power BI cash flow dashboard and metric movement page.';


DROP VIEW IF EXISTS vw_anomaly_detection CASCADE;

CREATE VIEW vw_anomaly_detection AS

WITH population_stats AS (
    -- Compute population-level statistics across all customers
    -- Used as the baseline since per-customer history is unavailable
    -- (PaySim is a 30-day simulation — single month per customer)
    SELECT
        AVG(withdrawal_deposit_ratio)           AS pop_avg_wd_ratio,
        STDDEV(withdrawal_deposit_ratio)        AS pop_std_wd_ratio,
        AVG(balance_vol_3m)                     AS pop_avg_vol,
        STDDEV(balance_vol_3m)                  AS pop_std_vol,
        AVG(distress_duration_6m)               AS pop_avg_distress,
        STDDEV(distress_duration_6m)            AS pop_std_distress,
        AVG(behavioral_risk_score)              AS pop_avg_risk,
        STDDEV(behavioral_risk_score)           AS pop_std_risk,
        PERCENTILE_CONT(0.95) WITHIN GROUP
            (ORDER BY withdrawal_deposit_ratio) AS wd_ratio_95th,
        PERCENTILE_CONT(0.95) WITHIN GROUP
            (ORDER BY behavioral_risk_score)    AS risk_score_95th,
        PERCENTILE_CONT(0.90) WITHIN GROUP
            (ORDER BY distress_duration_6m)     AS distress_90th
    FROM fact_features
),

anomaly_flags AS (
    SELECT
        f.customer_id,
        f.month_key,
        f.withdrawal_deposit_ratio,
        f.distress_duration_6m,
        f.closing_balance,
        f.mean_balance,
        f.balance_slope_3m,
        f.balance_vol_3m,
        f.overdraft_flag,
        f.behavioral_risk_score,
        f.cashout_proportion,
        f.total_outflow,
        f.total_inflow,

        p.pop_avg_wd_ratio,
        p.pop_std_wd_ratio,
        p.pop_avg_risk,
        p.pop_std_risk,
        p.wd_ratio_95th,
        p.risk_score_95th,
        p.distress_90th,

        -- ANOMALY TYPE 1: Withdrawal ratio spike
        -- Customer withdrawal ratio is in top 5% of population
        -- AND exceeds population mean by 2 std devs
        CASE
            WHEN f.withdrawal_deposit_ratio > p.wd_ratio_95th
             AND f.withdrawal_deposit_ratio >
                 p.pop_avg_wd_ratio + (2.0 * p.pop_std_wd_ratio)
            THEN TRUE ELSE FALSE
        END AS withdrawal_spike_flag,

        -- ANOMALY TYPE 2: Balance deterioration
        -- Closing balance is less than 10% of mean balance
        -- AND balance slope is negative (trending down)
        CASE
            WHEN f.closing_balance < f.mean_balance * 0.10
             AND f.balance_slope_3m < 0
            THEN TRUE ELSE FALSE
        END AS balance_deterioration_flag,

        -- ANOMALY TYPE 3: High behavioral risk migration
        -- Behavioral risk score in top 5% of population
        -- This replaces the time-based risk migration check
        CASE
            WHEN f.behavioral_risk_score > p.risk_score_95th
             AND f.distress_duration_6m >= p.distress_90th
            THEN TRUE ELSE FALSE
        END AS risk_migration_flag,

        -- Deviation from population mean in std devs
        CASE
            WHEN p.pop_std_wd_ratio > 0
            THEN ROUND(
                ((f.withdrawal_deposit_ratio - p.pop_avg_wd_ratio)
                 / p.pop_std_wd_ratio)::NUMERIC, 2
            )
            ELSE NULL
        END AS wd_ratio_deviation_std

    FROM fact_features f
    CROSS JOIN population_stats p
),

scored_anomalies AS (
    SELECT
        af.*,
        (af.withdrawal_spike_flag::INTEGER +
         af.balance_deterioration_flag::INTEGER +
         af.risk_migration_flag::INTEGER) AS anomaly_severity,

        CASE
            WHEN af.withdrawal_spike_flag
             AND af.balance_deterioration_flag
             AND af.risk_migration_flag
                THEN 'Multiple Anomalies'
            WHEN af.withdrawal_spike_flag
                THEN 'Withdrawal Ratio Spike'
            WHEN af.balance_deterioration_flag
                THEN 'Balance Deterioration'
            WHEN af.risk_migration_flag
                THEN 'High Risk Profile'
            ELSE 'None'
        END AS anomaly_type

    FROM anomaly_flags af
    WHERE (af.withdrawal_spike_flag::INTEGER +
           af.balance_deterioration_flag::INTEGER +
           af.risk_migration_flag::INTEGER) > 0
),

-- Aggregate to LC customer level via dim_customer mapping
customer_anomalies AS (
    SELECT
        dc.customer_id                          AS lc_customer_id,
        sa.anomaly_type,
        sa.anomaly_severity,
        sa.withdrawal_spike_flag,
        sa.balance_deterioration_flag,
        sa.risk_migration_flag,
        sa.withdrawal_deposit_ratio,
        sa.pop_avg_wd_ratio                     AS population_avg_wd_ratio,
        sa.wd_ratio_deviation_std,
        sa.closing_balance,
        sa.mean_balance,
        sa.balance_slope_3m,
        sa.distress_duration_6m,
        sa.overdraft_flag,
        sa.behavioral_risk_score,
        sa.cashout_proportion
    FROM scored_anomalies sa
    INNER JOIN dim_customer dc
        ON sa.customer_id = dc.paysim_customer_id
)

SELECT
    ca.lc_customer_id                           AS customer_id,
    ca.anomaly_type,
    ca.anomaly_severity,

    -- Behavioral metrics
    ROUND(ca.withdrawal_deposit_ratio::NUMERIC, 4)
                                                AS current_wd_ratio,
    ROUND(ca.population_avg_wd_ratio::NUMERIC, 4)
                                                AS population_avg_wd_ratio,
    ca.wd_ratio_deviation_std                   AS deviation_from_population_std,
    ROUND(ca.closing_balance::NUMERIC, 2)       AS closing_balance,
    ROUND(ca.mean_balance::NUMERIC, 2)          AS mean_balance,
    ROUND(ca.balance_slope_3m::NUMERIC, 2)      AS balance_slope_3m,
    ca.distress_duration_6m,
    ca.overdraft_flag,
    ROUND(ca.behavioral_risk_score::NUMERIC, 6) AS behavioral_risk_score,
    ROUND(ca.cashout_proportion::NUMERIC, 4)    AS cashout_proportion,

    -- Anomaly flags
    ca.withdrawal_spike_flag,
    ca.balance_deterioration_flag,
    ca.risk_migration_flag,

    -- Model scores
    p.default_probability,
    p.risk_tier,
    p.loan_amnt,
    p.grade,

    -- Loan details
    l.dti,
    l.int_rate,
    l.annual_inc,
    l.addr_state,
    l.fico_mid,
    l.interest_burden,
    l.term_months,

    -- Watchlist: high PD + severe anomaly
    CASE
        WHEN p.default_probability >= 0.50
         AND ca.anomaly_severity >= 2
        THEN TRUE ELSE FALSE
    END AS watchlist_flag

FROM customer_anomalies ca
INNER JOIN fact_predictions p
    ON ca.lc_customer_id = p.customer_id
LEFT JOIN fact_loans l
    ON ca.lc_customer_id = l.customer_id
ORDER BY
    ca.anomaly_severity DESC,
    p.default_probability DESC;

COMMENT ON VIEW vw_anomaly_detection IS
    'Population-level anomaly detection. PaySim provides 30 days of
     transaction history — insufficient for per-customer rolling baselines.
     Anomalies are therefore defined relative to population percentiles:
     top 5% withdrawal ratio, near-zero balance relative to mean,
     and top 5% behavioral risk score. Severity 1-3 scale.
     Documented limitation: production deployment would use 12+ months
     of transaction history for personalized baseline anomaly detection.';



DROP VIEW IF EXISTS vw_segment_risk_decomposition CASCADE;

CREATE VIEW vw_segment_risk_decomposition AS

WITH behavioral_stats AS (
    SELECT
        customer_id,
        AVG(behavioral_risk_score)       AS avg_behavioral_risk_score,
        AVG(withdrawal_deposit_ratio)    AS avg_withdrawal_deposit_ratio,
        AVG(distress_duration_6m)        AS avg_distress_duration,
        AVG(balance_vol_3m)              AS avg_balance_volatility,
        SUM(overdraft_flag)              AS total_overdraft_months
    FROM fact_features
    GROUP BY customer_id
),

segment_base AS (
    SELECT
        l.customer_id,
        l.loan_amnt,
        l.grade,
        l.grade_encoded,
        l.annual_inc,
        l.dti,
        l.default_flag,
        l.issue_year,
        l.addr_state,
        l.fico_mid,
        l.interest_burden,

        CASE
            WHEN l.grade IN ('A','B') THEN 'Grade_AB'
            WHEN l.grade IN ('C','D') THEN 'Grade_CD'
            ELSE 'Grade_EG'
        END AS grade_bucket,

        CASE
            WHEN l.addr_state IN
                ('CT','ME','MA','NH','RI','VT','NY','NJ','PA')
                THEN 'Northeast'
            WHEN l.addr_state IN
                ('DE','MD','DC','VA','WV','NC','SC','GA','FL',
                 'KY','TN','AL','MS','AR','LA','OK','TX')
                THEN 'South'
            WHEN l.addr_state IN
                ('OH','MI','IN','IL','WI','MN','IA','MO',
                 'ND','SD','NE','KS')
                THEN 'Midwest'
            WHEN l.addr_state IN
                ('MT','ID','WY','CO','NM','AZ','UT','NV',
                 'WA','OR','CA','AK','HI')
                THEN 'West'
            ELSE 'Other'
        END AS region,

        CASE
            WHEN l.annual_inc <  40000 THEN 'Low (<40K)'
            WHEN l.annual_inc <  75000 THEN 'Mid (40-75K)'
            WHEN l.annual_inc < 120000 THEN 'Upper-Mid (75-120K)'
            ELSE 'High (120K+)'
        END AS income_band,

        -- Model scores
        p.default_probability,
        p.risk_tier,

        -- Behavioral score from features
        ROUND(LEAST(
            COALESCE(b.avg_behavioral_risk_score, 0) * 100, 100
        )::NUMERIC, 2) AS behavioral_stress_score,

        -- Composite score: 60% PD + 40% behavioral
        ROUND((
            0.60 * p.default_probability +
            0.40 * LEAST(COALESCE(b.avg_behavioral_risk_score, 0), 1.0)
        )::NUMERIC, 6) AS composite_risk_score,

        -- Portfolio at risk flag
        CASE
            WHEN (0.60 * p.default_probability +
                  0.40 * LEAST(COALESCE(b.avg_behavioral_risk_score,0),1.0)
                 ) >= 0.45
            THEN TRUE ELSE FALSE
        END AS portfolio_at_risk_flag,

        -- Watchlist flag
        CASE
            WHEN p.default_probability >= 0.60
             AND COALESCE(b.avg_distress_duration, 0) >= 3
            THEN TRUE ELSE FALSE
        END AS watchlist_flag

    FROM fact_loans l
    INNER JOIN fact_predictions p
        ON l.customer_id = p.customer_id
    LEFT JOIN behavioral_stats b
        ON l.customer_id = b.customer_id
),

portfolio_totals AS (
    SELECT
        COUNT(*)                AS total_loans,
        SUM(loan_amnt)          AS total_book_value,
        SUM(default_flag)       AS total_defaults,
        AVG(default_probability) AS portfolio_avg_pd,
        SUM(CASE WHEN portfolio_at_risk_flag
            THEN loan_amnt ELSE 0 END) AS total_par_value
    FROM segment_base
),

segment_metrics AS (
    SELECT
        sb.grade_bucket,
        sb.region,
        sb.income_band,

        COUNT(*)                        AS total_loans,
        SUM(sb.default_flag)            AS actual_defaults,
        SUM(CASE WHEN sb.portfolio_at_risk_flag
            THEN 1 ELSE 0 END)          AS at_risk_loans,
        SUM(CASE WHEN sb.watchlist_flag
            THEN 1 ELSE 0 END)          AS watchlist_loans,

        SUM(sb.loan_amnt)               AS total_exposure,
        AVG(sb.loan_amnt)               AS avg_loan_size,
        SUM(CASE WHEN sb.portfolio_at_risk_flag
            THEN sb.loan_amnt ELSE 0
            END)                        AS par_value,

        ROUND(AVG(sb.default_probability)::NUMERIC, 6)
                                        AS avg_default_probability,
        ROUND((SUM(sb.default_flag)::NUMERIC /
               NULLIF(COUNT(*), 0) * 100)::NUMERIC, 4)
                                        AS actual_default_rate_pct,
        ROUND(AVG(sb.behavioral_stress_score)::NUMERIC, 2)
                                        AS avg_behavioral_stress_score,
        ROUND(AVG(sb.dti)::NUMERIC, 2)  AS avg_dti,
        ROUND(AVG(sb.fico_mid)::NUMERIC, 1) AS avg_fico,
        ROUND(AVG(sb.interest_burden)::NUMERIC, 6)
                                        AS avg_interest_burden,

        ROUND((SUM(sb.loan_amnt) /
               MAX(pt.total_book_value) * 100)::NUMERIC, 4)
                                        AS book_concentration_pct,
        ROUND((SUM(CASE WHEN sb.portfolio_at_risk_flag
                   THEN sb.loan_amnt ELSE 0 END) /
               MAX(pt.total_book_value) * 100)::NUMERIC, 4)
                                        AS par_concentration_pct

    FROM segment_base sb
    CROSS JOIN portfolio_totals pt
    GROUP BY sb.grade_bucket, sb.region, sb.income_band
)

SELECT
    sm.*,
    sm.grade_bucket || '_' || sm.region AS segment_id,

    CASE
        WHEN sm.par_concentration_pct >= 15 THEN 'CONCENTRATION RISK'
        WHEN sm.par_concentration_pct >= 10 THEN 'ELEVATED CONCENTRATION'
        ELSE 'Normal'
    END AS concentration_flag,

    CASE
        WHEN sm.avg_default_probability >= 0.50 THEN 'Critical'
        WHEN sm.avg_default_probability >= 0.35 THEN 'High Risk'
        WHEN sm.avg_default_probability >= 0.20 THEN 'Moderate Risk'
        ELSE 'Low Risk'
    END AS segment_risk_label,

    ROUND((sm.at_risk_loans::NUMERIC /
           NULLIF(sm.total_loans, 0) * 100)::NUMERIC, 2)
                                        AS at_risk_rate_pct

FROM segment_metrics sm
ORDER BY sm.par_value DESC, sm.avg_default_probability DESC;