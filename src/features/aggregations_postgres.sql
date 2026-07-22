-- Agrégations comportementales (historique user/carte) via window functions.
-- Variante Postgres - voir aggregations_sqlite.sql pour la démo locale.

SELECT
    t.transaction_id,
    t.user_id,
    t.card_id,
    t.timestamp,
    t.amount,
    t.merchant_category,
    t.merchant_country,
    t.device_type,
    t.transaction_type,
    t.is_fraud,

    COUNT(*) OVER (
        PARTITION BY t.user_id
        ORDER BY t.timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS user_txn_count_before,

    AVG(t.amount) OVER (
        PARTITION BY t.user_id
        ORDER BY t.timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS user_avg_amount_before,

    COUNT(DISTINCT t.merchant_country) OVER (
        PARTITION BY t.card_id
        ORDER BY t.timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS card_distinct_countries_so_far,

    EXTRACT(EPOCH FROM (
        t.timestamp - LAG(t.timestamp) OVER (PARTITION BY t.user_id ORDER BY t.timestamp)
    )) AS seconds_since_last_user_txn

FROM raw_transactions t
ORDER BY t.timestamp;
