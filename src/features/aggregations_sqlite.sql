-- Agrégations comportementales (historique utilisateur / carte) via window functions. Version SQLite.

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

    -- SQLite n'autorise pas COUNT(DISTINCT ...) en window function, d'où la
    -- sous-requête corrélée. Postgres n'a pas cette limite (voir
    -- aggregations_postgres.sql).

    (
        SELECT COUNT(DISTINCT t2.merchant_country)
        FROM raw_transactions t2
        WHERE t2.card_id = t.card_id AND t2.timestamp <= t.timestamp
    ) AS card_distinct_countries_so_far,

    (
        julianday(t.timestamp) - julianday(
            LAG(t.timestamp) OVER (PARTITION BY t.user_id ORDER BY t.timestamp)
        )
    ) * 86400.0 AS seconds_since_last_user_txn

FROM raw_transactions t
ORDER BY t.timestamp;
