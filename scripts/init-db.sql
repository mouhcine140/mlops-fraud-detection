-- Exécuté par docker-compose (postgres) au premier démarrage.
-- airflow_db séparée de fraud_db pour isoler métadonnées Airflow et
-- données applicatives.

CREATE DATABASE airflow_db;

CREATE USER airflow WITH PASSWORD 'airflow_pass';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow;
