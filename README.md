# Détection de fraude bancaire (MLOps)

Pipeline de bout en bout : ingestion, contrôle qualité, feature engineering SQL + pandas, entraînement XGBoost avec tracking MLflow, une API pour servir les prédictions, et un monitoring de drift, orchestré par Airflow.
Le but ici n'était pas de sortir le meilleur modèle possible, mais de construire tout ce qu'il y a autour : un pipeline qui tourne de bout en bout, testé, reproductible, avec un vrai monitoring derrière.


## Contexte

Je voulais un projet qui couvre tout le cycle MLOps, pas juste "entraîner un modèle" : ingestion, qualité de données, features réutilisables entre batch et serving, tracking d'expériences, une API, et un système qui
prévient quand les données de prod s'écartent de l'entraînement. La fraude bancaire s'y prête bien : peu de cas positifs (~1.5%), et des patterns qui évoluent dans le temps donc ça force à penser au ré-entraînement.

## Architecture

```
                      ┌─────────────────────┐
                      │   Airflow DAG        │
                      │ (@daily)              │
                      └──────────┬───────────┘
                                 │
   ┌──────────┐   ┌──────────┐  │  ┌──────────┐   ┌───────────┐   ┌────────────┐
   │  Ingest  │──▶│ Validate │──┴─▶│ Features │──▶│  Train    │──▶│ Quality    │
   │  (CSV)   │   │ (checks) │     │ (SQL+py) │   │ (MLflow)  │   │ gate       │
   └────┬─────┘   └────┬─────┘    └────┬─────┘   └─────┬─────┘   └─────┬──────┘
        │              │                │               │               │
        ▼              ▼                ▼               ▼               ▼
   PostgreSQL     reports/           PostgreSQL      mlruns/       reports/
   raw_transactions  data_quality_    features        MLflow        drift_report.json
                      report.json                     tracking +           │
                                                        registry            ▼
                                                              ┌──────────────────┐
                                                              │  Drift check      │
                                                              │  (PSI)             │
                                                              └──────────────────┘

                      ┌─────────────────────┐
                      │  FastAPI (/predict)  │◀── charge models/model.joblib
                      └─────────────────────┘
```

Chaque flèche du DAG correspond à une tâche Airflow indépendante (voir `dags/fraud_detection_pipeline.py`), qui appelle une fonction pure de `src/` — la logique métier ne dépend donc jamais d'Airflow et reste testable en isolation (voir `tests/`).

## Stack technique

- Python 3.11
- SQLite (démo) / PostgreSQL (docker-compose)
- SQL (window functions) + pandas pour les features
- XGBoost + MLflow (tracking et registry)
- FastAPI pour le serving
- Airflow (TaskFlow API) pour l'orchestration
- PSI et data quality checks maison, pas de lib externe (Evidently / Great Expectations) - pas nécessaire à cette échelle

## Structure du repo

```
mlops-fraud-detection/
├── .github/workflows/ci.yml          # CI : tests + smoke test du pipeline
├── dags/
│   └── fraud_detection_pipeline.py   # DAG Airflow (TaskFlow API)
├── src/
│   ├── common/                       # config centralisée + accès DB
│   ├── ingestion/                    # génération dataset + chargement en base
│   ├── validation/                   # data quality checks
│   ├── features/                     # SQL aggregations + feature engineering pandas
│   ├── training/                     # entraînement + tracking MLflow
│   ├── serving/                      # API FastAPI
│   └── monitoring/                   # détection de drift (PSI)
├── tests/                            # tests pytest (features, validation, API, drift)
├── examples/                         # sorties d'un run réel (preuve de fonctionnement)
├── data/raw/                         # généré par generate_sample_data.py (non commité)
├── scripts/
│   ├── run_pipeline_local.sh         # exécute tout le pipeline sans docker
│   └── init-db.sql                   # init des bases Postgres (docker-compose)
├── docker-compose.yml                # Postgres + Airflow + API
├── Dockerfile / Dockerfile.airflow
├── requirements.txt
└── pytest.ini
```

## Installation & lancement

### Option A : démo rapide en local (sans Docker)

Nécessite Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # garde DATABASE_URL=sqlite:... par défaut

bash scripts/run_pipeline_local.sh
```

Ce script génère les données, les ingère, les valide, construit les features, entraîne le modèle et calcule le drift — dans cet ordre, avec les logs de chaque étape.

Pour lancer l'API ensuite :

```bash
uvicorn src.serving.api:app --reload
# puis http://localhost:8000/docs pour la doc interactive (Swagger)
```

Pour visualiser les runs MLflow :

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
# http://localhost:5000
```

### Option B : environnement complet (Docker + Airflow + Postgres)

```bash
docker-compose up --build
```

- Airflow UI : http://localhost:8080 (admin / admin) — déclencher manuellement le DAG `fraud_detection_pipeline`
- API : http://localhost:8000/docs
- PostgreSQL exposé sur `localhost:5432` (fraud_user / fraud_pass / fraud_db)

Le DAG tourne quotidiennement (`@daily`) une fois activé dans l'UI Airflow.

### Utiliser le vrai dataset Kaggle plutôt que les données synthétiques

Remplacer `data/raw/fraud_transactions_sample.csv` par un CSV respectant le même schéma (voir `EXPECTED_COLUMNS` dans `src/ingestion/ingest.py`), ou adapter `src/ingestion/ingest.py` pour mapper les colonnes du dataset choisi (ex : [Credit Card Fraud Detection, ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) ou [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)).

## Comment c'est construit

La génération de données synthétiques (`generate_sample_data.py`) simule des patterns de fraude assez marqués (montants élevés, transactions de nuit, pays inhabituels) avec une seed fixe, pour que tout le monde puisse
reproduire le même run sans compte Kaggle.

Avant de passer à la suite, une dizaine de règles de qualité (`data_quality.py`) vérifient schéma, doublons, nulls, plage de montants, taux de fraude plausible. Les critiques bloquent le pipeline, les warnings
sont juste loggés.

Le feature engineering se fait en deux temps : les agrégations d'historique utilisateur/carte tournent en SQL (window functions, variante SQLite ou Postgres selon le dialecte), puis pandas prend le relais pour les features
dérivées.

XGBoost pour l'entraînement, avec `scale_pos_weight` pour le déséquilibre de classes et un seuil optimisé sur le F1 plutôt que 0.5. Chaque run est tracké dans MLflow.

L'API recharge le modèle depuis `model.joblib` et applique le même feature engineering à une transaction reçue en ligne. Le monitoring de drift calcule un PSI par feature entre référence et batch courant. Et le DAG Airflow enchaîne tout ça, avec une quality gate qui peut rejeter un modèle sous les seuils métier.





## Résultats

Sur le dataset synthétique fourni (20 000 transactions, 1.5% de fraude), un run complet du pipeline donne ces chiffres :

| Métrique | Valeur observée |
|---|---|
| ROC-AUC | 0.996 |
| PR-AUC | 0.872 |
| Precision (seuil optimisé) | 0.831 |
| Recall (seuil optimisé) | 0.817 |

Ces sorties (rapport qualité, métriques, drift) sont dans `examples/`, tu peux les régénérer avec `bash scripts/run_pipeline_local.sh`. Ces chiffres sont hauts parce que les patterns de fraude synthétiques sont assez
marqués - avec de la vraie fraude ce serait sûrement plus dur. Le but ici c'était le pipeline, pas la perf du modèle.

## Tests

```bash
pytest
```

Couvre : les règles de qualité de données (cas valides et invalides), le feature engineering (features temporelles, ratios, encodage, alignement de schéma en inférence), et le contrat de l'API (`/health`, `/predict`, validation des entrées).

## Ce que je ferais différemment

Les règles de qualité sont maison, ce qui va bien pour une dizaine de règles mais je passerais à Great Expectations ou Soda Core si ça devait grossir. Pareil pour le drift : le PSI en batch fait le taf pour la démo,
mais une vraie solution (Evidently, whylogs) donnerait des rapports visuels et de l'alerting, ce qui manque clairement ici.

Les features d'historique utilisateur sont recalculées à chaque batch - un vrai feature store (Feast) éviterait ça en partageant le calcul entre entraînement et serving. Niveau CI, les tests et un smoke test tournent sur chaque push, mais ça ne couvre ni le build Docker ni un vrai déploiement.
