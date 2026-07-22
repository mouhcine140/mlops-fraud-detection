# Exemple de run

Sorties réelles d'une exécution complète du pipeline (`bash scripts/run_pipeline_local.sh`) sur le dataset synthétique généré par défaut (20 000 transactions, 1.5% de fraude). Committées ici à titre d'exemple, contrairement aux données/modèles/rapports générés à chaque run (voir `.gitignore`).

- `data_quality_report.json` — sortie de `src/validation/data_quality.py` : 14 checks, tous passés
- `training_metrics.json` — métriques du run d'entraînement (ROC-AUC, PR-AUC, precision/recall au seuil optimisé)
- `drift_report.json` — sortie de `src/monitoring/drift.py`, comparant la première et la seconde moitié du dataset (les features cumulatives comme `user_txn_count_before` montrent naturellement du drift sur un split temporel, c'est le comportement attendu)

Pour régénérer ces fichiers avec tes propres données : `bash scripts/run_pipeline_local.sh`.
