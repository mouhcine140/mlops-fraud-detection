# Exemple de run

J'ai laissé les sorties d'un vrai run ici (pipeline complet sur le dataset par défaut) pour que ce soit vérifiable sans avoir à me faire confiance sur parole : `data_quality_report.json` (14 checks, tous passés),`training_metrics.json` (les métriques du tableau du README principal), et `drift_report.json` (voir la note dans `drift.py` sur pourquoi les features cumulatives ressortent "critiques" avec le split par défaut).

Normalement ces fichiers sont générés à chaque run et pas commités (voir `.gitignore`), mais je garde ceux-là en dur. `bash scripts/run_pipeline_local.sh` les régénère.
