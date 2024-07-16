# MLflow

MLflow em Docker usando Postgres como base de metadados e Google Cloud Storage (GCS) como repositório de artefatos (e.g., modelos).

1. `cp env.sample .env` e configure-o.
1. Salve a conta de serviço do MLflow como `./secrets/service-account.json`.
1. `docker compose up -d`

A conta de serviço deve ter papel _Storage Object Admin_ (`roles/storage.objectAdmin`) no bucket.

obs.: o ambiente de treinamento também deve instalar os mesmos `requirements.txt`.
