# Exemplo de uso do MLflow no GCP

## Instruções
1. Crie uma cópia do arquivo `eg.env`, nomeando-o `.env`.
1. Configure a URL do servidor MLflow.
1. Execute `python run.py`. Isso gerará um modelo _dummy_ que servirá de referência (AKA _baseline model_).
1. Copie o ID da execução no passo acima para a variável `BASELINE_RUN_ID`, em `.env`.
1. Re-execute `python run.py`, que dessa vez criará um modelo melhor e usará o modelo de referência para avaliar seu desempenho.
