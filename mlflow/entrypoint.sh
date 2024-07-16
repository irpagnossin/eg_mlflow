#!/bin/sh
mlflow server --host 0.0.0.0 --default-artifact-root "${DEFAULT_ARTIFACT_ROOT}" --backend-store-uri "${BACKEND_STORE_URI}"
