FROM ghcr.io/mlflow/mlflow:v3.0.0rc2

WORKDIR /app

ENV BACKEND_STORE_URI=${BACKEND_STORE_URI}
ENV DEFAULT_ARTIFACT_ROOT=${DEFAULT_ARTIFACT_ROOT}
ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/service-account.json

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 5000

