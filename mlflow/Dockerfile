FROM ghcr.io/mlflow/mlflow:v2.0.1

WORKDIR /app

ENV BACKEND_STORE_URI=${BACKEND_STORE_URI}
ENV DEFAULT_ARTIFACT_ROOT=${DEFAULT_ARTIFACT_ROOT}
ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/service-account.json

COPY requirements.txt .
COPY entrypoint.sh entrypoint.sh

RUN chmod +x entrypoint.sh \
  && pip install --upgrade pip \
  && pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["/app/entrypoint.sh"]

