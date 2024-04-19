from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (mean_squared_error, r2_score,
                             root_mean_squared_error)
from typing import Dict

import mlflow
import numpy as np
import pandas as pd

EXPERIMENT_NAME: str = 'end-to-end-project'
ARTIFACTS_PATH: Path = Path('artifacts')
MODEL_FOLDER_NAME: str = 'model'
MODEL_PATH: Path = ARTIFACTS_PATH / MODEL_FOLDER_NAME
TARGET_VARIABLE: str = 'MedHouseVal'


def mkdirs() -> None:
    """
    Cria os diretórios necessários para executar o experimento.
    """
    if not ARTIFACTS_PATH.exists():
        ARTIFACTS_PATH.mkdir()


def log_input(dataframe: pd.DataFrame, context: str) -> None:
    """
    Salva um conjunto de dados no sistema de arquivos e registra-o no mlflow
    """
    filepath = ARTIFACTS_PATH / f'{context}.csv'

    dataframe.to_csv(filepath, index=False)

    mlflow.log_input(
        mlflow.data.from_pandas(
            dataframe,
            source=filepath,
            targets=TARGET_VARIABLE,
            name=context
        ),
        context=context
    )


def load_raw_input_dataframe() -> pd.DataFrame:
    """
    Une preditores e variável-alvo num só dataframe pela conveniência
    de eu já ter um modelo de projeto funcionando assim.
    """
    df = fetch_california_housing(as_frame=True)
    return df.data.merge(df.target, left_index=True, right_index=True)


def xy(dataframe: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Separa preditores da variável-alvo.
    """
    x = dataframe.drop(TARGET_VARIABLE, axis=1)
    y = dataframe[TARGET_VARIABLE]
    return x, y


def regression_metrics(y_actual, y_predicted) -> Dict[str, float]:
    """
    As métricas da regressão num formato conveniente para adicioná-las
    to tracking do MLflow. Porém, aqui servem mais como demonstração,
    pois o MLflow já calcula essas métricas, e várias outras, por conta
    própria.
    """
    return {
        'mse': mean_squared_error(y_actual, y_predicted),
        'r2': r2_score(y_actual, y_predicted),
        'rmse': root_mean_squared_error(y_actual, y_predicted)
    }


def custom_metric_1_definition(dataframe, _):
    """
    Nesta definição eu calculo a métrica a partir do dataframe que contém
    as previsões e os valores reais da variável-alvo, o primeiro argumento.
    """
    y_predicted = dataframe['prediction']
    y_actual = dataframe['target']
    return np.sqrt(np.abs(np.power(y_predicted - y_actual, 3)))


def custom_metric_2_definition(_, builtin_metrics):
    """
    Nesta definição eu calculo a métrica a partir de métricas pré-existentes
    e disponibilizadas no segundo argumento.
    """
    return builtin_metrics['r2_score'] / 2
