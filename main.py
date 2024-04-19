from mlflow.metrics import make_metric
from mlflow.models import MetricThreshold
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (mean_squared_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from typing import Dict

import click
import mlflow
import numpy as np
import pandas as pd

ARTIFACTS_PATH: Path = Path('./artifacts')
BASELINE_MODEL_PATH: Path = ARTIFACTS_PATH.joinpath('baseline_model')
CANDIDATE_MODEL_PATH: Path = ARTIFACTS_PATH.joinpath('model')
TARGET_VARIABLE: str = 'MedHouseVal'


def mkdirs() -> None:
    if not BASELINE_MODEL_PATH.exists():
        BASELINE_MODEL_PATH.mkdir()

    if not CANDIDATE_MODEL_PATH.exists():
        CANDIDATE_MODEL_PATH.mkdir()

    if not ARTIFACTS_PATH.exists():
        ARTIFACTS_PATH.mkdir()


def log_input(dataframe: pd.DataFrame, context: str) -> None:
    filepath = Path(ARTIFACTS_PATH, f'{context}.csv')

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


def apply_baseline_model(x_train, y_train, x_test, y_test):
    baseline_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', DummyRegressor())
    ])
    baseline_model.fit(x_train, y_train)

    mlflow.sklearn.log_model(baseline_model,
                             artifact_path=BASELINE_MODEL_PATH.name)

    y_predicted = baseline_model.predict(x_test)

    regression_metrics(y_test, y_predicted)


def apply_candidate_model(regularization, kernel, degree,
                         x_train, y_train, x_test, y_test):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(C=regularization, kernel=kernel, degree=degree))
    ])
    model.fit(x_train, y_train)
    mlflow.sklearn.log_model(model, artifact_path=CANDIDATE_MODEL_PATH.name)

    y_predicted = model.predict(x_test)

    metrics = regression_metrics(y_test, y_predicted)
    mlflow.log_metrics(metrics)


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


@click.command()
@click.option('--regularization', default=1.0, help='Regularization factor')
@click.option('--kernel', default='rbf', help='SVM kernel')
@click.option('--degree', default=3, help='Degree for the "poly" kernel')
@click.option('--random_state', default=42, help='Random state')
def main(regularization, kernel, degree, random_state):

    experiment = mlflow.set_experiment(experiment_name='end-to-end-project')

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        raw_input = load_raw_input_dataframe()
        RAW_INPUT_FILENAME = Path(ARTIFACTS_PATH, 'raw_input.csv')
        raw_input.to_csv(RAW_INPUT_FILENAME, index=False)

        mlflow.set_tags({
            'project': 'mlflow/udemy',
            'client': 'myself',
            'purpose': 'learn',
            'team': 'solo',
            'ml-task': 'regression',
            'model': 'SVM'
        })

        mlflow.log_params({
            'random_state': random_state,
            'regularization': regularization,
            'kernel': kernel,
            'degree': degree
        })

        train, test = train_test_split(raw_input, random_state=random_state)
        log_input(train, context='train')
        log_input(test, context='test')

        x_train, y_train = xy(train)
        x_test, y_test = xy(test)

        apply_baseline_model(x_train, y_train, x_test, y_test)

        apply_candidate_model(regularization, kernel, degree,
                              x_train, y_train, x_test, y_test)

        thresholds = {
            'r2_score': MetricThreshold(
                threshold=0.5,
                min_absolute_change=0.01,
                greater_is_better=True
            ),
        }

        custom_metric_1 = make_metric(eval_fn=custom_metric_1_definition,
                                      greater_is_better=True,
                                      name='custom-metric-1')

        custom_metric_2 = make_metric(eval_fn=custom_metric_2_definition,
                                      greater_is_better=True,
                                      name='custom-metric-2')
        mlflow.evaluate(
            model=mlflow.get_artifact_uri(CANDIDATE_MODEL_PATH.name),
            data=test,
            targets=TARGET_VARIABLE,
            model_type='regressor',
            baseline_model=mlflow.get_artifact_uri(CANDIDATE_MODEL_PATH.name),
            extra_metrics=[custom_metric_1, custom_metric_2],
            validation_thresholds=thresholds
        )


if __name__ == '__main__':
    mkdirs()
    main()
