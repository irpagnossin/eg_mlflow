from mlflow.metrics import make_metric
from mlflow.models import MetricThreshold
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import click
import mlflow

from helpers import (mkdirs, log_input, load_raw_input_dataframe, xy,
                     regression_metrics, ARTIFACTS_PATH, MODEL_PATH,
                     EXPERIMENT_NAME)


def create_baseline_model(x_train, y_train, x_test, y_test):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', DummyRegressor())
    ])

    model.fit(x_train, y_train)
    mlflow.sklearn.log_model(model, artifact_path=MODEL_PATH.name)

    y_predicted = model.predict(x_test)
    metrics = regression_metrics(y_test, y_predicted)

    return model, metrics


@click.command()
@click.option('--random_state', default=42, help='Random state')
def main(random_state):
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME,
                                       run_name='dummy-baseline-model')

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        raw_input = load_raw_input_dataframe()
        RAW_INPUT_FILENAME = ARTIFACTS_PATH / 'raw_input.csv'
        raw_input.to_csv(RAW_INPUT_FILENAME, index=False)

        mlflow.set_tags({
            'project': 'mlflow/udemy',
            'client': 'myself',
            'purpose': 'learn',
            'team': 'solo',
            'ml-task': 'regression',
            'model': 'DummyRegressor'
        })

        mlflow.log_params({
            'random_state': random_state,
        })

        train, test = train_test_split(raw_input, random_state=random_state)
        log_input(train, context='train')
        log_input(test, context='test')

        x_train, y_train = xy(train)
        x_test, y_test = xy(test)

        _, metrics = create_baseline_model(x_train, y_train, x_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(ARTIFACTS_PATH)


if __name__ == '__main__':
    mkdirs()
    main()

