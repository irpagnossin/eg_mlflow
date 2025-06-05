from mlflow.metrics import make_metric
from mlflow.models import MetricThreshold
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import click
import mlflow

from helpers import (mkdirs, log_input, load_raw_input_dataframe, xy,
                     regression_metrics, custom_metric_1_definition,
                     custom_metric_2_definition, ARTIFACTS_PATH,
                     MODEL_PATH, TARGET_VARIABLE, EXPERIMENT_NAME)


def create_candidate_model(regularization, kernel, degree,
                           x_train, y_train, x_test, y_test):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR(C=regularization, kernel=kernel, degree=degree))
    ])

    model.fit(x_train, y_train)
    mlflow.sklearn.log_model(model, artifact_path=MODEL_PATH.name)

    y_predicted = model.predict(x_test)
    metrics = regression_metrics(y_test, y_predicted)
    mlflow.log_metrics(metrics)


@click.command()
@click.option('--regularization', default=1.0, help='Regularization factor')
@click.option('--kernel', default='rbf', help='SVM kernel')
@click.option('--degree', default=3, help='Degree for the "poly" kernel')
@click.option('--random_state', default=42, help='Random state')
@click.option('--baseline_run_id', default='86d62c3ccd9546fe880c6d7cee9f87a0',
              help='Run ID which generated baseline model')
def main(regularization, kernel, degree, random_state, baseline_run_id):

    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

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
            'model': 'SVM'
        })

        mlflow.log_params({
            'random_state': random_state,
            'regularization': regularization,
            'kernel': kernel,
            'degree': degree,
            'baseline_run_id': baseline_run_id
        })

        train, test = train_test_split(raw_input, random_state=random_state)
        log_input(train, context='train')
        log_input(test, context='test')

        x_train, y_train = xy(train)
        x_test, y_test = xy(test)

        create_candidate_model(regularization, kernel, degree,
                               x_train, y_train, x_test, y_test)

        mlflow.log_artifacts(ARTIFACTS_PATH)

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

        previous_run = mlflow.get_run(baseline_run_id)
        baseline_model_uri = previous_run.info.artifact_uri + "/model"

        mlflow.evaluate(
            model=mlflow.get_artifact_uri(MODEL_PATH.name),
            data=test,
            targets=TARGET_VARIABLE,
            model_type='regressor',
            baseline_model=baseline_model_uri,
            extra_metrics=[custom_metric_1, custom_metric_2],
            validation_thresholds=thresholds
        )


if __name__ == '__main__':
    mkdirs()
    main()
