import mlflow
import os

from dotenv import find_dotenv, load_dotenv
from helpers import EXPERIMENT_NAME


load_dotenv(find_dotenv())

mlflow.set_tracking_uri(os.environ.get('MLFLOW_SERVER_URL', '127.0.0.01'))

baseline_run_id = os.environ.get('BASELINE_RUN_ID')
if baseline_run_id:
    mlflow.projects.run(
        uri='.',
        entry_point='candidate_model_training',
        experiment_name=EXPERIMENT_NAME,
        env_manager='conda',
        parameters={'baseline_run_id': baseline_run_id}
    )
else:
    mlflow.projects.run(
        uri='.',
        entry_point='baseline_model_training',
        experiment_name=EXPERIMENT_NAME,
        env_manager='conda',
    )


