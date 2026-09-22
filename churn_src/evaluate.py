import os
import argparse
from mlflow.entities import ViewType
from mlflow.client import MlflowClient
from churn_src.feature_engineer import ChurnPreprocessor
from churn_src.config import root_logger
import pandas as pd
import numpy as np
import datetime
from sklearn import metrics

root_logger.setLevel('ERROR')

client=MlflowClient()

filter_string = "name LIKE '%decision%'"

runs = client.search_runs(run_view_type=ViewType.ACTIVE_ONLY,
                          filter_string=filter_string,
                          max_results=5,
                          order_by=['metrics.test_f1_score ASC'])



