from pathlib import Path

import great_expectations as ge
import pandas as pd
import pytest


def pytest_addoption(parser):
    # parser arguments
    parser.addoption(
        "--dataset-loc",
        action="store",
        default="churn_data/train_churn.csv",
        help="Path to the dataset",
    )


@pytest.fixture(scope="session")
def context():
    return ge.get_context()


@pytest.fixture(scope="session")
def expect_data(request, context):
    # Load the dataset

    dataset_loc = request.config.getoption("--dataset-loc")

    # Initialize Great Expectations context
    # Read the dataset into a pandas Dataframe

    # Convert to Path object
    dataset_path = Path(dataset_loc)

    if dataset_loc is None:
        raise ValueError("Please provide the path to the dataset using --dataset-loc option.")

    # Load the dataset using Great Expectations
    # Method 1: If you have a DataContext
    datasource_name = "my_pandas_datasource"
    asset_name = "my_data_asset"

    ge_df = pd.read_csv(dataset_path)
    # Define source , first with context to give us access
    df_source = context.sources.add_pandas(name=datasource_name)
    data_asset = df_source.add_dataframe_asset(name=asset_name)
    # Convert the DataFrame to a Great Expectations DataFrame
    batch = data_asset.build_batch_request(dataframe=ge_df)

    return batch
