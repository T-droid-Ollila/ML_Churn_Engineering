from pathlib import Path

import great_expectations as ge
import pandas as pd
import pytest

# sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pytest_addoption(parser):
    # parser arguments
    parser.addoption(
        "--dataset-loc",
        action="store",
        default="churn_data/train_churn.csv",
        help="Path to the dataset",
    )


@pytest.fixture(scope="session")
def expect_data(request):
    # Load the dataset

    dataset_loc = request.config.getoption("--dataset-loc")

    # Initialize Great Expectations context
    # Read the dataset into a pandas DataFrame
    context = ge.get_context()

    # Convert to Path object
    dataset_path = Path(dataset_loc)

    if dataset_loc is None:
        raise ValueError("Please provide the path to the dataset using --dataset-loc option.")
    # root_logger.info("Please provide the path to the dataset using --dataset-loc option.")

    # Load the dataset using Great Expectations
    # Method 1: If you have a DataContext

    ge_df = pd.read_csv(dataset_path)
    # Define source , first with context to give us access
    df_source = context.data_sources.add_pandas(name="my_pandas_datasource")
    data_asset = df_source.add_dataframe_asset(name="my_data_asset")
    data_batch = data_asset.add_batch_definition_whole_dataframe(name="Whole file batch")
    # Convert the DataFrame to a Great Expectations DataFrame
    batch = data_batch.get_batch(batch_parameters={"dataframe": ge_df})

    return batch
