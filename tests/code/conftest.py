import sys
from pathlib import Path

import pytest

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn_src.data_loader import load_data


@pytest.fixture
def dataset_load():
    """
    Fixture to load the dataset for testing.

    """
    return load_data()
