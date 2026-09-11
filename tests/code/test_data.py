# Test to ensure that the data is loaded correctly
def data_retrieval(dataset_load):
    """
    Test to ensure that the data is loaded correctly.
    """
    X_train, X_test, y_train, y_test, df = dataset_load()
    assert not df.empty, "DataFrame is empty."
    assert len(X_train) > 0 and len(X_test) > 0, "Train/Test split failed."
    assert len(y_train) > 0 and len(y_test) > 0, "Train/Test split failed."
