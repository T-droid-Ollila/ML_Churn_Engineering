import great_expectations as gx


def test_dataset(expect_data):
    """Test dataset quality and integrity."""
    batch = expect_data

    column_list = [
        "security_no",
        "membership_category",
        "joined_through_referral",
        "medium_of_operation",
        "points_in_wallet",
        "churn_risk_score",
    ]

    suite = gx.ExpectationSuite(name="churn_data_suite")

    # Check for missing values — one Expectation object per column
    for col in column_list:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    # Check for duplicate values in a single column
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="security_no"))

    # Ensure data types match expected types
    expected_dtypes = {
        "membership_category": "object",
        "points_in_wallet": "float64",
        "churn_risk_score": "int64",
    }
    for col, dtype in expected_dtypes.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(column=col, type_=dtype)
        )

    # Validate the whole batch against the whole suite in one call
    results = batch.validate(suite)

    assert results.success == False
