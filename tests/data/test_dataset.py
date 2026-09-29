def test_dataset(expect_data, context):
    """Test dataset quality and integrity."""
    batch_request = expect_data

    column_list = [
        "security_no",
        "membership_category",
        "joined_through_referral",
        "medium_of_operation",
        "points_in_wallet",
        "churn_risk_score",
    ]

    suite_name = "churn_data_suite"

    # 1. Create the (empty) suite first — not gx.expectations(name=...)
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    # 2. Get a Validator bound to both the batch and the suite name
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # 3. Add expectations as direct method calls on the validator —
    #    snake_case methods, not gx.expectations.ExpectX(...) class instances
    for col in column_list:
        validator.expect_column_values_to_not_be_null(column=col)

    validator.expect_column_values_to_be_unique(column="security_no")

    expected_dtypes = {
        "membership_category": "object",
        "points_in_wallet": "float64",
        "churn_risk_score": "int64",
    }
    for col, dtype in expected_dtypes.items():
        validator.expect_column_values_to_be_of_type(column=col, type_=dtype)

    # 4. Persist everything accumulated above into the suite
    validator.save_expectation_suite(discard_failed_expectations=False)

    # 5. Actually run validation via a Checkpoint — not batch.validate(suite)
    checkpoint = context.add_or_update_checkpoint(
        name="churn_data_checkpoint",
        validations=[
            {"batch_request": batch_request, "expectation_suite_name": suite_name},
        ],
    )
    checkpoint_result = checkpoint.run()

    assert checkpoint_result.success == False
