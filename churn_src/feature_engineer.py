# This code contains feature engineering section for our pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder


def feature_engineer_full_workflow(x, y):
    """
    This function includes multiple feature engineering steps

    to clean and transform the dataset.

    The first part applies a mapping to the feedback column

    to consolidate categories into positive, negative, and neutral.

    The second part assigns "Yes" or "No" to the joined_through_referral

    The third part converts the last_visit_time column to a datetime format

    and extracts the time component.

    The fourth part replaces 'Unknown' in the gender column with 'M'

    because in the earlier sampling stages we Female we the larger proportion

    so to attain a more balanced dataset we replace the 'Unknown' with "M" to rep male

    The fifth part removes negative values in the avg_time_spent column and replaces them with NaN.

    The sixth part We can see that there are negative values

    in the days_since_last_login column

    which is not possible and we can coerce those values to NaN

    Then replace them with the median of the column.

    The seventh part We coerce the ? in medium of operation to NaN

    then replace the NaN with the Smartphone

    The eigth part Let's coerce the error value in the average frequency login days column

    To numeric and replace it with NaN

    Then compute median on the now-clean numeric column.

    The ninth part extracts month and year from the

    joining_date column and creates new columns joining_month and joining_year.

    The tenth part drops the joining_date and referral_id columns from the dataset.

    The eleventh part applies quantile-based binning to the avg_frequency_login_days

    and avg_time_spent columns.

    The twelveth part applies fixed-width binning to the age colum

    The 13th part is to Fill missing values in a DataFrame using pandas.

    Categorical features are filled with the mode, while numerical features

    are filled with the median.

    The 14th part Identify features that are highly correlated with the target variable.

    The 15th Implementation of mutual information for feature selection

    Calculate mutual information scores for all features with respect to the target variable

    We write a function based to select the top features

    If mutual information scores above a certain threshold of >= 0.3

    Apply log transformation to skewed numerical features in the DataFrame.

    This helps in stabilizing variance

    making the data more normally distributed.

    One-hot encode categorical features in the DataFrame.

    This converts categorical variables into a format

    That can be provided to ML algorithms to do a better

    job in prediction.

    Remove the index from the DataFrame and the target variable.

    """

    eda_df_ = x.copy()

    y1 = y.copy().squeeze()

    positive = [
        "Reasonable Price",
        "Products always in Stock",
        "User Friendly Website",
        "Quality Customer Care",
    ]
    negative = [
        "Too many ads",
        "No reason specified",
        "Poor Product Quality",
        "Poor Customer Service",
        "Poor Website",
    ]
    neutral = ["No reason specified"]

    mapping = {cat: "positive" for cat in positive}
    mapping.update({cat: "negative" for cat in negative})
    mapping.update({cat: "neutral" for cat in neutral})

    eda_df_.feedback = eda_df_.feedback.replace(mapping)

    eda_df_["has_referral"] = (
        eda_df_["referral_id"].str.startswith("CID", na=False).astype(int)
    )
    # We explicitly pass Yes and No as explicit values for joined through referral
    values = ["Yes", "No"]
    eda_df_["joined_through_referral"] = eda_df_.joined_through_referral.where(
        eda_df_.joined_through_referral.isin(values), other=np.nan
    )
    eda_df_.joined_through_referral = eda_df_.joined_through_referral.astype("category")
    # Convert to datetime — pandas infers HH:MM:SS automatically
    eda_df_["last_visit_time"] = pd.to_datetime(eda_df_["last_visit_time"], format="%H:%M:%S")
    # Extract just the time component as a clean display
    eda_df_["last_visit_time"] = eda_df_["last_visit_time"].dt.strftime("%H:%M:%S")
    # Replace 'Unknown' with ' Male ' denoted as M in the dataset
    eda_df_["gender"] = eda_df_["gender"].str.strip().replace("Unknown", "M")
    # Remove negative values in the avg_time_spent column and replace them with NaN
    eda_df_["avg_time_spent"] = eda_df_["avg_time_spent"].where(
        eda_df_["avg_time_spent"] >= 0, other=np.nan
    )
    # We can see that there are negative values in the days_since_last_login column
    # which is not possible and we can coerce those values to NaN
    # Then replace them with the median of the column
    eda_df_["days_since_last_login"] = eda_df_["days_since_last_login"].where(
        eda_df_["days_since_last_login"] >= 0, other=np.nan
    )
    # We coerce the ? in medium of operation to NaN , then replace the NaN with the Smartphone
    eda_df_["medium_of_operation"] = eda_df_["medium_of_operation"].replace("?", np.nan)
    eda_df_["medium_of_operation"] = eda_df_["medium_of_operation"].fillna("Smartphone")
    # Let's coerce the error value in the average frequency login days column
    # To numeric and replace it with NaN
    # Then compute median on the now-clean numeric column
    eda_df_["avg_frequency_login_days"] = pd.to_numeric(
        eda_df_["avg_frequency_login_days"], errors="coerce"
    )
    eda_df_["avg_frequency_login_days"] = eda_df_["avg_frequency_login_days"].where(
        eda_df_.avg_frequency_login_days >= 0, other=np.nan
    )
    # Replace the joining_dates with joining_year and joining_month
    # Joining date is a string and we can use pandas to_datetime function
    # To convert it to datetime and then extract the year and month from it.
    eda_df_["joining_date"] = pd.to_datetime(eda_df_["joining_date"], dayfirst=True)
    eda_df_["joining_month"] = eda_df_["joining_date"].dt.strftime("%m")
    eda_df_["joining_year"] = eda_df_["joining_date"].dt.strftime("%Y")
    eda_df_["joining_month"] = eda_df_["joining_month"].astype(int)
    eda_df_["joining_year"] = eda_df_["joining_year"].astype(int)
    # We can drop the joining_date column now and apend joining_month and joining_year
    # to the end of the dataframe
    eda_df_new = eda_df_.drop(["joining_date", "referral_id"], axis=1)
    # Let us apply quantile based binning to the avg_frequency_login_days column
    # Create a new column called avg_frequency_login_days_binned
    # Also to the avg_time_spent_company column
    # Create a new column called avg_time_spent_company_binned
    eda_df_new["avg_frequency_login_days_binned"] = pd.qcut(
        eda_df_new["avg_frequency_login_days"], q=5, labels=False
    )
    eda_df_new["avg_time_spent_binned"] = pd.qcut(
        eda_df_new["avg_time_spent"], q=5, labels=False
    )
    # Let's apply fixed width bining to the age column and create a new column called age_binned
    eda_df_new["age_binned"] = pd.cut(eda_df_new["age"], bins=6, labels=False)

    # Fill in missing values for numerical and categorical features

    for column in eda_df_new.select_dtypes(include="str").columns:
        eda_df_new[column] = eda_df_new[column].astype("category")
    for column in eda_df_new.columns:
        # Categorical feature
        if isinstance(eda_df_new[column].dtype, pd.CategoricalDtype):
            eda_df_new[column] = eda_df_new[column].fillna(
                eda_df_new[column].mode()[0], inplace=True
            )
        else:
            # Numerical feature
            eda_df_new[column] = eda_df_new[column].fillna(
                eda_df_new[column].median(), inplace=True
            )

    # A function to indentify highly correlated numerical features
    # then subsequent functions will apply
    # numerical transformations to our data
    # Combine X and y temporarily

    df = pd.concat([eda_df_new, y], axis=1)

    # Correlation matrix
    corr = df.corr(numeric_only=True)

    # Target column name
    target_col = y.columns[0]

    # Features correlated with target above threshold
    most_correlated = [
        col
        for col in corr.columns
        if abs(corr.loc[col, target_col]) >= 0.2 and col != target_col
    ]

    features_categorical = eda_df_new.select_dtypes(include=["str", "object", "category"])

    features_numerical = eda_df_new.select_dtypes(include=[np.number])[most_correlated]

    mi_scores = {
        feature: mutual_info_score(eda_df_new[feature], y1)
        for feature in features_categorical.columns
    }
    mi_scores = {feature: score for feature, score in mi_scores.items() if score >= 0.2}
    selected_features = pd.Series(mi_scores).sort_values(ascending=False)

    # a dataframe containing numerical and most important categorical features

    selected_df = pd.concat([features_numerical, eda_df_new[selected_features.index]], axis=1)

    list_of_columns = ["avg_transaction_value", "points_in_wallet"]
    for column in list_of_columns:
        # handles both negative and positive values safely
        # log1p is used to handle zero values
        selected_df[column] = np.sign(selected_df[column]) * np.log1p(
            np.abs(selected_df[column])
        )
    df = selected_df.drop(["last_visit_time"], axis=1)

    # One hot encoding the categorical features
    # in the top_df

    categorical_cols = df.select_dtypes(include=["str", "object", "category"]).columns

    # drop='first' to avoid dummy variable trap
    encoder = OneHotEncoder(drop="first", sparse_output=False)

    encoded_data = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )

    df = pd.concat(
        [df.drop(columns=categorical_cols), encoded_df], axis=1, ignore_index=False
    )

    target = y.iloc[:, 0]

    return df, target
