import pickle
import pandas as pd
import pandas.api.types as pdtypes
import pytest
from sklearn.model_selection import train_test_split

from core.data import process_data
from core.model import infer, caculator_metrics

expected_column_values = {
    "workclass": {
        'Self-emp-inc', 'State-gov', 'Federal-gov', 'Self-emp-not-inc', 'Local-gov', 
        'Private', 'Without-pay', '?', 'Never-worked'
    },
    "education": {
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
        "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
        "Doctorate", "5th-6th", "Preschool"
    },
    "marital-status": {
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
        "Married-spouse-absent", "Married-AF-spouse"
    },
    "occupation": {
        'Protective-serv', 'Other-service', 'Tech-support', 'Farming-fishing', 'Priv-house-serv',
        'Armed-Forces', 'Handlers-cleaners', '?', 'Machine-op-inspct', 'Sales', 'Prof-specialty', 
        'Adm-clerical', 'Transport-moving', 'Craft-repair', 'Exec-managerial'
    },
    "relationship": {
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    },
    "sex": {
        "Male", "Female"
    },
    "salary": {
        "<=50K", ">50K"
    }
}

ranges = {
        "age": (17, 90),
        "education-num": (1, 16),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
    }


fake_categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="module")
def data():
    return pd.read_csv("artifact/dataset/census_clean.csv", skipinitialspace=True)

def test_column_presence_and_type(data):
    """Tests columns expect and type expect.

    Args:
        data (pd.DataFrame): Dataset for test
    """

    required_columns = {
        "age": pdtypes.is_int64_dtype,
        "workclass": pdtypes.is_object_dtype,
        "fnlgt": pdtypes.is_int64_dtype,
        "education": pdtypes.is_object_dtype,
        "education-num": pdtypes.is_int64_dtype,
        "marital-status": pdtypes.is_object_dtype,
        "occupation": pdtypes.is_object_dtype,
        "relationship": pdtypes.is_object_dtype,
        "race": pdtypes.is_object_dtype,
        "sex": pdtypes.is_object_dtype,
        "capital-gain": pdtypes.is_int64_dtype,
        "capital-loss": pdtypes.is_int64_dtype,
        "hours-per-week": pdtypes.is_int64_dtype,
        "native-country": pdtypes.is_object_dtype,
        "salary": pdtypes.is_object_dtype,
    }

    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"


def test_column_values(data: pd.DataFrame) -> None:
    """Tests that specific columns in the dataframe have the expected unique values.

    Args:
        data (pd.DataFrame): Dataset for testing.
        column_values_dict (dict): Dictionary where keys are column names and 
                                  values are sets of expected unique values for the column.

    Raises:
        AssertionError: If a column's unique values do not match the expected values.
    """
    for column_name, expected_values in expected_column_values.items():
        assert set(data[column_name].unique()) == expected_values, f"Unexpected values in {column_name}."



def test_column_ranges(data):
    """Tests that specific columns in the dataframe fall within expected ranges.

    Args:
        data (pd.DataFrame): Dataset for testing.

    Raises:
        AssertionError: If any value in a specified column is outside the expected range.

    """

    for col_name, (minimum, maximum) in ranges.items():
        assert data[col_name].min() >= minimum, f"Values in {col_name} below the expected minimum {minimum}."
        assert data[col_name].max() <= maximum, f"Values in {col_name} exceed the expected maximum {maximum}."


def test_inference(data):
    """
    Tests the inference of the logistic regression model on a subset of the data.
    Args:
        data (pd.DataFrame): Dataset for testing.

    Raises:
        AssertionError: If the number of predictions doesn't match the number of data points 
                        in the test set.
    """
    _, test_df = train_test_split(data, test_size=0.20)
    [encoder, lb, lr_model] = pickle.load(open("artifact/models/lr_model.pkl", "rb"))

    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=fake_categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = infer(lr_model, X_test)

    assert len(preds) == len(X_test)



