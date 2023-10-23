"""
Filename: data.py
Description: Preprocess data for train
Author: tannd22
Date: 21-10-2023
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

def process_data(X, categorical_features=[], label=None, training=True, encoder=None, lb=None):
    """
    Process the data used in the machine learning pipeline.

    Parameters:
    ----------
    X : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[]).
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned for y (default=None).
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns:
    -------
    X_processed : np.array
        Processed data.
    y : np.array
        Processed labels if label is provided, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer passed in.
    """

    # Extract and process labels
    if label:
        y = X[label].values
        X = X.drop(columns=[label])
        if training:
            lb = LabelBinarizer()
            y = lb.fit_transform(y).ravel()
        else:
            y = lb.transform(y).ravel()
    else:
        y = np.array([])

    # Separate categorical and continuous features
    X_cat = X[categorical_features]
    X_cont = X.drop(columns=categorical_features)

    # Process categorical features
    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat_encoded = encoder.fit_transform(X_cat)
    else:
        X_cat_encoded = encoder.transform(X_cat)

    # Combine processed features back
    X_processed = np.hstack([X_cont.values, X_cat_encoded])

    return X_processed, y, encoder, lb