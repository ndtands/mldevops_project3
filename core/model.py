"""
Filename: model.py
Description: train model and evaluate model and inference
Author: tannd22
Date: 21-10-2023
"""
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import hydra
from omegaconf import DictConfig
from .data import process_data

def train_model(X_train, y_train):
    """
    Train ML model
    Args:
        - X_train: np.array
        - y_train: np.array

    Returns:
        - Model trained
    """
    lr_model = LogisticRegression(max_iter=1000, random_state=8071)
    lr_model.fit(X_train, y_train.ravel())
    return lr_model


def caculator_metrics(y_true, y_pred):
    """
    Caculator metrics for evaluation
    Args:
        - y_true: np.array
        - y_pred: np.array

    Returns:
        f1: float
        recall: float
        precision: float
    """
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=True)
    precision = precision_score(y_true=y_true, y_pred=y_pred, zero_division=True)
    recall = recall_score(y_true=y_true, y_pred=y_pred, zero_division=True)
    return f1, recall, precision


def infer(model, x):
    """
    Run model
    Args:
        - model: model trained
        - x: np.array

    Returns:
        - preds: np.array
    """
    return model.predict(x)


def caculator_metrics_for_category(df, feature, category, cat_columns, label, encoder, lb, model):
    """Caculator metrics for a specific feature-category combination."""
    subset_df = df[df[feature] == category]
    x, y, _, _ = process_data(
        X=subset_df,
        categorical_features=cat_columns,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    preds = infer(model, x)
    f1, recall, precision = caculator_metrics(y, preds)

    return {
        'feature': feature,
        'category': category,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def caculator_metrics_with_slices_data(df, cat_columns, label, encoder, lb, model, slice_output_path):
    """
    Caculator metrics of the model on slices of the data.

    Args:
        df (pd.DataFrame): Input dataframe
        cat_columns (list): list of categorical columns
        label (str): Class label string
        encoder (OneHotEncoder): fitted One Hot Encoder
        lb (LabelBinarizer): label binarizer
        model (module.model): Trained model binary file
        slice_output_path (str): path to save the slice output

    Returns:
        metrics (pd.DataFrame): Dataframe containing the metrics
    """

    metrics_data = [
        caculator_metrics_for_category(df, feature, category, cat_columns, label, encoder, lb, model)
        for feature in cat_columns
        for category in df[feature].unique()
    ]

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(slice_output_path, index=False)

    return metrics_df