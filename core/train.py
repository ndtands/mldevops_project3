"""
Filename: train.py
Description: pipeline train model
Author: tannd22
Date: 21-10-2023
"""

import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from .model import (
        infer,
        train_model,
        caculator_metrics,
        caculator_metrics_with_slices_data,
        train_model
    )

from .data import process_data

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

def training(config: DictConfig):
    """
    train model and save model

    """
    logger.info(f"Hydra config: {config}")
    LABEL = config['main']['label']
    CATEGORY_FEATURES = config['main']['cat_features']
    MODEL_PATH = config['main']['model_path']
    DATA_PATH = config['main']['data_path']
    TEST_SIZE = config['main']['test_size']
    FOLDER_MODEL = config['main']['folder_path_model']
    SLICE_OUTPUT_PATH = config['main']['slice_output_path']

    logger.info("Reading data...")
    df = pd.read_csv(DATA_PATH)
    logger.info(df.describe())

    logger.info("Splitting data...")
    train, test = train_test_split(df, test_size=TEST_SIZE)

    logger.info("Processing data...")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CATEGORY_FEATURES, label=LABEL, training=True)
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=CATEGORY_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb)

    logger.info("Training model...")
    model = train_model(X_train, y_train)
    logger.info(model)

    logger.info("Saving model...")
    if not os.path.exists(FOLDER_MODEL):
        os.mkdir(FOLDER_MODEL)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump([encoder, lb, model], f)
    logger.info("Model saved.")

    logger.info("Inference model...")
    preds = infer(model, X_test)

    logger.info("Calculating model metrics...")
    f1, recall, precision = caculator_metrics(y_test, preds)
    logger.info(f">>>Precision: {precision}")
    logger.info(f">>>Recall: {recall}")
    logger.info(f">>>F1: {f1}")

    logger.info("Calculating model metrics on slices data...")
    metrics = caculator_metrics_with_slices_data(
        df=test,
        cat_columns=CATEGORY_FEATURES,
        label=LABEL,
        encoder=encoder,
        lb=lb,
        model=model,
        slice_output_path=SLICE_OUTPUT_PATH
    )
    logger.info(f">>>Metrics with slices data: {metrics}")