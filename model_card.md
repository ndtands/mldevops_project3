# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model Type: Logistic Regression
- Version: 1.0.0
- Date: 23 October 2023

## Intended Use
The model can be used for predicting income classes on census data. There are two income classes >50K and <=50K (binary classification task).

## Training Data
- Source: [https://archive.ics.uci.edu/ml/datasets/census+income](https://archive.ics.uci.edu/ml/datasets/census+income)
- Training: 80% (26,561 instances)

## Evaluation Data
For evaluation 20% of the 32561 rows were used (6513 instances) in the test set.

## Metrics
Three metrics were used for model evaluation (performance on test set):
- Precision: 0.71
- Recall: 0.267
- F1 Score: 0.388

## Ethical Considerations
Used public, aggregated census data ensuring privacy.

## Caveats and Recommendations
Consider hyperparameter optimization for better performance.
