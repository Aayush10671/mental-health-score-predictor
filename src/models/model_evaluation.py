import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('model_evaluation')


def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info('Data loaded successfully from %s, shape=%s', data_path, df.shape)
        return df
    except Exception as e:
        logger.error('Unexpected error loading data from %s: %s', data_path, str(e))
        raise


def load_joblib(input_path: str):
    try:
        obj = joblib.load(input_path)
        logger.info('Object loaded successfully from %s', input_path)
        return obj
    except Exception as e:
        logger.error('Error loading object from %s: %s', input_path, str(e))
        raise


def align_features(X: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    X = pd.get_dummies(X, drop_first=True)
    # Ensures test-time dummy columns exactly match what the model was trained on,
    # even if a category present in train is missing (or an unseen one appears) in test
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    logger.info('Evaluation metrics: %s', metrics)
    return metrics


def save_json(data: dict, output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info('Metrics saved successfully to %s', output_path)
    except Exception as e:
        logger.error('Error saving metrics to %s: %s', output_path, str(e))
        raise


def main():
    X_test_path = 'data/processed/X_test.csv'
    y_test_path = 'data/processed/y_test.csv'
    model_path = 'models/model.pkl'
    features_path = 'models/model_features.pkl'
    output_path = 'models/metrics.json'

    try:
        model = load_joblib(model_path)
        feature_columns = load_joblib(features_path)

        X = load_data(X_test_path)
        y = load_data(y_test_path)['Mental_Health_Score']

        X = align_features(X, feature_columns)

        metrics = evaluate_model(model, X, y)
        save_json(metrics, output_path)

        logger.info('Model evaluation completed successfully')
    except Exception as e:
        logger.error('Model evaluation failed: %s', str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()