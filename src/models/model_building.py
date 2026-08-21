import pandas as pd
import os
import sys
import joblib
import logging
import yaml
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('model_building')

DEFAULT_PARAMS = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        model_params = params['model_building']

        if model_params.get('max_depth') == 'None':
            model_params['max_depth'] = None

        logger.info('Model params retrieved successfully: %s', model_params)
        return model_params
    except Exception as e:
        logger.error('Unexpected error loading params: %s', str(e))
        return DEFAULT_PARAMS


def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info('Data loaded successfully from %s, shape=%s', data_path, df.shape)
        return df
    except Exception as e:
        logger.error('Unexpected error loading data from %s: %s', data_path, str(e))
        raise


def save_model(obj, output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(obj, output_path)
        logger.info('Model saved successfully to %s', output_path)
    except Exception as e:
        logger.error('Error saving model to %s: %s', output_path, str(e))
        raise


def main():
    params_path = 'params.yaml'
    X_train_path = 'data/processed/X_train.csv'
    y_train_path = 'data/processed/y_train.csv'

    try:
        model_params = load_params(params_path)

        X = load_data(X_train_path)
        y = load_data(y_train_path)['Mental_Health_Score']

        X = pd.get_dummies(X, drop_first=True)

        model = RandomForestRegressor(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            min_samples_split=model_params['min_samples_split'],
            min_samples_leaf=model_params['min_samples_leaf'],
            random_state=42
        )
        model.fit(X, y)
        logger.info('Model trained on %d samples, %d features', X.shape[0], X.shape[1])

        save_model(model, 'models/model.pkl')
        save_model(list(X.columns), 'models/model_features.pkl')

        logger.info('Model building completed successfully')
    except Exception as e:
        logger.error('Model building failed: %s', str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()