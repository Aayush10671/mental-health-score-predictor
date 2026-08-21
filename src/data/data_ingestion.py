import pandas as pd
import os
import sys
import logging
import yaml

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('data_ingestion')


def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info('Data loaded successfully from %s, shape=%s', data_path, df.shape)
        return df
    except Exception as e:
        logger.error('Unexpected error loading data from %s: %s', data_path, str(e))
        raise


def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        logger.error('Loaded dataframe is empty')
        raise ValueError('Loaded dataframe is empty')

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        logger.warning('Found %d null values across columns:\n%s', total_nulls, null_counts[null_counts > 0])
    else:
        logger.info('No null values found')

    dup_count = int(df.duplicated().sum())
    logger.info('Found %d duplicate rows (left untouched at this stage)', dup_count)


def save_data(df: pd.DataFrame, output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info('Data saved successfully to %s', output_path)
    except Exception as e:
        logger.error('Error saving data to %s: %s', output_path, str(e))
        raise


def main():
    source_path = 'data/external/dataset.csv'
    output_path = 'data/raw/dataset.csv'

    try:
        df = load_data(source_path)
        validate_data(df)
        save_data(df, output_path)

        logger.info('Data ingestion completed successfully')
    except Exception as e:
        logger.error('Data ingestion failed: %s', str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()