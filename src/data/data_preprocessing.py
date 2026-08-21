import pandas as pd
import numpy as np
import os
import sys
import logging
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('data_preprocessing')


def load_params(params_path: str) -> dict:
    """Load parameters from yaml file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info('Parameters loaded successfully from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading params from %s: %s', params_path, str(e))
        return {}


def load_data(data_path: str) -> pd.DataFrame:
    """Load raw data from CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.info('Data loaded successfully from %s, shape=%s', data_path, df.shape)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', data_path, str(e))
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning."""
    df = df.copy()
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    logger.info('Duplicates removed, shape=%s', df.shape)
    
    # Clip negative physical activity hours
    if 'Physical_Activity_Hours' in df.columns:
        df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)
        logger.info('Physical_Activity_Hours clipped to non-negative values')
    
    return df


def group_countries(df: pd.DataFrame, top_n: int = 11) -> pd.DataFrame:
    """Group less frequent countries into 'Other' category."""
    df = df.copy()
    
    # Get top N countries
    top_countries = df['Country'].value_counts().head(top_n).index.tolist()
    
    # Apply grouping
    df['Grouped_Country'] = df['Country'].apply(
        lambda x: x if x in top_countries else 'Other'
    )
    
    logger.info('Countries grouped: %d categories after grouping', df['Grouped_Country'].nunique())
    return df


def create_preprocessor() -> ColumnTransformer:
    """
    Create a ColumnTransformer with appropriate preprocessing pipelines.
    This follows the preprocessing logic from your notebook.
    """
    
    # Define column groups based on your notebook
    skewed_cols = ['Study_Hours']  # Skewed numeric columns
    other_numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                          'Physical_Activity_Hours', 'Daily_Unlocks']
    ordinal_cols = ['Stress_Level']
    nominal_cols = ['Gender', 'Academic_Level', 'Most_Used_Platform', 
                    'Grouped_Country', 'Purpose_Of_Use']
    
    # Define pipelines for each column type
    skew_pipeline = Pipeline(steps=[
        ('log_transformer', FunctionTransformer(np.log1p, validate=True)),
        ('scaler', StandardScaler())
    ])
    
    plain_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    ordinal_pipeline = Pipeline(steps=[
        ('ordinal_encoder', OrdinalEncoder(
            categories=[['Low', 'Medium', 'High', 'Very High']]
        ))
    ])
    
    nominal_pipeline = Pipeline(steps=[
        ('onehot_encoder', OneHotEncoder(
            drop='first', 
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('skew', skew_pipeline, skewed_cols),
            ('plain', plain_pipeline, other_numeric_cols),
            ('ordinal', ordinal_pipeline, ordinal_cols),
            ('nominal', nominal_pipeline, nominal_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    logger.info('Preprocessor created with transforms: %s', 
                [t[0] for t in preprocessor.transformers])
    
    return preprocessor


def remove_outliers_iqr(df: pd.DataFrame, columns: list, multiplier: float = 1.5) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    df = df.copy()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Filter out outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    logger.info('Outliers removed using IQR method, shape=%s', df.shape)
    return df


def get_feature_names(preprocessor: ColumnTransformer, df: pd.DataFrame) -> list:
    """Get feature names after transformation."""
    try:
        # Get feature names from each transformer
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'skew' or name == 'plain':
                feature_names.extend(columns)
            elif name == 'ordinal':
                feature_names.extend(columns)
            elif name == 'nominal':
                # Get the one-hot encoder and its feature names
                encoder = transformer.named_steps['onehot_encoder']
                if hasattr(encoder, 'get_feature_names_out'):
                    feature_names.extend(encoder.get_feature_names_out(columns))
                else:
                    # Fallback for older versions
                    feature_names.extend(encoder.get_feature_names(columns))
        return feature_names
    except Exception as e:
        logger.warning('Could not get feature names: %s', str(e))
        return []


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str) -> None:
    """Save preprocessor using joblib."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(preprocessor, filepath)
        logger.info('Preprocessor saved to %s', filepath)
    except Exception as e:
        logger.error('Error saving preprocessor to %s: %s', filepath, str(e))
        raise


def save_objects(preprocessor: ColumnTransformer, feature_names: list, output_dir: str) -> None:
    """Save all preprocessing objects."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessor
        preprocessor_path = os.path.join(output_dir, 'preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_path)
        logger.info('Preprocessor saved to %s', preprocessor_path)
        
        # Save feature names
        features_path = os.path.join(output_dir, 'feature_names.joblib')
        joblib.dump(feature_names, features_path)
        logger.info('Feature names saved to %s', features_path)
        
        # Save column groups information
        col_groups = {
            'skewed_cols': ['Study_Hours'],
            'other_numeric_cols': ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                                   'Physical_Activity_Hours', 'Daily_Unlocks'],
            'ordinal_cols': ['Stress_Level'],
            'nominal_cols': ['Gender', 'Academic_Level', 'Most_Used_Platform', 
                            'Grouped_Country', 'Purpose_Of_Use']
        }
        col_groups_path = os.path.join(output_dir, 'column_groups.joblib')
        joblib.dump(col_groups, col_groups_path)
        logger.info('Column groups saved to %s', col_groups_path)
        
    except Exception as e:
        logger.error('Error saving objects to %s: %s', output_dir, str(e))
        raise


def main():
    """Main preprocessing pipeline."""
    try:
        # Load parameters
        params = load_params('params.yaml')
        test_size = params.get('data_preprocessing', {}).get('test_size', 0.2)
        random_state = params.get('data_preprocessing', {}).get('random_state', 42)
        outlier_multiplier = params.get('data_preprocessing', {}).get('outlier_multiplier', 1.5)
        
        # Load raw data
        df = load_data('data/raw/dataset.csv')
        
        # Clean data
        df = clean_data(df)
        
        # Group countries
        df = group_countries(df, top_n=11)
        logger.info('Country grouping completed. Value counts:\n%s', 
                    df['Grouped_Country'].value_counts())
        
        # Split data (BEFORE preprocessing to prevent data leakage)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        logger.info('Data split: train shape=%s, test shape=%s', train_df.shape, test_df.shape)
        
        # Remove outliers from training data only
        numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'Mental_Health_Score']  # Exclude target
        
        # Check skewness for decision (from your notebook)
        skew_threshold = 0.5
        for col in numeric_cols:
            skewness = train_df[col].skew()
            logger.info('Skewness for %s: %.4f', col, skewness)
        
        # Remove outliers using IQR
        train_df = remove_outliers_iqr(train_df, numeric_cols, outlier_multiplier)
        logger.info('After outlier removal: train shape=%s', train_df.shape)
        
        # Prepare features and target
        X_train = train_df.drop(columns=['Mental_Health_Score'])
        y_train = train_df['Mental_Health_Score']
        
        X_test = test_df.drop(columns=['Mental_Health_Score'])
        y_test = test_df['Mental_Health_Score']
        
        # Create and fit preprocessor
        preprocessor = create_preprocessor()
        preprocessor.fit(X_train)
        logger.info('Preprocessor fitted on training data')
        
        # Transform data
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = get_feature_names(preprocessor, X_train)
        logger.info('Total features after transformation: %d', len(feature_names))
        
        # Save preprocessor and transformed data
        save_preprocessor(preprocessor, 'models/preprocessor.joblib')
        save_objects(preprocessor, feature_names, 'models/')
        
        # Ensure processed data directory exists before saving
        os.makedirs('data/processed', exist_ok=True)

        # Save transformed data
        pd.DataFrame(X_train_transformed, columns=feature_names).to_csv('data/processed/X_train.csv', index=False)
        pd.DataFrame(X_test_transformed, columns=feature_names).to_csv('data/processed/X_test.csv', index=False)
        pd.Series(y_train, name='Mental_Health_Score').to_csv('data/processed/y_train.csv', index=False)
        pd.Series(y_test, name='Mental_Health_Score').to_csv('data/processed/y_test.csv', index=False)
        
        logger.info('Preprocessing completed successfully!')
        logger.info('Files saved:')
        logger.info('  - models/preprocessor.joblib')
        logger.info('  - models/feature_names.joblib')
        logger.info('  - models/column_groups.joblib')
        logger.info('  - data/processed/X_train.csv')
        logger.info('  - data/processed/X_test.csv')
        logger.info('  - data/processed/y_train.csv')
        logger.info('  - data/processed/y_test.csv')
        
    except Exception as e:
        logger.error('Preprocessing failed: %s', str(e), exc_info=True)
        sys.exit(1)


def load_preprocessor(filepath: str = 'models/preprocessor.joblib'):
    """Helper function to load the saved preprocessor."""
    try:
        preprocessor = joblib.load(filepath)
        logger.info('Preprocessor loaded from %s', filepath)
        return preprocessor
    except Exception as e:
        logger.error('Error loading preprocessor from %s: %s', filepath, str(e))
        return None


if __name__ == '__main__':
    main()