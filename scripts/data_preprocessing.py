import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def handle_missing_values(df):
    '''Impute missing values with median for numerical columns'''
    num_imputer = SimpleImputer(strategy='median')
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    return df

def cap_outliers(df, column):
    '''Capping outliers using IQT'''
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, 
                         np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

def encode_categorical_variable(df):
    """Encode categorical variables (CHAS as binary, RAD as one-hot)."""
    df['CHAS'] = df['CHAS'].astype('category')
    df['RAD'] = df['RAD'].astype("category")
    df = pd.get_dummies(df, columns = ['RAD'], prefix='RAD', drop_first=True)
    return df

def standardize_features(df, target_col):
    """Standardize numerical features (excluding categorical features)."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Identify numerical features to scale (exclude categorical features)
    scale_cols = [col for col in X.select_dtypes(include=['float64', 'int64']).columns if col not in ['CHAS'] and not col.startswith('RAD_')]

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[scale_cols] = scaler.fit_transform(X[scale_cols])
    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2, random_state =42):
    """Split data into train/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, scaler, path):
    """Save processed data and scaler to disk."""
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }, path)

# --- Full Pipeline Execution ---
def run_preprocessing_pipeline(df, output_path):
    """Complete preprocessing workflow."""
    
    # 1. Handle missing values
    df = handle_missing_values(df)
    
    # 2. Cap outliers (all numerical columns)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df = cap_outliers(df, col)
    
    # 3. Encode categoricals
    df = encode_categorical_variable(df)
    
    # 4. Standardize features
    X, y, scaler = standardize_features(df, target_col='MEDV')
    
    # 5. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 6. Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, scaler, output_path)
    print(f"Preprocessing complete. Data saved to {output_path}")

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', 'data', 'boston_housing.csv')
    df = pd.read_csv(data_file)
    return df

def get_output_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, '..', 'data', 'processed_data.pkl')
    return output_file

if __name__ == "__main__":
    df = load_data()
    output_file = get_output_path()
    run_preprocessing_pipeline(
        df,
        output_file
    )
