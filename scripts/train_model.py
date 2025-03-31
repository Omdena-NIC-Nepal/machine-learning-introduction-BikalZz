import joblib
import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error

def train_model(X_train, y_train, X_test=None, y_test=None):
    """
    Trains a Gradiant Boost model as hyperparameter tuning.
    Returns trained model and metrics (if test data provided).
    """
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6],
        'gamma': [0, 0.1]  # Added to control overfitting
    }

    # Initialize the RandomForestRegressor model
    xgb_model = XGBRegressor(
        random_state=42,
        enable_categorical=True,
        early_stopping_rounds=10,  # Add early stopping
        eval_metric='rmse'
    )

    # Set up GridSearchCV with cross-validation
    xgb_grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',  # Directly optimize for RMSE
        n_jobs=-1  # Parallelize computation   
    )

    xgb_grid_search.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  # Validation data for early stopping
        verbose=False
    )
    
    # Retrieve the best model and its parameters
    best_xgb_model = xgb_grid_search.best_estimator_
    best_params = xgb_grid_search.best_params_

    metrics = {}
    if X_test is not None and y_test is not None:
        # Predict on the test set using the best model
        y_pred = best_xgb_model.predict(X_test)
        metrics = {
            'rmse': root_mean_squared_error(y_test, y_pred),
            'best_params': best_params
        }
    
    return best_xgb_model, metrics

def load_data():
    """Load processed data with path handling."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', 'data', 'processed_data.pkl')
    data = joblib.load(data_file)
    return data

def save_model(model, metrics):
    """Save model and metrics with path handling."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'best_model.pkl')
    metrics_path = os.path.join(current_dir, '..', 'models', 'metrics.pkl')
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(metrics, metrics_path)
    return model_path


if __name__ == "__main__":
    # Load data
    data = load_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Feature selection (same as notebook)
    selected_features = ['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'TAX', 'RAD_4.0', 'CHAS']
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    # Train and evaluate
    model, metrics = train_model(X_train, y_train, X_test, y_test)
    
    saved_path = save_model(model, metrics)

    print(f"Model saved to {saved_path}")
    print(f"-Best Params: {metrics['best_params']}")
    print(f"-RMSE: {metrics['rmse']:.2f}")