import joblib
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import os

class FeatureEngineer:
    """Production-ready feature engineering pipeline"""
    
    def __init__(self):
        """
        Initialize with original feature names.
        If None, will use default Boston housing features.
        """
        self.original_features = ['RM', 'LSTAT', 'PTRATIO', 'INDUS', 
            'TAX', 'RAD_4.0', 'CHAS'
        ]
        
        # These will be set during fitting
        self.selected_features_ = None
        self.selector_ = None
    
    def create_features(self, X):
        """Create new features while preserving original columns"""
        X = X[self.original_features].copy()
        
        # 1. Interaction terms
        X['ROOMS_PER_TAX'] = X['RM'] / (X['TAX'] + 1e-6)
        X['LSTAT_RM_INTERACTION'] = X['LSTAT'] * X['RM']
        
        # 2. Non-linear transforms
        X['LSTAT_SQUARED'] = X['LSTAT'] ** 2
        
        # 3. Binning
        X['RM_BINNED'] = pd.cut(X['RM'], bins=5, labels=False)
        
        return X
    
    def select_features(self, X, X_test, y, model, threshold='median'):
        """
        Perform feature selection using XGBoost importance.
        
        Args:
            X: DataFrame with engineered features
            X_test: DataFrame with engineered test features
            y: Target values
            model: Pre-trained XGBoost model 
            threshold: Feature importance threshold
            
        Returns:
            Transformed DataFrame with selected features
        """
        # Create safe model for feature selection
        params = {k: v for k, v in model.get_params().items() 
                     if k not in ['early_stopping_rounds', 'eval_set']}
        selector_model = xgb.XGBRegressor(**params)

        # Fit selector
        self.selector_ = SelectFromModel(
            estimator = selector_model, 
            threshold = threshold
        ).fit(X.values, y)
        
        # Store selected features
        self.selected_features_ = X.columns[self.selector_.get_support()]

        # Selecting the top features from selected dataset and engineered dataset
        X_train_selected = self.transform(X)
        X_test_selected = self.transform(X_test)

        # printing information
        print(f"\nReduced from {X.shape[1]} to {X_train_selected.shape[1]} features.")
        print("Selected features:", list(self.selected_features_))
        return X_train_selected, X_test_selected
    
    def evaluate(self, best_params, X_train, X_test, y_train, y_test):    
        """
        Perform evaluation using XGBoost.
        
        Args:
            best_params: best parameters of the model
            X_train: DataFrame with Training Features
            X_test: DataFrame with Testing Features
            y_train: DataFrame with Training Target Variables
            y_test: DataFrame with Testing Target Variables           
        Returns:
            Target Variable predictions on y_test
        """   
        # First ensure all columns are numeric
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        
        # Update parameters
        params = {
            **best_params,
            'early_stopping_rounds': 10,
            'eval_metric': 'rmse',
        }

        # Initialize and train
        xgb_model = xgb.XGBRegressor(**params)
        
        try:
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=True
            )
        
            y_pred = xgb_model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)
            print(f"\nTest RMSE with {len(X_train.columns)} features: {rmse:.4f}")

            if hasattr(xgb_model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': xgb_model.feature_importances_
                }).sort_values('Importance', ascending=False)
            print(importance)
            importance.to_csv(f"{self.file_path('data', 'feature_importance.csv')}")
            return y_pred
        
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            print("Data types in X_train:", X_train.dtypes)
            print("Data types in X_test:", X_test.dtypes)
            raise
        

    def transform(self, X):
        """Apply feature engineering and selection"""
        if self.selector_ is not None:
            return pd.DataFrame(
                self.selector_.transform(X.values),
                columns=self.selected_features_
            )
        else:
            print("No selector found.")
        return 
    
    def save(self, path):
        """Save the entire feature engineering pipeline"""
        joblib.dump(self, path)
    
    def file_path(self, folder, file):
        """Load processed data with path handling."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', folder, file)
        return data_path
        
    @classmethod
    def load(cls, path):
        """Load a saved pipeline"""
        return joblib.load(path)

def run_feature_engineering(data_path, model_path=None, output_path=None):
    """
    Complete feature engineering workflow.
    
    Args:
        data_path: Path to processed_data.pkl
        model_path: Path to trained model (optional)
        output_dir: path with filename to save outputs
    """
    # Load data
    data = joblib.load(data_path)
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    
    # Initialize and fit pipeline
    fe = FeatureEngineer()
    
    # Create features
    X_train_engineered = fe.create_features(X_train)
    X_test_engineered = fe.create_features(X_test)
    
    # load model for feature selection
    model = joblib.load(model_path)
        
    if isinstance(model, dict):
        model_ = model['model']
        best_params = model['params']
    else:
        model_ = model
        best_params = model.get_params()



    # Feature Evaluation
    y_pred1 = fe.evaluate(best_params, X_train_engineered, X_test_engineered, y_train, y_test)

    print("Performing feature selection with pretrained model...")
    X_train_selected, X_test_selected = fe.select_features(X_train_engineered, X_test_engineered, y_train, model_)

    # Retraining the model with selected features 
    y_pred2 = fe.evaluate(best_params, X_train_selected, X_test_selected, y_train, y_test)

    # Save predictions
    predictions = pd.DataFrame({'Real_MEDV': y_test,'Predicted_MEDV_with_all_FE_features': y_pred1,'Predicted_MEDV_With_Selected_FE_features': y_pred2}).sort_index()
    predictions.to_csv(f'{fe.file_path('data', 'predictions.csv')}')

    # Save pipeline
    fe.save(output_path)

    print(f"Selected Number of Features after Feature Engineering: {len(X_train_selected.columns)}")
    print("Selected features:", list(X_train_selected.columns)) 

    # Print ALL rows 
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)    # Prevent column wrapping

    print("\nReal MEDV vs Predicted MEDV\n", predictions)
    print(f"\n\nFeature engineering complete. Pipeline saved to {output_path}")
    return fe

def file_path(folder, file):
    """provide filepath for provided foler and filename."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', folder, file)
    return data_path



if __name__ == "__main__":
    # Example usage
    run_feature_engineering(
        data_path = file_path("data", "processed_data.pkl"),
        model_path = file_path("models", "best_model.pkl"),
        output_path=file_path("models", "feature_engineering.pkl")
    )