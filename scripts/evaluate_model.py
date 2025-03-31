import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
import os

def evaluate_model():
    """
    Evaluate model and save metrics/residuals.
    """
    # Load data and model
    data = joblib.load(file_path('data', 'processed_data.pkl'))
    model = joblib.load(file_path('models', 'best_model.pkl'))
    
    # Training features
    training_features = ['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'TAX', 'RAD_4.0', 'CHAS']

    # assinging X_test and y_test from processed data for model evaluation
    X_test_all = data['X_test']

    # Match training features
    X_test = X_test_all[training_features]
    y_test = data['y_test']
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }
    
    # Saving metrics to the model
    joblib.dump(metrics, file_path('models', 'metrics.pkl'))
    
    # predictions for monitoring
    predictions = pd.DataFrame({
                        'Real': y_test,
                        'predicted': y_pred
                    }).sort_index()
    
    # Saving the predicitons to the csv file
    predictions.to_csv(f"{file_path('data', 'predictions.csv')}", index=False)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

    # Saving the feature importance to the csv file
    importance.to_csv(f"{file_path('data', 'feature_importance.csv')}", index=False)

    return metrics, predictions, importance

def file_path(folder, file):
    """Load processed data with path handling."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', folder, file)
    return data_path



if __name__ == "__main__":
    metrics, predictions, feature_importance = evaluate_model()
    print("Evaluation complete. Metrics:", metrics)
    print("\nFeature importance: \n", feature_importance)
    print("\nReal and Predicted values:\n", predictions)