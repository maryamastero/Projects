import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# befor running the code we need to run 
#mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5002
#in a terminal

mlflow.set_tracking_uri("http://localhost:5002")

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

def train_and_log_model(model, model_name, params):
    with mlflow.start_run():
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        mlflow.sklearn.log_model(model, model_name)

        prediction_df = pd.DataFrame(y_pred, columns=["predictions"])
        prediction_df.to_csv(f"{model_name}_predictions.csv", index=False)
        mlflow.log_artifact(f'{model_name}_predictions.csv')
        print(f'model {model_name} logged successfully')

rf_params = {
    'n_estimators': 100,
    'max_depth': 3
}

rf = RandomForestRegressor(**rf_params)
train_and_log_model(rf, 'rf', rf_params)

lr_params = {
    'fit_intercept': True
}

lr = LinearRegression(**lr_params)
train_and_log_model(lr, 'lr', lr_params)

print("Run completed. You can view the results in the MLflow UI.")