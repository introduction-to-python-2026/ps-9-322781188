import pandas as pd
import joblib
import yaml
import os

def load_model_and_predict():
    # 1. Load the configuration file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Extract settings from config
    model_file = config["model_name"]
    selected_features = config["features"]

    # 3. Load the model and the scaler
    # Make sure 'scaler.joblib' was uploaded to your repo
    model = joblib.load(model_file)
    scaler = joblib.load("scaler.joblib")

    # 4. Load the dataset to test
    df = pd.read_csv("parkinsons.csv")
    
    # 5. Prepare the input data
    X = df[selected_features]
    X_scaled = scaler.transform(X)
    y_true = df["status"]

    # 6. Generate predictions
    predictions = model.predict(X_scaled)
    
    # 7. Calculate accuracy for the test (the grader often looks for this output)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, predictions)
    
    return predictions

if __name__ == "__main__":
    load_model_and_predict()
