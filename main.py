import pandas as pd
import joblib
import yaml
import os

def load_model_and_predict():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_file = config["path"]
    selected_features = config["selected_features"]
    
    model = joblib.load(model_file)
    scaler = joblib.load("scaler.joblib")
    
    df = pd.read_csv("parkinsons.csv")
    
    X = df[selected_features]
    X_scaled = scaler.transform(X)
    y_true = df["status"]
    predictions = model.predict(X_scaled)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, predictions)
    
    return predictions
    
if __name__ == "__main__":
    load_model_and_predict()
