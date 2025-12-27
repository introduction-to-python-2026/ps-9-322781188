import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/parkinsons.csv')
X = df[['spread1', 'PPE']]
y = df['status']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.8, random_state=42
)
model = SVC(kernel='rbf', C=1.0, probability=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)



