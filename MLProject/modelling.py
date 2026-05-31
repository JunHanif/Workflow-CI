import pandas as pd
import numpy as np
import os
import sys
import mlflow
from sklearn.ensemble import RandomForestClassifier

def train_model(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    mlflow.sklearn.autolog()

    # menggunakan best parameter yang didapat dari eksperimen sebelumnya
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=4,
        random_state=42
        )
    model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
        

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'hantavirus_dataset_preprocessing'
    train_model(data_dir)
