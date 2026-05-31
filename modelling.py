import pandas as pd
import numpy as np
import os
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def train_model(dir):
    repo_owner = "junhanif" 
    repo_name = "SMSML_Jun"
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    mlflow.set_experiment('Hantavirus detection basic')

    X_train = pd.read_csv(os.path.join(dir, 'X_train_scaled.csv'))
    X_test = pd.read_csv(os.path.join(dir, 'X_test_scaled.csv'))
    y_train = pd.read_csv(os.path.join(dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(dir, 'y_test.csv')).values.ravel()

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name='RandomForest_base'):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        

if __name__ == '__main__':
    train_model('hantavirus_dataset_preprocessing')