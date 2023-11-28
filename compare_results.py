from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef

if __name__ == "__main__":
    # Load data
    data_path = Path("/media/aazzari/DATA/ADMEP/0_0_0_0/")
    data_train = pd.read_csv(data_path / "0_0_0_0_TRAIN.tsv", sep="\t", header=None)
    data_test = pd.read_csv(data_path / "0_0_0_0_TEST.tsv", sep="\t", header=None)


    X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
    X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

    # Train model
    model = IsolationForest(n_estimators=100)
    model.fit(X_train, y_train)

    # Predict anomaly scores for the entire dataset
    anomaly_scores = model.score_samples(X_test)

    # Convert the scores to anomaly probabilities
    anomaly_probs = 2**(-anomaly_scores / anomaly_scores.mean())

    print(anomaly_scores)

    # Evaluate model
    #print("accuracy score:", accuracy_score(y_test, y_pred))
    #print("Accuracy score:", matthews_corrcoef(y_test, y_pred))