from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef

if __name__ == "__main__":
    # Load data
    data_path = Path("/media/aazzari/DATA/UCRArchive_2018/ArrowHead/")
    data_train = pd.read_csv(data_path / "ArrowHead_TRAIN.tsv", sep="\t", header=None)
    data_test = pd.read_csv(data_path / "ArrowHead_TEST.tsv", sep="\t", header=None)


    X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
    X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

    # Train model
    model = RandomForestClassifier()#IsolationForest(n_estimators=100)
    model.fit(X_train, y_train)

    # Print averege depth of trees
    print("Average depth of trees:", np.mean([estimator.tree_.max_depth for estimator in model.estimators_]))

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate model
    print("accuracy score:", accuracy_score(y_test, y_pred))
    #print("Accuracy score:", matthews_corrcoef(y_test, y_pred))