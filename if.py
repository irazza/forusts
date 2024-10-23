from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import numpy as np
import pathlib
import time

def test_if(datasets):
    for p in datasets:
        avg_score = 0
        data = np.loadtxt(p, delimiter=",")
        X = data[:, 1:]
        y = data[:, 0]
        y = np.where(y == 1, -1, 1)
        for _ in range(10):
            model = IsolationForest(n_estimators=100)
            model.fit(X)
            score = roc_auc_score(y, model.decision_function(X))
            avg_score += score
        # print dataset name and average score with 2 decimal digits
        print(p.name, round(avg_score / 10, 2))


def test_rf(datasets, repetitions=10):
        for p in datasets[:5]:
            avg_score = 0
            train = np.loadtxt(p/f"{p.name}_TRAIN.tsv", delimiter="\t")
            test = np.loadtxt(p/f"{p.name}_TEST.tsv", delimiter="\t")
            X_train, y_train = train[:, 1:], train[:, 0]
            X_test, y_test = test[:, 1:], test[:, 0]
            start_time = time.time()
            for _ in range(repetitions):
                model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features=None)
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                avg_score += score
            # print dataset name and average score with 2 decimal digits
            print(p.name, round(avg_score / repetitions, 2), round(time.time() - start_time, 2))

if __name__ == "__main__":
    datasets = sorted([x for x  in pathlib.Path("../DATA/ucr").iterdir() if x.is_dir()])
    test_rf(datasets, 1)