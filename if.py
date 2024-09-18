from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import numpy as np

if __name__ == "__main__":
    avg_score = 0
    data = np.loadtxt("/media/DATA/albertoazzari/IFDatasets/annthyroid.csv", delimiter=",")
    X = data[:, 1:]
    y = data[:, 0]
    y = np.where(y == 1, -1, 1)
    for _ in range(10):
        model = IsolationForest(n_estimators=100)
        model.fit(X)
        print(sum([x.get_depth() for x in model.estimators_])/100)
        score = roc_auc_score(y, model.decision_function(X))
        avg_score += score
        # print(score)
    print(avg_score / 10)
