from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, adjusted_rand_score
import numpy as np
import pathlib

if __name__ == "__main__":
    true = np.loadtxt("true.csv", delimiter=",")
    n_classes = len(np.unique(true))
    dist_matrix = np.loadtxt("dist.csv", delimiter=",")
    model = AgglomerativeClustering(n_clusters=n_classes, metric="precomputed", linkage="average")
    model.fit(dist_matrix)
    prediction = model.labels_

    # prediction = np.loadtxt("prediction.csv", delimiter=",")

    print(adjusted_rand_score(true, prediction))
    # datasets = sorted([x for x in pathlib.Path("/media/DATA/albertoazzari/ADMEP/").iterdir()])
    # # first figit is int all the others 3000 are doubles
    # fmt = "%d" + ",%f" * 3000
    # for d in datasets:
    #     data0 = np.loadtxt(d/(d.name+"_0.csv"), delimiter=",")
    #     X0, y0 = data0[:, 1:], data0[:, 0].reshape(-1, 1)
    #     data0 = np.hstack([y0, X0])
    #     np.savetxt(d/(d.name+"_0.csv"), data0, delimiter=",", fmt=fmt)
    #     data1 = np.loadtxt(d/(d.name+"_1.csv"), delimiter=",")
    #     X1, y1 = data1[:, 1:], data1[:, 0].reshape(-1, 1)
    #     data1 = np.hstack([y1, X1])
    #     np.savetxt(d/(d.name+"_1.csv"), data1, delimiter=",", fmt=fmt)
    # datasets = sorted([x for x in pathlib.Path("/media/DATA/albertoazzari/IFDatasets/").glob("*.csv")])
    # for p in datasets:
    #     avg_score = 0
    #     data = np.loadtxt(p, delimiter=",")
    #     X = data[:, 1:]
    #     y = data[:, 0]
    #     y = np.where(y == 1, -1, 1)
    #     for _ in range(10):
    #         model = IsolationForest(n_estimators=100)
    #         model.fit(X)
    #         score = roc_auc_score(y, model.decision_function(X))
    #         avg_score += score
    #         # print(score)
    #     print(avg_score / 10)

    # data = np.loadtxt("annthyroid.csv", delimiter=",")
    # X = data[:, 1:]
    # y = data[:, 0]
    # y = np.where(y == 1, -1, 1)
    # for _ in range(1):
    #     model = IsolationForest(n_estimators=100)
    #     model.fit(X)
    #     print(sum([x.get_depth() for x in model.estimators_])/100)
    #     score = roc_auc_score(y, model.decision_function(X))
    #     avg_score += score
    #     # print(score)
    # print(avg_score / 10)
