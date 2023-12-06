from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
import pathlib
import pandas as pd
import numpy as np

distances = ["euclidean", "squared", "dtw", "ddtw", "wdtw", "wddtw", "lcss", "edr", "erp", "msm", "twe"]

path_to_data = pathlib.Path("/media/aazzari/DATA/UCRArchive_2018/")
datasets = sorted(path_to_data.iterdir())

scores = np.zeros((len(datasets), len(distances)))
# Iterate over folders in the path
for i, folder in enumerate(datasets):
    print("Processing ", folder.name)
    train = pd.read_csv(folder / (folder.name + "_TRAIN.tsv"), sep="\t", header=None)
    test = pd.read_csv(folder / (folder.name + "_TEST.tsv"), sep="\t", header=None)

    X_train, y_train = train.iloc[:, 1:].to_numpy(), train.iloc[:, 0].to_numpy()
    X_test, y_test = test.iloc[:, 1:].to_numpy(), test.iloc[:, 0].to_numpy()

    for j, distance in enumerate(distances):
        classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance=distance, n_jobs=-1)
        classifier.fit(X_train, y_train)
        scores[i, j] = classifier.score(X_test, y_test)
        np.savetxt("other_distances_scores.csv", scores, delimiter=",")

scores = pd.DataFrame(scores, index=[folder.name for folder in datasets], columns=distances)
scores.to_csv("other_distances.csv")