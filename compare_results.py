import pandas as pd
import  matplotlib.pyplot as plt
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    data_train = pd.read_csv("UCRArchive_2018/ACSF1/ACSF1_TRAIN.tsv", sep="\t", header=None)
    X_train, y_train = data_train.iloc[:, 1:].to_numpy(), data_train.iloc[:, 0].to_numpy()

    data_test = pd.read_csv("UCRArchive_2018/ACSF1/ACSF1_TEST.tsv", sep="\t", header=None)
    X_test, y_test = data_test.iloc[:, 1:].to_numpy(), data_test.iloc[:, 0].to_numpy()

    clf = TimeSeriesForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    print(accuracy_score(clf.predict(X_test), y_test))
    # # Read the results
    # results = pd.read_csv(f"tsf_200.csv", index_col=0, header=0)
    # # Group by index and take the mean
    # results = results.groupby(results.index).mean()
    # # Rename the columns
    # results.columns = ["Breiman_1NN", "Ancestor_1NN", "Zhu_1NN"]
    # # Mean over all the datasets
    # results = results.mean(axis=0)
    # # Read summary
    # benchmark = pd.read_csv("UCRArchive2018.csv", header=0, index_col=0)
    # # keep only columns with 1NN in the name
    # benchmark = benchmark[benchmark.columns[benchmark.columns.str.contains("1NN")]].mean(axis=0)
    # # Rename the columns
    # overall_accuracy = pd.concat([results, benchmark], axis=0)
    # # Sort the results
    # overall_accuracy = overall_accuracy.sort_values(ascending=True)
    # # Plot the results
    # fig, ax = plt.subplots(figsize=(15, 12))
    # bars = ax.barh(overall_accuracy.index, overall_accuracy.to_numpy())
    # ax.bar_label(bars, fmt='%.4f')
    # plt.xlabel("Distance metrics")
    # plt.ylabel("Accuracy")
    # plt.title("Comparison of the accuracies over all UCRArchive2018")
    # plt.savefig("distances.png", dpi=300)