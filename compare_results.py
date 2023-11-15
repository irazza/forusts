from pathlib import Path
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    n_samples, n_outliers = 120, 40
    rng = np.random.RandomState(0)
    covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
    cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
    cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
    outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

    X = np.concatenate([cluster_1, cluster_2, outliers])
    y = np.concatenate(
        [np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)]
    )
    print(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    print((y_train == -1).mean())
    # np.savetxt("IF_TRAIN.tsv", np.hstack([y_train.reshape(-1, 1), X_train]), delimiter="\t")
    # np.savetxt("IF_TEST.tsv", np.hstack([y_test.reshape(-1, 1), X_test]), delimiter="\t")

    clf = IsolationForest()
    clf.fit(X_train)
    y_pred = clf.predict(X_test)
    y_test = np.where(y_test == -1, 1, 0)
    y_pred = np.where(y_pred == -1, 1, 0)

    print((y_pred==y_test).mean())
    # Read the results
    # path = Path("tsf_300_g_b_10.csv")
    # results = pd.read_csv(path, index_col=0, header=0)
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
    # # Remove Ancestor and Zhu distances
    # # overall_accuracy = overall_accuracy[~overall_accuracy.index.str.contains("Ancestor")]
    # # overall_accuracy = overall_accuracy[~overall_accuracy.index.str.contains("Zhu")]
    # # Plot the results
    # fig, ax = plt.subplots(figsize=(15, 12))
    # sns.barplot(x=overall_accuracy.values, y=overall_accuracy.index, ax=ax)
    # ax.bar_label(ax.containers[0], fontsize=14, fmt='%.4f');
    # plt.title("Average accuracy of 1NN on the UCR Archive")
    # plt.xlabel("Accuracy")
    # plt.ylabel("Distance")
    # plt.tight_layout()
    # plt.savefig(path.with_suffix(".png"))