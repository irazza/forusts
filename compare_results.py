from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_splitq

if __name__ == "__main__":

    # Read the results
    admep = pd.read_csv("admep.csv", index_col=0, header=0)
    # Group by index and take the mean
    admep = admep.groupby(admep.index).mean()
    # print the results
    print(admep)
    # ecg = pd.read_csv("ecg.csv", header=None)
    # X, y = ecg.iloc[:, :-1].to_numpy(), ecg.iloc[:, -1].to_numpy()
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    # print(np.unique(y_train, return_counts=True))
    # print(np.unique(y_test, return_counts=True))
    
    # y_train = np.where(y_train == 1, 0, 1)
    # y_test = np.where(y_test == 1, 0, 1)

    # print(np.unique(y_train, return_counts=True))
    # print(np.unique(y_test, return_counts=True))

    # pd.DataFrame(np.hstack([y_train.reshape(-1, 1), X_train])).to_csv("AD/ECG/ECG_TRAIN.tsv", index=False, header=False, sep="\t")
    # pd.DataFrame(np.hstack([y_test.reshape(-1, 1), X_test])).to_csv("AD/ECG/ECG_TEST.tsv", index=False, header=False, sep="\t")
    
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