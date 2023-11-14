import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    # adiac_train = pd.read_csv("UCRArchive_2018/ACSF1/ACSF1_TRAIN.tsv", sep="\t", header=None).to_numpy()
    # X_train, y_train = adiac_train[:, 1:], adiac_train[:, 0]
    # for x in X_train:
    #     catch22 = pycatch22.catch22_all(x)
    #     for i in range(len(catch22["names"])):
    #         print(f'{catch22["names"][i]}={catch22["values"][i]}')
    #     break
    # Read the results
    results = pd.read_csv(f"tsf_200_g_b_50.csv", index_col=0, header=0)
    # Group by index and take the mean
    results = results.groupby(results.index).mean()
    # Rename the columns
    results.columns = ["Breiman_1NN", "Ancestor_1NN", "Zhu_1NN"]
    # Mean over all the datasets
    results = results.mean(axis=0)
    # Read summary
    benchmark = pd.read_csv("UCRArchive2018.csv", header=0, index_col=0)
    # keep only columns with 1NN in the name
    benchmark = benchmark[benchmark.columns[benchmark.columns.str.contains("1NN")]].mean(axis=0)
    # Rename the columns
    overall_accuracy = pd.concat([results, benchmark], axis=0)
    # Sort the results
    overall_accuracy = overall_accuracy.sort_values(ascending=True)
    # Remove Ancestor and Zhu distances
    # overall_accuracy = overall_accuracy[~overall_accuracy.index.str.contains("Ancestor")]
    # overall_accuracy = overall_accuracy[~overall_accuracy.index.str.contains("Zhu")]
    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.barplot(x=overall_accuracy.values, y=overall_accuracy.index, ax=ax)
    ax.bar_label(ax.containers[0], fontsize=14, fmt='%.4f');
    plt.title("Average accuracy of 1NN on the UCR Archive")
    plt.xlabel("Accuracy")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("results.png")