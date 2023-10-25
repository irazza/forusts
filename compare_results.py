import pandas as pd
import  matplotlib.pyplot as plt

if __name__ == "__main__":
    # Read the results
    results = pd.read_csv(f"results_500.csv", index_col=0, header=0)
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
    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 12))
    bars = ax.barh(overall_accuracy.index, overall_accuracy.to_numpy())
    ax.bar_label(bars, fmt='%.4f')
    plt.xlabel("Distance metrics")
    plt.ylabel("Accuracy")
    plt.title("Comparison of the accuracies over all UCRArchive2018")
    plt.savefig("distances.png", dpi=300)
    