import pandas as pd
import  matplotlib.pyplot as plt

if __name__ == "__main__":
    distances = []
    max_trees = 500
    # Loop over the number of trees
    for n_trees in range(100, max_trees+1, 100):
        # Read the results
        results = pd.read_csv(f"results_{n_trees}.csv", index_col=0, header=0)
        # Remove the first column
        results = results.iloc[:, 1:]
        # Rename the columns
        results.columns = ["Breiman_1NN", "Ancestor_1NN", "Zhu_1NN"]
        # Groub by index and get the mean of the results
        concat_results = pd.concat([concat_results, results.groupby(results.index).mean().mean(axis=0)], axis=1) if n_trees != 100 else results.groupby(results.index).mean().mean(axis=0)
    # Read summary
    benchmark = pd.read_csv("UCRArchive2018.csv", header=0, index_col=0)
    # keep only columns with 1NN in the name
    benchmark = benchmark[benchmark.columns[benchmark.columns.str.contains("1NN")]].mean(axis=0)
    # Repeat the values of benchmark for 5 times
    benchmark = pd.concat([benchmark]*5, axis=1)
    # Rename the columns
    concat_results.columns = benchmark.columns
    overall_accuracy = pd.concat([concat_results, benchmark], axis=0)
    overall_accuracy.columns = [str(n) for n in range(100, max_trees+1, 100)]
    # Plot the results
    overall_accuracy.T.plot.line(figsize=(12, 6))
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.title("Comparison of the accuracies over all UCRArchive2018")
    plt.savefig("distances.png")
    