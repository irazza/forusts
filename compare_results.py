import pandas as pd

if __name__ == "__main__":
    for n_trees in [100, 200, 300]:
        # Read the results
        results = pd.read_csv(f"results_{n_trees}.csv", index_col=0, header=0)
        # Remove the first column
        results = results.iloc[:, 1:]
        # Rename the columns
        results.columns = ["Breiman_1NN", "Ancestor_1NN", "Zhu_1NN"]
        # Groub by index and get the mean of the results
        results = results.groupby(results.index).mean().mean(axis=0)
        # Read summary
        benchmark = pd.read_csv("UCRArchive2018.csv", header=0, index_col=0)
        # keep only columns with 1NN in the name
        benchmark = benchmark[benchmark.columns[benchmark.columns.str.contains("1NN")]].mean(axis=0)
        # Compute the mean of every column
        overall_mean = pd.concat([results, benchmark], axis=0).sort_values(ascending=False)
        # Print the results
        print(overall_mean)