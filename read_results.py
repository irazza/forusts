import pandas as pd

if __name__ == "__main__":
    # Read the results
    results = pd.read_csv("results_100.csv", index_col=0, header=0)
    # Groub by index and get the mean of the results
    results = results.groupby(results.index).mean()
    results = results.applymap(lambda x: 1 - x)
    # Read summary
    summary = pd.read_csv("UCRArchive2018.csv", header=0)
    summary.index = summary["Name"]
    summary = summary.drop("Name", axis=1)
    # Join results and summary
    results = results.join(summary)
    # Look how many times accuracy_breiman is better than DTW (learned_w)
    print((results["accuracy_breiman"] < results["DTW (learned_w) "].apply(lambda x: float(x.split()[0]))).sum())