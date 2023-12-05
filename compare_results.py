from pathlib import Path
import pandas as pd
from scipy.stats import ttest_rel


if __name__ == "__main__":
    results = pd.read_csv(Path("admepTSIF_T100_R50.csv"), index_col=0, header=0)

    # Group by index and get the mean of the results
    results = results.groupby(results.index).max()

    # Print the results
    print(results)