import pandas as pd
import numpy as np
import os

# WORKING VERSION!!!
missing_value = ["N/a", "na", np.nan, np.NAN, np.NaN, "null"]
df = pd.read_csv('./data/amazon_reviews_us_Electronics_v1_00.tsv', sep='\t', error_bad_lines=False, na_values=missing_value)
df = df.dropna()

if os.path.exists('./'):
    df.to_csv('./data/dataset_raw.csv', index=False)
else:
    os.mkdir('./data')
    df.to_csv('./data/dataset_raw.csv', index=False)