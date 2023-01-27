# TODO: Statistical data anylsis with some plots
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from autoviz import data_cleaning_suggestions, AutoViz_Class


def data_cleaning_suggestions():
    missing_value = ["N/a", "na", np.nan, np.NAN, np.NaN, "null"]
    #df_raw = pd.read_table('./data/amazon_reviews_us_Electronics_v1_00.tsv', error_bad_lines=False, na_values=missing_value)

    df_test = pd.read_csv('./data/test.csv')

    #data_cleaning_suggestions(df_raw)
    data_cleaning_suggestions(df_test)


missing_value = ["N/a", "na", np.nan, np.NAN, np.NaN, "null"]
df = pd.read_table('./data/amazon_reviews_us_Electronics_v1_00.tsv', error_bad_lines=False, na_values=missing_value)

sns.pairplot(df, hue="star_rating")
plt.show()