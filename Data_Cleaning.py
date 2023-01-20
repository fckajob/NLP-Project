import numpy as np
import pandas as pd
import warnings
import re
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

import numpy as np


warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

missing_value = ["N/a", "na", np.nan, np.NAN, np.NaN, "null"]
#df = pd.read_table('./data/amazon_reviews_us_Electronics_v1_00.tsv', error_bad_lines=False, na_values=missing_value)
# TODO: Run full dataset once sure
# Debug dataset
df = pd.read_csv('./data/debug_10000.csv')
df = df[['star_rating', 'review_body']]
df = df.dropna()
# review_id = len(df["review_id"].unique())

# Normalization : 1- converting all the characters to lowercase
df['review_body'] = df['review_body'].str.lower()


# Normalization : 2- converting all whitespace and punctuation into a single space to get rid of any inconsistencies.
for idx, x in enumerate(df['review_body']):
    df['review_body'][idx] = " ".join(x.split())
    df['review_body'][idx] = re.sub(r'[^\w\s]', '', x)


# Noise Removal: Removing HTML Tags (using BeautifulSoup’s)
for idx, x in enumerate(df['review_body']):
    df['review_body'][idx] = BeautifulSoup(x, "lxml").text


# Noise Removal: Expanding Contractions
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    return phrase

# What is this for?
# for idx, x in enumerate(df['review_body']):
#     df['review_body'][idx] = (x)

# Create balanced data
df_grouped = df.groupby('star_rating').count()
min_ratings = df_grouped['review_body'].min()
max_ratings = df_grouped['review_body'].max()

df_r1 = df[df['star_rating'] == 1].sample(n = min_ratings, random_state=1)
df_r2 = df[df['star_rating'] == 2].sample(n = min_ratings, random_state=1)
df_r3 = df[df['star_rating'] == 3].sample(n = min_ratings, random_state=1)
df_r4 = df[df['star_rating'] == 4].sample(n = min_ratings, random_state=1)
df_r5 = df[df['star_rating'] == 5].sample(n = min_ratings, random_state=1)

assert(len(df_r1) == len(df_r2) == len(df_r3) == len(df_r4) == len(df_r5))

df_balanced = pd.concat([df_r1, df_r2, df_r3, df_r4, df_r5])


# Train/Test Split
training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)
train_balanced, test_balanced = train_test_split(df_balanced, test_size=0.2, random_state=25)

# Save to csv
training_data.to_csv('./data/train.csv', index=False)
testing_data.to_csv('./data/test.csv', index=False)

train_balanced.to_csv('./data/train_balanced.csv',  index=False)
test_balanced.to_csv('./data/test_balanced.csv',  index=False)

