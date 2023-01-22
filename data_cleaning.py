import numpy as np
import pandas as pd
import warnings
import re
import nltk

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Directly reading the amazon .tsv caused errors in data at the end, without concrete cause =>
# missing_value = ["N/a", "na", np.nan, np.NAN, np.NaN, "null"]
# df = pd.read_csv('./data/amazon_reviews_us_Electronics_v1_00.tsv', sep='\t', error_bad_lines=False, na_values=missing_value)
df = df.dropna()
#df = pd.read_csv('./data/dataset_raw.csv')
df = df[['star_rating', 'review_body']]

# Normalization : 1- converting all the characters to lowercase
df['review_body'] = df['review_body'].str.lower()

# Normalization : 2- converting all whitespace and punctuation into a single space to get rid of any inconsistencies.
# Not the Error
for idx, row in enumerate(df['review_body']):
    df['review_body'][idx] = " ".join(row.split())
    df['review_body'][idx] = re.sub(r'[^\w\s]', '', row)


# Remove HTML elements
# Not the Error
df['review_body'] = df['review_body'].str.replace(r'<[^<>]*>', '', regex=True)


# Noise Removal: Expanding Contractions
# Not the Error
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

for idx, row in enumerate(df['review_body']):
    df['review_body'][idx] = decontracted(row)

# Since in our use case the outcome could be dependent on the stop word (e.g. is vs. isn't) we keep them in 
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# cachedStopWords = stopwords.words("english")

# df_stopword = df
# # Remove stopwords
# for idx, row in enumerate(df_stopword['review_body']):
#     df_stopword['review_body'][idx] = ' '.join([word for word in row.split() if word not in cachedStopWords])

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
# training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)
# train_balanced, test_balanced = train_test_split(df_balanced, test_size=0.2, random_state=25)

## Manually creating train-test-split
train_size = int(len(df)*0.8)
training_data = df[:train_size]
testing_data = df[train_size:]

train_size_balanced = (int(len(df_balanced)*0.8))

# To also have balanced datasets for train and test split
train_balanced = pd.concat([df_r1[:train_size_balanced], df_r2[:train_size_balanced], df_r3[:train_size_balanced], df_r4[:train_size_balanced], df_r5[:train_size_balanced]])
test_balanced = pd.concat([df_r1[train_size_balanced:], df_r2[train_size_balanced:], df_r3[train_size_balanced:], df_r4[train_size_balanced:], df_r5[train_size_balanced:]])
train_balanced = train_balanced.sample(frac=1, random_state=1).reset_index()
test_balanced = test_balanced.sample(frac=1, random_state=1).reset_index()

# Save to csv
training_data.to_csv('./data/train.csv', index=False)
testing_data.to_csv('./data/test.csv', index=False)

df_balanced.to_csv('./data/balanced_full.csv', index=False)
train_balanced.to_csv('./data/train_balanced.csv',  index=False)
test_balanced.to_csv('./data/test_balanced.csv',  index=False)

