import os
import numpy as np
import pandas as pd
import warnings
import re
import spacy

from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup


warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Data cleaning

missing_value = ["N/a", "na", np.nan, np.NAN, np.NaN, "null"]
df = pd.read_table('amazon_reviews_us_Electronics_v1_00.tsv', error_bad_lines=False, na_values=missing_value)

df = df.dropna()
data = df.copy()
review_id = len(data["review_id"].unique())
#print("review_id: " + str(review_id))

# Visualizing the distributions of numerical variables:

#data.hist(bins=50, figsize=(20,15))
#plt.show()



# Train/Test Split
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=25)

#data.info()

# Normalization : 1- converting all the characters to lowercase
training_data['review_body'] = training_data['review_body'].str.lower()
training_data['review_headline'] = training_data['review_headline'].str.lower()



# Normalization : 2- converting all whitespace and punctuation into a single space to get rid of any inconsistencies.
review_body = re.sub(' +', ' ', str(training_data['review_body']))
review_headline = re.sub(' +', ' ', str(training_data['review_headline']))



review_body = re.sub(r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               str(training_data['review_body']), flags=re.VERBOSE)
review_headline = re.sub(r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               str(training_data['review_headline']), flags=re.VERBOSE)


# Noise Removal: Removing HTML Tags (using BeautifulSoup’s)

review_body = BeautifulSoup(review_body, "lxml").text



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

review_body = decontracted(review_body)


print(len(review_body))
print(review_body)


# TODO: Either save cleaned dataset as new file or warp in function and return to model.py