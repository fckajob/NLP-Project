import numpy as np
import pandas as pd
import warnings
import re
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

import numpy as np

import string

warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Data cleaning

missing_value = ["N/a", "na", np.nan, np.NAN, np.NaN, "null"]
df = pd.read_table('amazon_reviews_us_Electronics_v1_00.tsv', error_bad_lines=False, na_values=missing_value)
df = df.dropna()
review_id = len(df["review_id"].unique())
#print("review_id: " + str(review_id))

# Visualizing the distributions of numerical variables:

#data.hist(bins=50, figsize=(20,15))
#plt.show()





#data.info()

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

for idx,x in enumerate(df['review_body']):
    df['review_body'][idx] = decontracted(x)


# Train/Test Split
training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)


#df['review_body']=training_data['review_body']

header = ["star_rating","review_body"]
training_data.to_csv('output.csv', columns = header)
