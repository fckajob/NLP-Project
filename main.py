import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Data clean

missing_value = ["N/a", "na", np.nan, np.NAN, np.NaN, "null"]
df = pd.read_table('sample_us.tsv', error_bad_lines=False, na_values=missing_value)

df = df.dropna()
data = df.copy()
review_id = len(data["review_id"].unique())
print("review_id: " + str(review_id))

# Visualizing the distributions of numerical variables:

#data.hist(bins=50, figsize=(20,15))
#plt.show()



# Train/Test Split
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=25)

data.info()

# Normalization : 1- converting all the characters to lowercase

training_data['review_body'] = training_data['review_body'].str.lower()
training_data['review_headline'] = training_data['review_headline'].str.lower()


# Normalization : 2- converting all whitespace and punctuation into a single space to get rid of any inconsistencies.
training_data['review_body'] = " ".join(training_data['review_body'].split())
training_data['review_headline'] = " ".join(training_data['review_headline'].split())

training_data['review_headline'] = re.sub(r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               training_data['review_body'], flags=re.VERBOSE)
training_data['review_body'] = re.sub(r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               training_data['review_headline'], flags=re.VERBOSE)

print(training_data['review_body'])

#df=df.dropna()
#data = df.copy()
#print(data.isnull().sum())

#print(data.info())
#print(data["customer_id"].unique())
#asins_unique = len(data["customer_id"].unique())
#print("Number of customer_id: " + str(asins_unique))

#data.hist(bins=50, figsize=(20,15))
#plt.show()
#data["asins"].unique()