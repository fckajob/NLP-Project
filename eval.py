import spacy
import pandas as pd
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import ReviewModel

nlp = spacy.load('./models/reviews_1_balanced_full')



train_data = pd.read_csv('./data/train_balanced.csv')
test_data = pd.read_csv('./data/testing.csv')

model = ReviewModel(train_data, test_data, spacy_model=None, evaluate_only=False)
scores = model.evaluation(nlp)
print(scores)

json_object = json.dumps(scores, indent=4)

with open("./eval/model_eval.json", "w") as f:
    f.write(json_object)
