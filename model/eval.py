import spacy
import pandas as pd
import json

from model import ReviewModel

nlp = spacy.load('./models/reviews_1_balanced_25000_per_class')



train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
test_data = test_data[:1000]

model = ReviewModel(train_data, test_data, spacy_model=None, evaluate_only=False)
scores = model.evaluation(nlp)
print(scores)

json_object = json.dumps(scores, indent=4)

with open("./eval/model_evaluation_25000.json", "w") as f:
    f.write(json_object)
