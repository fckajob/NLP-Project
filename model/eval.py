import spacy
import pandas as pd
import json

from model import ReviewModel

nlp = spacy.load('./models/reviews_2')



train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

model = ReviewModel(train_data, test_data)
scores = model.evaluation(nlp)
print(scores)

json_object = json.dumps(scores, indent=4)

with open("./eval/model_evaluation.json", "w") as f:
    f.write(json_object)
