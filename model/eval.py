import spacy
import pandas as pd

from model import ReviewModel

nlp = spacy.load('./models/reviews_2')



train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

model = ReviewModel(train_data, test_data)
scores = model.evaluation(nlp)