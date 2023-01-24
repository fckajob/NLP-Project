import pandas as pd
import json
import os

from model import ReviewModel


# TODO: Load data from dataloader

if __name__ == '__main__':
    train_data = pd.read_csv('./data/train_balanced.csv')
    test_data = pd.read_csv('./data/testing.csv')
    model = ReviewModel(train=train_data, test=test_data, spacy_model=None, evaluate_only=False)
    sample_size = len(train_data)
    model.executeTraining(sample_size=100)

    scores = model.evaluate()
    print(scores)

    json_object = json.dumps(scores, indent=4)

    if os.path.exists('../eval'):
        with open("../eval/model_evaluation_balanced_150000.json", "w") as f:
            f.write(json_object)
    else:
        os.mkdir('../eval',  exist_ok=True)
        with open("../eval/model_evaluation_balanced_150000.json", "w") as f:
            f.write(json_object)
