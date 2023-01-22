import pandas as pd
import json

from model import ReviewModel


# TODO: Load data from dataloader

if __name__ == '__main__':
    train_data = pd.read_csv('./data/train_balanced.csv')
    test_data = pd.read_csv('./data/test_balanced.csv')
    model = ReviewModel(train=train_data, test=test_data)
    model.executeTraining()

    scores = model.evaluate()
    print(scores)

    json_object = json.dumps(scores, indent=4)

    with open("./eval/model_evaluation.json", "w") as f:
        f.write(json_object)
