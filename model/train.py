from model import ReviewModel
import pandas as pd


# TODO: Load data from dataloader

if __name__ == '__main__':
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    model = ReviewModel(train=train_data, test=test_data)

    model.execute()

