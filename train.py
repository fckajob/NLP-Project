from model.model import ReviewModel


# TODO: Load data from dataloader

if __name__ == '__main__':
    model = ReviewModel()
    model.setup()
    model.train()