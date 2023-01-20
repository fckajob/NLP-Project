# TODO: Spacy Modelimport spacy
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.util import minibatch
from tqdm import tqdm # loading bar
from spacy.training.example import Example
import random
import spacy
import os



config = {
        "threshold": 0.5,
        "model": DEFAULT_SINGLE_TEXTCAT_MODEL,
        "version": 1
        }

class ReviewModel:
    def __init__(self, train, test):
        #Enable if GPU is preferred
        #spacy.prefer_gpu()
        self.nlp =  spacy.load("en_core_web_sm")
        self.config = config
        self.textcat = self.nlp.add_pipe("textcat")
        self.TRAIN_DATA = list()
        self.train = train
        self.test = test
        self.allowed_labels = ['1','2','3','4','5']


    # Create Text categorizer instance
    #textcat = nlp.add_pipe("textcat",config=config)

    def setup(self):
        #Disable all other pipes except Text Categorizer
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'textcat']
        self.nlp.disable_pipes(*other_pipes)

        # Add our desired labels for the Text Categorizer
        self.textcat.add_label("1")
        self.textcat.add_label("2")
        self.textcat.add_label("3")
        self.textcat.add_label("4")
        self.textcat.add_label("5")

        #Initialize the model with a couple of records as training data
        initial_data = self.createTrainData(5000)
        self.textcat.initialize(lambda: initial_data, nlp=self.nlp)


    #Returns a annotation dict in our desired format
    def createAnnotation(self, rating: str):
        annot = {
            "cats":{
                '5' : False,
                '4' : False,
                '3' : False,
                '2' : False,
                '1' : False,
                }
            }

        annot['cats'][rating] = True
        return annot

    #Creates finished annotated training data as a list of Example objects
    def createTrainData(self, sample_size):
        data = self.train
        input_list = list()
        for index, row in data.sample(n=sample_size).iterrows():
             # Only take valid labels
            if str(row['star_rating']).strip() not in self.allowed_labels or len(str(row['review_body'])) < 3:
                continue
            doc = self.nlp.make_doc(str(row['review_body']))
            annotation = self.createAnnotation(str(row['star_rating']).strip())
            train_dp = Example.from_dict(doc, annotation)
            input_list.append(train_dp)
        return input_list


    def full_training(self):
    #Training
        optimizer = self.nlp.resume_training()
        self.TRAIN_DATA = list()

        for index, row in self.train.sample(n=100000).iterrows():
            # Only take valid labels
            if str(row['star_rating']).strip() not in self.allowed_labels or len(str(row['review_body']).strip()) < 3:
                continue
            annotation = self.createAnnotation(str(row['star_rating']).strip())
            self.TRAIN_DATA.append((str(row['review_body']).strip(), annotation))

        for itn in tqdm(range(30)):
            print()
            print("Starting iteration " + str(itn))

            random.shuffle(self.TRAIN_DATA)
            #create batches of training data
            batches = minibatch(self.TRAIN_DATA, size=100)
            losses = {}
            #Implement batching
            for batch in batches:
                exampleLst = []
                for text, annotations in batch:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    exampleLst.append(example)
                losses = self.textcat.update(exampleLst, sgd=optimizer)
            print(losses)

        if os.path.exists('./models'):
            os.makedirs(f'./models/reviews_{self.config["version"]}', exist_ok=True)
            self.nlp.to_disk(f'./models/./reviews_{self.config["version"]}')
        else:
            os.makedirs(f'./models{self.config["version"]}')
            os.makedirs(f'./models/reviews_{self.config["version"]}')
            self.nlp.to_disk(f'./models/reviews_{self.config["version"]}')
    ##print('Iterations',iterations,'ExecutionTime',time.time()-start)

    def executeTraining(self):
        self.setup()
        self.full_training()

    def evaluation(self):
        pass

    def evaluate(self):
        self.setup()
        self.createTrainData(self.test)