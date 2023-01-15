# TODO: Spacy Modelimport spacy
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.util import minibatch
from tqdm import tqdm # loading bar
from spacy.training.example import Example
import random
import spacy
import os

#spacy.prefer_gpu()

config = {
        "threshold": 0.5,
        "model": DEFAULT_SINGLE_TEXTCAT_MODEL,
        "version": 1
        }

class ReviewModel:
    def __init__(self):
        self.nlp =  spacy.load("en_core_web_sm")
        self.config = config
        self.textcat = self.nlp.add_pipe("textcat")
        self.TRAIN_DATA = list()


    # Create Text categorizer instance
    #textcat = nlp.add_pipe("textcat",config=config)

    def setup(self):
        #Disable all other pipes except Text Categorizer
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'textcat']
        self.nlp.disable_pipes(*other_pipes)

        # Add our desired labels for the Text Categorizer
        self.textcat.add_label("1 Star")
        self.textcat.add_label("2 Star")
        self.textcat.add_label("3 Star")
        self.textcat.add_label("4 Star")
        self.textcat.add_label("5 Star")


    def createExample(self):
        exampleText = "i used to beats headphones but after their partnership with monster cables ended, the quality of their headphones went down hill. i was replacing my beats with new ones every 9-12 months since the headphones keep blowing out. these v-moda headphones are great, never had any issue on the construction and durability of these headphones. the sound quality is top notch and provides a deeper bass sound than beats and bose in-ear headphones. bit on the pricey side, but worth the purchase if you're an avid listener."  

        annot = {
            "cats":{
                "5 Star" : True,
                "4 Star" : False,
                "3 Star" : False,
                "2 Star" : False,
                "1 Star" : False,
                }
        }

        for i in range(1):
            self.nlp.make_doc(exampleText)
            exampleTuple = (exampleText,annot)
            self.TRAIN_DATA.append(exampleTuple)

        exampleList = list()
        for text, annotations in self.TRAIN_DATA:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            exampleList.append(example)

        self.textcat.initialize(lambda: exampleList, nlp=self.nlp)

    #print(self.TRAIN_DATA)
    def train(self):
    #Training
        optimizer = self.nlp.resume_training()
        for itn in tqdm(range(5)):
            print("Starting iteration " + str(itn))
            random.shuffle(self.TRAIN_DATA)
            #create batches of training data
            batches = minibatch(self.TRAIN_DATA, size=50)
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
            os.mkdir(f'./models/reviews_{self.config["version"]}')
            self.nlp.to_disk(f'./models/./reviews_{self.config["version"]}')
        else:
            os.mkdir(f'./models{self.config["version"]}')
            os.mkdir(f'./models/reviews_{self.config["version"]}')
            self.nlp.to_disk(f'./models/reviews_{self.config["version"]}')
    ##print('Iterations',iterations,'ExecutionTime',time.time()-start)

    def execute(self):
        self.setup()
        self.train()