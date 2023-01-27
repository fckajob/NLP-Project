# TODO: Spacy Modelimport spacy
import random
import spacy
import os

from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.util import minibatch
from tqdm import tqdm # loading bar
from spacy.training.example import Example
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc




config = {
        'threshold': 0.5,
        'model': DEFAULT_SINGLE_TEXTCAT_MODEL,
        'version': 1,
        'data_used': 'balanced_50000_per_class'
        }

class ReviewModel:
    def __init__(self, train, test, spacy_model, evaluate_only:bool):
        #Enable if GPU is preferred
        spacy.prefer_gpu()
        if evaluate_only:
            self.nlp = spacy_model
        else:
            self.nlp =  spacy.load('en_core_web_sm')
            self.textcat = self.nlp.add_pipe('textcat')
        
        self.config = config
        self.train = train
        self.test = test
        self.allowed_labels = ['1','2','3','4','5']


    # Create Text categorizer instance
    #textcat = nlp.add_pipe('textcat',config=config)

    def setup(self):
        #Disable all other pipes except Text Categorizer
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'textcat']
        self.nlp.disable_pipes(*other_pipes)

        # Add our desired labels for the Text Categorizer
        self.textcat.add_label('1')
        self.textcat.add_label('2')
        self.textcat.add_label('3')
        self.textcat.add_label('4')
        self.textcat.add_label('5')

        #Initialize the model with a couple of records as training data
        initial_data = self.createTrainData(10)
        self.textcat.initialize(lambda: initial_data, nlp=self.nlp)


    #Returns a annotation dict in our desired format
    def createAnnotation(self, rating: str):
        annot = {
            'cats':{
                '5' : False,
                '4' : False,
                '3' : False,
                '2' : False,
                '1' : False,
                }
            }

        annot['cats'][rating] = True
        return annot

    #Only used to create data required for initialisation of textcat
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


    def full_training(self, sample_size: int):
    #Training
        optimizer = self.nlp.resume_training()
        # Spacy requires certain form of training data => TRAIN_DATA
        TRAIN_DATA = list()

        for index, row in self.train.sample(n=sample_size).iterrows():
            # Only take valid labels
            if str(row['star_rating']).strip() not in self.allowed_labels or len(str(row['review_body']).strip()) < 3:
                continue
            annotation = self.createAnnotation(str(row['star_rating']).strip())
            TRAIN_DATA.append((str(row['review_body']).strip(), annotation))

        for itn in tqdm(range(30)):
            print()
            print('Starting iteration ' + str(itn))

            random.shuffle(TRAIN_DATA)
            #create batches of training data
            batches = minibatch(TRAIN_DATA, size=100)
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
            self.nlp.to_disk(f'./models/reviews_{self.config["version"]}_{self.config["data_used"]}')
        else:
            os.makedirs(f'./models')
            self.nlp.to_disk(f'./models/reviews_{self.config["version"]}_{self.config["data_used"]}')
    ##print('Iterations',iterations,'ExecutionTime',time.time()-start)

    def executeTraining(self, sample_size: int):
        self.setup()
        self.full_training(sample_size)

    def evaluation(self, model):
        TEST_DATA = list()
        y_true = []
        for index, row in self.test.sample(n=len(self.test)).iterrows():
        # Only take valid labels
            if str(row['star_rating']).strip() not in self.allowed_labels or len(str(row['review_body']).strip()) < 3:
                continue
            y_true.append(int(row['star_rating']))
            annotation = self.createAnnotation(str(row['star_rating']).strip())
            TEST_DATA.append((str(row['review_body']).strip(), annotation))

        y_pred = []
        for input_, annotations in TEST_DATA:
            pred = model(input_)
            y_pred.append(int(max(pred.cats, key = lambda k: pred.cats[k])))

        f1 = f1_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=5)
        area_under_the_curve = auc(fpr, tpr)

        return {
            'f1_score': f1,
            'accuracy': accuracy,
            'AUC': area_under_the_curve,
            'model_version': config['version'],
            'data_used': config['data_used']
        }

    def evaluate(self):
        #self.setup()
        scores = self.evaluation(self.nlp)
        return scores