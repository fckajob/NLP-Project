# NLP-Project

Dataset
https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz

# Setup project
1. Create a `venv`with `python -m venv venv`
2. Afterward activate the venv with `source bin/venv/activate` (`.\bin\Scripts\activate.bat` on Windows)
3. Run `pip install -r requirements.txt`
Your project all set up

# Get Data
1. Download the amazon_reviews_us_Electronics dataset from the official source by amazon
2. Create  a "data" folder in the root directory and drop the unzipped data there
3. Run `python data_loader.py` to gain access to the raw data
4. Afterwards run `python data_cleaning.py` to preprocess the data and create a balanced training dataset as well as a testing dataset

# Run training
Either run the `train.py` file or from the console run `python train.py`to start training. That requires to have the training data already cleaned and in the correct form saved under `/data` with `train.csv` and `test.csv` naming convention

# Evaluate trained model
To evaluate a trained model after the initial training evaluation run `python eval.py``
That will initialise the model class and run the evaluation on a given test dataset. The results will be saved as JSON to `./eval` directory

# Run inference
## Run backend server
Run `uvicorn main:app --reload`
Afterward easily send `post`request via e.g. Postman

## Create request
Create a post request to `http://127.0.0.1:8000/api/predict` with a body in the form of `{"text":"[YOUR TEXT]"}`