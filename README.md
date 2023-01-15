# NLP-Project

Dataset
https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz

# Setup project
1. Create a `venv`with `python -m venv venv`
2. Afterward activate the venv with `source bin/venv/activate` (`.\bin\Scripts\activate.bat` on Windows)
3. Run `pip install -r requirements.txt`
Your project all set up

# Run training
Either run the `train.py` file or from the console run `python train.py`to start training. That requires to have the training data already cleaned and in the correct form saved under `/data` with `train.csv` and `test.csv` naming convention

# Run backend server
Run `uvicorn main:app --reload`
Afterward easily send `post`request via e.g. Postman

# Run inference
Create a post request to `http://127.0.0.1:8000/api/predict` with a body in the form of `{"text":"[YOUR TEXT]"}`