from flask import Flask , request
import numpy as np
import pickle
from flasgger import Swagger
import pandas as pd

app = Flask(__name__)
Swagger(app)
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)
@app.route('/')

def welcome():
    
    return "welcome all"

@app.route('/predict',methods = ["GET"])

def predict_note_authentication():

    """Lets Authentication the Bank Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true

    responses:
        200:
            description: The Output Values
    """
  
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    
    return "Hello The Answer is " + str(prediction)

@app.route("/predict_file",methods = ["POST"])

def predict_note_file():

    """Lets Authentication the Bank Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The Output Values
    """
    
    df_test = pd.read_csv(request.files.get("file"))
    
    print(df_test.head())
    
    prediction = classifier.predict(df_test)
    
    return str(list(prediction))


if __name__ == "__main__":
    app.run()
