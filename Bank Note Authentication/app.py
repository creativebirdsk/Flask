from flask import Flask , request
import numpy as np 
import pickle 
import pandas as pd 

app = Flask(__name__)

pickle_in = open("classifier.pkl" , 'rb')

classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():

	return "WELCOME ALL"

@app.route('/predict')

def predict_bank_Authentication():


	variance = request.args.get("variance")
	skewness = request.args.get("skewness")
	curtosis = request.args.get("curtosis")
	entropy  = request.args.get("entropy")

	prediction= classifier.predict([[variance, skewness , curtosis , entropy]])

	print(prediction)

	return  "HELLO THE ANSWER IS " + str(prediction)

@app.route("/predict_file")

def predict_file_bankauthentication():

	df = pd.read_csv(request.files.get("file"))

	prediction= classifier.predict(df)

	return str(list(prediction))

if __name__ == '__main__':

	app.run()