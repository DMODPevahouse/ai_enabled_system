from flask import Flask, render_template
from flask import request
import os
import shutil
from model import Movie_Review_Model
import json
from pandas import json_normalize
import pickle



app = Flask(__name__)


# http://localhost:8786/infer?transmission=automatic&color=blue&odometer=12000&year=2020&bodytype=suv&price=20000

@app.route('/stats', methods=['GET'])
def getStats():
    return "Results in results/deep_report.csv and results/trad_report.csv"


@app.route('/test', methods=['GET'])
def test():
    model.train(True)
    return "Results in results/deep_report.csv and results/trad_report.csv"

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    data = json_normalize(json_data)
    prediction = modal_model.json_predict(data)
    return f'The review you have left gives a rating of {prediction}'

if __name__ == "__main__":
    flaskPort = 8792
    print("Training the model, this may take 15-30 minutes depending on data size, the server will start after")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

