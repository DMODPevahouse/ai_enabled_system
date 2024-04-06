from flask import Flask, render_template
from flask import request
import os
import shutil
from model import Fraud_Detector_Model
import json
from pandas import json_normalize



app = Flask(__name__)


# http://localhost:8786/infer?transmission=automatic&color=blue&odometer=12000&year=2020&bodytype=suv&price=20000

@app.route('/stats', methods=['GET'])
def getStats():
    return "run /crossvalidate, then the results will be in results/report.csv"

@app.route('/infer', methods=['GET'])
def getInfer():
    

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    data = json_normalize(json_data)
    prediction = fraud_model.json_predict(data)
    if prediction == 1:
        return "Fraud"
    else: 
        return "Not Fraud"

@app.route('/backup', methods=['POST'])
def create_backup():
    if os.path.exists("transactions.csv") and os.path.exists("transactions_backup.csv"):
            os.remove("transactions.csv")
    shutil.copy2("transactions_backup.csv", "transactions.csv")
    return "backup created"

@app.route('/crossvalidate', methods=['GET'])
def cross_validate_post():
    fraud_model.train(True)
    return "Results in results/report.csv"

if __name__ == "__main__":
    flaskPort = 8788
    fraud_model = Fraud_Detector_Model("transactions.csv")
    print("Training the model, this may take 15-30 minutes depending on data size, the server will start after")
    fraud_model.train()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

