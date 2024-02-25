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
    args = request.args
    merchant = args.get('merchant')
    category = args.get('category')
    amt = float(args.get('amt'))
    first = args.get('first')
    last = args.get('last')
    sex = args.get('sex')
    lat = float(args.get('lat'))
    long = float(args.get('long'))
    city_pop = float(args.get('city_pop'))
    job = args.get('job')
    merch_lat = float(args.get('merch_lat'))
    merch_long = float(args.get('merch_long'))
    day_of_week = float(args.get('day_of_week'))
    day_of_month = float(args.get('day_of_month'))
    time = float(args.get('time'))
    generation = float(args.get('generation'))
    print([merchant, category, amt, first, last, sex, lat, long, city_pop, job, merch_lat, merch_long, day_of_week, day_of_month, time, generation])
    prediction = fraud_model.predict([merchant, category, amt, first, last, sex, lat, long, city_pop, job, merch_lat, merch_long, day_of_week, day_of_month, time, generation])
    if prediction == 1:
        return "Fraud"
    else: 
        return "Not Fraud"

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    data = json_normalize(json_data)
    prediction = fraud_model.json_predict(data)
    if prediction == 1:
        return "Fraud"
    else: 
        return "Not Fraud"

@app.route('/post', methods=['POST'])
def hellopost():
    args = request.args
    name = args.get('name')
    location = args.get('location')
    print("Name: ", name, " Location: ", location)
    new_transaction = request.files.get('transactions', '')
    print("new transaction data: ", new_transaction.filename)
    if not os.path.exists("transactions_backup.csv"):
        os.rename("transactions.csv", "transactions_backup.csv")
    new_transaction.save('/workspace/transactions.csv')
    fraud_model.train()
    return 'File Received - Thank you'

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

