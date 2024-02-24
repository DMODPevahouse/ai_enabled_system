from flask import Flask
from flask import request
import os
from model import Models
import shutil
from models import Fraud_Detector_Model
import json

app = Flask(__name__)

# http://localhost:8786/infer?transmission=automatic&color=blue&odometer=12000&year=2020&bodytype=suv&price=20000

@app.route('/stats', methods=['GET'])
def getStats():
    return str(cf.model_stats())

@app.route('/infer', methods=['GET'])
def getInfer():
    args = request.args
    transmission = args.get('transmission')
    color = args.get('color')
    odometer = int(args.get('odometer'))
    year = int(args.get('year'))
    bodytype = args.get('bodytype')
    price = int(args.get('price'))
    return cf.model_infer(transmission, color, odometer, year, bodytype, price)

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    data = json.loads(json_data['data'])
    prediction = fraud_model.predict(data)
    result = {'prediction': prediction.tolist()}
    return jsonify(result)

@app.route('/post', methods=['POST'])
def hellopost():
    args = request.args
    name = args.get('name')
    location = args.get('location')
    print("Name: ", name, " Location: ", location)
    new_transaction = request.files.get('custom transactions data', '')
    print("new transaction data: ", new_transaction.filename)
    if not os.path.exists("transactions_backup.csv):
        os.rename("transactions.csv", "transactions_backup.csv")
    new_transaction.save('/workspace/transactions.csv')
    return 'File Received - Thank you'

@app.route('/backup', methods=['POST'])
def hellopost():
    if os.path.exists("transactions.csv") and os.path.exists("transactions_backup.csv"):
            os.remove("transactions.csv")
    shutil.copy2("transactions_backup.csv", "transactions.csv")

@app.route('/crossvalidate', methods=['POST'])
def hellopost():
    fraud_model.train(True)
    return "Results in results/report.csv
                      
if __name__ == "__main__":
    flaskPort = 8900
    fraud_model = Fraud_Detector_Model("transactions.csv")
    print("Training the model, this may take 15-30 minutes depending on data size, the server will start after")
    fraud_model.train()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

