from flask import Flask, render_template
from flask import request
import os
import shutil
from model import Fraud_Predictor_Traditional_Model, Fraud_Predictor_Deep_Model
import json
from pandas import json_normalize



app = Flask(__name__)


# http://localhost:8786/infer?transmission=automatic&color=blue&odometer=12000&year=2020&bodytype=suv&price=20000

@app.route('/stats', methods=['GET'])
def getStats():
    return "Results in results/deep_report.csv and results/trad_report.csv"

@app.route('/infer', methods=['GET'])
def getInfer():
    args = request.args
    date = args.get('mm-dd')
    answer_count_deep, answer_fraud_deep = deep.predict(date)
    answer_count_trad, answer_fraud_trad = trad.predict(date)
    return f'The deep model reported: total={answer_count_deep} and fraud={answer_fraud_deep}, The traditional model reported: total={answer_count_trad} and fraud={answer_fraud_trad}'

@app.route('/test', methods=['GET'])
def test():
    deep.train(True)
    trad.train(True)
    return "Results in results/deep_report.csv and results/trad_report.csv"

if __name__ == "__main__":
    flaskPort = 8791
    deep = Fraud_Predictor_Deep_Model("CreditCardFraudFourYears.csv")
    trad = Fraud_Predictor_Traditional_Model("CreditCardFraudFourYears.csv")
    print("Training the model, this may take 15-30 minutes depending on data size, the server will start after")
    deep.train()
    trad.train()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

