from flask import Flask, render_template
from flask import request
import os
import shutil
from data_pipeline import LicensePlateETL
import json
from pandas import json_normalize
import model
import csv


app = Flask(__name__)


# http://localhost:8786/infer?transmission=automatic&color=blue&odometer=12000&year=2020&bodytype=suv&price=20000

@app.route('/stats', methods=['GET'])
def getStats():
    return "stats for the testing data will be in results/report.csv, there exists two others for comparison between tiny and normal weights"


@app.route('/predict_normal', methods=['GET'])
def predict_normal():
    file_path = 'answers_normal.csv'

    # Check if the file exists
    if os.path.isfile(file_path):
        # Delete the file
        os.remove(file_path)
    image_path = "output_normal"
    for img in os.listdir(image_path):
        if img.endswith('.jpeg') or img.endswith('.jpg'):
            image = os.path.join(image_path, img)
            text = model.ocr_license_plate(image)
            report = [
                ('Image name', 'License Plate number'),
                (image, text)
            ]
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(report)
    return "Predicted! Answers in answer_normal.csv"

@app.route('/predict_tiny', methods=['GET'])
def predict_tiny():
    file_path = 'answers_tiny.csv'

    # Check if the file exists
    if os.path.isfile(file_path):
        # Delete the file
        os.remove(file_path)
    image_path = "output_tiny"
    for img in os.listdir(image_path):
        if img.endswith('.jpeg') or img.endswith('.jpg'):
            image = os.path.join(image_path, img)
            text = model.ocr_license_plate(image)
            report = [
                ('Image name', 'License Plate number'),
                (image, text)
            ]
            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(report)
    return "Predicted! Answers in answer_tiny.csv"


if __name__ == "__main__":
    flaskPort = 8790
    etl = LicensePlateETL()
    print("awaiting video")
    in_file = 'udp://127.0.0.1:23000'  # Example UDP input URL
    width = 3840  # Example width
    height = 2160  # Example height
    frames_per_second = 1
    etl.extract(in_file, width, height, frames_per_second)
    print("Video recieved! Reading in video, could take up to 15 minutes, server will start after. ")
    etl = LicensePlateETL(output_directory="output_normal")
    etl.transform('lpr-yolov3.weights', 'lpr-yolov3.cfg')
    etl = LicensePlateETL(output_directory="output_tiny")
    etl.transform('lpr-yolov3-tiny.weights', 'lpr-yolov3-tiny.cfg')
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

