from flask import Flask
from flask import request
import os

from GraphicDataProcessing import ObjectDetection

app = Flask(__name__)


def convert_to_string(objects):
    detection_strings = []
    for i, obj in enumerate(objects):
        detection_strings.append(f"For test {i}: the object is:{obj['class_name']} with ({obj['confidence']}) level of confidence: in location {obj['box']} of the image")
    return "\n".join(detection_strings)
# Use postman to generate the post with a graphic of your choice

@app.route('/detect', methods=['POST'])
def detection():
    args = request.args
    name = args.get('name')
    location = args.get('description')

    print("Name: ", name, " Location: ", location)
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)
    imagefile.save('image.jpg')
    # The file is now downloaded and available to use with your detection class
    findings = ot.detect("image.jpg")
    # covert to useful string
    findingsString = convert_to_string(findings)
    return findingsString

if __name__ == "__main__":
    flaskPort = 8789
    ot = ObjectDetection()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

