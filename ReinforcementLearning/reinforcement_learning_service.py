from flask import Flask, render_template
from flask import request
import os
import shutil
from model import QLearningAgent
import json
from pandas import json_normalize
import pickle


app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    stats = agent.play()
    return stats

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    data = json_normalize(json_data)
    prediction = agent.get_best_action(data.values[0])
    return f'The best email subject line to use is associated with the number {prediction[0]}'


@app.route('/train', methods=['GET'])
def full_train():
    from model import QLearningAgent
    from environment import EmailEnvironment
    import pandas as pd
    from data_pipeline import ETL_Pipeline
    etl = ETL_Pipeline()
    etl.extract()
    etl.transform()
    df = pd.read_csv("transformed_data.csv", index_col=0, low_memory=False)
    data = df.values  # Load your data here
    env = EmailEnvironment(data)

    # Create the Q-learning agent
    agent = QLearningAgent(env)

    # Train the agent
    agent.learn(num_episodes=100)

    # Play the agent
    stats = agent.play()
if __name__ == "__main__":
    flaskPort = 8793
    print("Importing model... /train will do a full training of a new model but takes an undetermined amount of time")
    with open('model.pkl', 'rb') as f:
        agent = pickle.load(f)
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

