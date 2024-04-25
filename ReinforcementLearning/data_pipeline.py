import pandas as pd
import numpy as np
import os

class ETL_Pipeline:
    def __init__(self):
        self.data = None
        self.sent = None
        self.recieved = None
        self.email_data = None

    def extract(self, sent="sent_emails.csv", recieved="recieved_emails.csv", email_data="userbase.csv"):
        self.sent = pd.read_csv(sent)
        self.recieved = pd.read_csv(recieved)
        self.email_data = pd.read_csv(email_data)

    def transform(self):
        export_data = pd.concat([self.sent, self.recieved, self.email_data], axis=0)
        export_data['one_day_response'] = export_data.apply(one_day_response, axis=1)

        self.data = export_data

    def load(self, path="transformed_data.csv"):
        self.data.to_csv(path, index=False)


def one_day_response(row):
    if row['Sent_Date'] == row['Responded_Date']:
        return 1
    else:
        return 0