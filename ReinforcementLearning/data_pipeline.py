import pandas as pd
import numpy as np
import os

class ETL_Pipeline:
    def __init__(self):
        self.data = None

    def extract(self, sent="sent_emails.csv", received="responded.csv", email_data="userbase.csv"):
        df1, df2, df3 = pd.read_csv(sent), pd.read_csv(received), pd.read_csv(email_data)
        merged_df = df1.merge(df2, on=['Customer_ID', 'SubjectLine_ID'])
        merged_df = merged_df.merge(df3, on='Customer_ID')
        self.data = merged_df

    def transform(self):
        self.data['one_day_response'] = self.data.apply(one_day_response, axis=1)
        self.data = self.data.drop(columns=['Sent_Date', 'Responded_Date', 'Customer_ID'])

    def load(self, path="transformed_data.csv"):
        self.data.to_csv(path)


def one_day_response(row):
    if row['Sent_Date'] == row['Responded_Date']:
        return 1
    else:
        return 0