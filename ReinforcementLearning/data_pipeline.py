import pandas as pd
import numpy as np
import os

class ETL_Pipeline:
    """
    Extract, transform, and load data from three different csv email data files

    Attributes:
    data: DataFrame, the transformed data
    df1: DataFrame, the sent_emails data
    df2: DataFrame, the responded data
    df3: DataFrame, the userbase data
    """
    def __init__(self):
        """
        Initialize the ETL_Pipeline object
        """
        self.data = None
        self.df1 = None
        self.df2 = None
        self.df3 = None

    def extract(self, sent="sent_emails.csv", received="responded.csv", email_data="userbase.csv"):
        """
        Extract data from three different csv files

        Args:
        sent: str, the file path to the sent_emails csv file
        received: str, the file path to the responded csv file
        email_data: str, the file path to the userbase csv file
        """
        self.df1, self.df2, self.df3 = pd.read_csv(sent), pd.read_csv(received), pd.read_csv(email_data)

    def transform(self):
        """
        Transform the data by merging the three dataframes and finding matches between the dataframes
        """
        merged_df = pd.merge(self.df1, self.df3, on='Customer_ID', how='left')
        merged_df = find_matches(merged_df, self.df2)
        merged_df = find_same_day_matches(merged_df, self.df2)
        merged_df.drop(columns=['Sent_Date'], inplace=True)
        self.data = merged_df

    def load(self, path="transformed_data.csv"):
        """
        Load the transformed data to a csv file

        Args:
        path: str, the file path to save the transformed data
        
        """
        self.data.to_csv(path)


def find_matches(merged_df, df2):
    """
    Find matches between the two dataframes based on Customer_ID and SubjectLine_ID

    Args:
    merged_df: DataFrame, the merged dataframe that takes the sent_emails and userbase dataframes
    df2: DataFrame, the responded dataframe

    """
    merged_df['Match'] = 0
    for index, row in merged_df.iterrows():
        customer_id = row['Customer_ID']
        subjectline_id = row['SubjectLine_ID']
        match = (df2['Customer_ID'] == customer_id) & (df2['SubjectLine_ID'] == subjectline_id)
        if match.any():
            merged_df.at[index, 'Match'] = 1
    return merged_df


def find_same_day_matches(merged_df, df2):
    """
    Find matches between the two dataframes based on Customer_ID, SubjectLine_ID, Responded_Date, and Sent_Date

    Args:
    merged_df: DataFrame, the merged dataframe that takes the sent_emails and userbase dataframes
    df2: DataFrame, the responded dataframe

    """
    merged_df['Date_Match'] = 0
    for index, row in merged_df[merged_df['Match'] == 1].iterrows():
        customer_id = row['Customer_ID']
        subjectline_id = row['SubjectLine_ID']
        sent_date = row['Sent_Date']
        match_row = df2[(df2['Customer_ID'] == customer_id) & (df2['SubjectLine_ID'] == subjectline_id)]
        responded_date = match_row['Responded_Date'].values[0]
        if pd.to_datetime(sent_date) == pd.to_datetime(responded_date):
            merged_df.at[index, 'Date_Match'] = 1

    return merged_df