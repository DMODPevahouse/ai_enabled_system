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
        transformed_df = self.df3.copy()
        transformed_df = self.find_matches(transformed_df)
       # print(transformed_df)
       #  transformed_df = self.find_same_day_matches(transformed_df)
       # transformed_df.drop('Sent_Date', axis=1, inplace=True)
        self.data = transformed_df

    def load(self, path="transformed_data.csv"):
        """
        Load the transformed data to a csv file

        Args:
        path: str, the file path to save the transformed data
        
        """
        self.data.to_csv(path)

    def find_matches(self, transformed_df):
        """
        Find matches between the two dataframes based on Customer_ID and SubjectLine_ID

        Args:
        merged_df: DataFrame, the merged dataframe that takes the sent_emails and userbase dataframes
        df2: DataFrame, the responded dataframe

        """
        transformed_df['SubjectLine_ID'] = 1
        transformed_df['Match'] = 0
        transformed_df['Date_Match'] = 0
       # transformed_df['Sent_Date'] = ''
        #transformed_df['Sent_Date'] = pd.to_datetime(transformed_df['Sent_Date'])
        for index, row in self.df1.iterrows():
            customer_id = row['Customer_ID']
            subjectline_id = row['SubjectLine_ID']
            sent_date = row['Sent_Date']
            match = (self.df2['Customer_ID'] == customer_id) & (self.df2['SubjectLine_ID'] == subjectline_id)
            if customer_id in transformed_df.index and match.any() and transformed_df.at[
                customer_id, 'Date_Match'] == 0:
                transformed_df.at[customer_id, 'Match'] = 1
                transformed_df.at[customer_id, 'SubjectLine_ID'] = subjectline_id
                transformed_df.at[customer_id, 'Sent_Date'] = sent_date
                match_row = self.df2[
                    (self.df2['Customer_ID'] == customer_id) & (self.df2['SubjectLine_ID'] == subjectline_id)]
                responded_date = match_row['Responded_Date'].values[0]
                if pd.to_datetime(sent_date) == pd.to_datetime(responded_date):
                    transformed_df.at[index, 'Date_Match'] = 1
                    transformed_df.at[index, 'SubjectLine_ID'] = subjectline_id
        return transformed_df

#     def find_same_day_matches(self, transformed_df):
#         """
#         Find matches between the two dataframes based on Customer_ID, SubjectLine_ID, Responded_Date, and Sent_Date
#
#         Args:
#         merged_df: DataFrame, the merged dataframe that takes the sent_emails and userbase dataframes
#         df2: DataFrame, the responded dataframe
#
#         """
#         transformed_df['Date_Match'] = 0
#  #       transformed_df['Sent_Date'] = pd.to_datetime(transformed_df['Sent_Date'])
# #        self.df2['Responded_Date'] = pd.to_datetime(self.df2['Responded_Date'])
#
#         # Iterate over the rows of merged_df where Match is 1
#         for index, row in transformed_df[transformed_df['Match'] == 1].iterrows():
#             customer_id = row['Customer_ID']
#             subjectline_id = row['SubjectLine_ID']
#             sent_date = row['Sent_Date']
#             match_row = self.df2[(self.df2['Customer_ID'] == customer_id) & (self.df2['SubjectLine_ID'] == subjectline_id)]
#             responded_date = match_row['Responded_Date'].values[0]
#             if pd.to_datetime(sent_date) == pd.to_datetime(responded_date):
#                 transformed_df.at[index, 'Date_Match'] = 1
#                 transformed_df.at[index, 'SubjectLine_ID'] = subjectline_id
#         return transformed_df
