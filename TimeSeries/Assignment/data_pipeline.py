import pandas as pd


# this is a template to get started, will continue to flesh out. Not in a finished state whatsoever.
class ETL_Pipeline:
    """
    ETL_Pipeline class for extracting, transforming, and loading data.
    """

    def __init__(self):
        """
        Initializes the ETL_Pipeline class.
        """
        self.data = None
        self.data_daily = None
        self.data_weekly = None
        self.data_monthly = None

    def extract(self, filename):
        """
        Extracts data from a .csv file.
        
        Parameters:
        filename (str): The filename of the .csv file.
        
        Returns:
        None
        """
        self.data = pd.read_csv(filename)

    def transform(self):
        """
        Transforms the extracted data by cleaning, processing, and preparing it for modeling.
        
        Returns:
        None
        """
        self.data_daily = self.day_transform()
        #self.data_weekly = self.week_transform()
        #self.data_monthly = self.month_transform()        

    def load(self):
        """
        Saves the transformed data to a .csv file.
        
        Parameters:
        output_filename (str): The filename of the output .csv file.
        
        Returns:
        None
        """
        self.data_daily.to_csv("daily.csv", index=True)
        #self.data_weekly.to_csv("weekly.csv", index=True)
        #self.data_monthly.to_csv("monthly.csv", index=True)
        
#    def week_transform(self):
#        self.data['trans_date'] = pd.to_datetime(self.data['trans_date'])
#        self.data['trans_week'] = self.data['trans_date'].dt.to_period('W').dt.strftime('Week %V')
#        self.data['transaction_count'] = self.data.groupby('trans_week').transform('size')
#        self.data['fraud_count'] = self.data['is_fraud'].groupby(self.data['trans_week']).transform('sum')
#        self.data['non_fraud_count'] = self.data['transaction_count'] - self.data['fraud_count']
#        result_df = self.data[['trans_week', 'transaction_count', 'fraud_count', 'non_fraud_count']]
#        result_df = result_df.sort_values(by='trans_week', ascending=True)
#        result_df = result_df.set_index("trans_week")
#        result_df = result_df.drop_duplicates()
#        return result_df

#    def month_transform(self):
#        self.data['transaction_month'] = self.data['trans_date'].dt.to_period('M').dt.strftime('%b %Y')
#        self.data['transaction_count'] = self.data.groupby('transaction_month').transform('size')
 #       self.data['fraud_count'] = self.data['is_fraud'].groupby(self.data['transaction_month']).transform('sum')
#        self.data['non_fraud_count'] = self.data['transaction_count'] - self.data['fraud_count']
#        result_df = self.data[['transaction_month', 'transaction_count', 'fraud_count', 'non_fraud_count']]
#        result_df = result_df.sort_values(by='transaction_month', ascending=True).reset_index(drop=True)
#        result_df = result_df.set_index("transaction_month")
#        result_df = result_df.drop_duplicates()
#        return result_df

    
    def day_transform(self):
        self.data['trans_date'] = pd.to_datetime(self.data['trans_date']).dt.strftime('%Y-%m-%d')
        self.data['transaction_count'] = self.data.groupby('trans_date').transform('size')
        self.data['fraud_count'] = self.data['is_fraud'].groupby(self.data['trans_date']).transform('sum')
        self.data['non_fraud_count'] = self.data['transaction_count'] - self.data['fraud_count']
        result_df = self.data[['trans_date', 'transaction_count', 'fraud_count', 'non_fraud_count']]
        result_df = result_df.sort_values(by='trans_date', ascending=True)
        result_df = result_df.drop_duplicates()
        result_df['trans_date'] = pd.to_datetime(result_df['trans_date']).dt.strftime('%m-%d')
        result_df = result_df.set_index("trans_date")
        return result_df

# Example usage
#etl_pipeline = ETL_Pipeline()
#etl_pipeline.extract('input_data.csv')
#etl_pipeline.transform()
#etl_pipeline.load('output_data.csv')