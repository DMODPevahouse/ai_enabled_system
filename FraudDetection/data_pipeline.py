import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

        to_export_df = self.data.copy()
        to_export_df['sex'] = to_export_df['sex'].astype("category").cat.codes
        to_export_df['first'] = to_export_df['first'].astype("category").cat.codes.astype('float')
        to_export_df['last'] = to_export_df['last'].astype("category").cat.codes.astype('float')
        to_export_df['category'] = to_export_df['category'].astype("category").cat.codes.astype('float')
        to_export_df['job'] = to_export_df['job'].astype("category").cat.codes.astype('float')
        to_export_df['merchant'] = to_export_df['merchant'].astype("category").cat.codes.astype('float')
        to_export_df['trans_date_trans_time'] = pd.to_datetime(to_export_df['trans_date_trans_time'])
        to_export_df['day_of_week'] = to_export_df['trans_date_trans_time'].dt.day_name()
        to_export_df['day_of_month'] = to_export_df['trans_date_trans_time'].dt.day
        to_export_df['time'] = to_export_df['trans_date_trans_time'].dt.time
        to_export_df.drop(columns='cc_num', inplace=True)
        to_export_df.drop(columns='unix_time', inplace=True)
        to_export_df.drop(columns='city', inplace=True)
        to_export_df.drop(columns='state', inplace=True)
        to_export_df.drop(columns='Unnamed: 0', inplace=True)
        to_export_df.drop(columns='zip', inplace=True)
        to_export_df.drop(columns='street', inplace=True)
        to_export_df.drop(columns='trans_num', inplace=True)
        
        bins = [1883, 1901, 1928, 1946, 1965, 1981, 1997, 2013, 2024]

        to_export_df['dob'] = to_export_df['dob'].str.slice(0,4).astype(int)
        to_export_df['generation'] = pd.cut(to_export_df['dob'], bins=bins)
        to_export_df.drop(columns='dob', inplace=True)
        to_export_df['time'] = to_export_df['trans_date_trans_time'].apply(lambda x: to_float(x.time()))
        to_export_df['generation'] = to_export_df['generation'].astype("category").cat.codes.astype('float')
        to_export_df['day_of_week'] = to_export_df['day_of_week'].astype("category").cat.codes.astype('float')
        to_export_df.drop(columns='trans_date_trans_time', inplace=True)

        scaler = MinMaxScaler()
        to_export_df = pd.DataFrame(scaler.fit_transform(to_export_df), columns=to_export_df.columns)
        self.data = to_export_df
        

    def load(self, output_filename):
        """
        Saves the transformed data to a .csv file.
        
        Parameters:
        output_filename (str): The filename of the output .csv file.
        
        Returns:
        None
        """
        self.data.to_csv(output_filename, index=False)
        

def to_float(time):
    hours, minutes = time.hour, time.minute
    return hours + minutes / 60

# Example usage
#etl_pipeline = ETL_Pipeline()
#etl_pipeline.extract('input_data.csv')
#etl_pipeline.transform()
#etl_pipeline.load('output_data.csv')