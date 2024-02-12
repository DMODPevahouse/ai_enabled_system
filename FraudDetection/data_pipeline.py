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
        # Add your data cleaning and processing code here
        # For example, you might want to remove unnecessary columns, handle missing values,
        # convert data types, etc.

        # Here's an example of removing unnecessary columns:
        self.data.drop(columns=['unnecessary_column_1', 'unnecessary_column_2'], inplace=True)

        # Here's an example of handling missing values:
        self.data.fillna(value='missing', inplace=True)

        # Here's an example of converting data types:
        self.data['date_column'] = pd.to_datetime(data['date_column'])

    def load(self, output_filename):
        """
        Saves the transformed data to a .csv file.
        
        Parameters:
        output_filename (str): The filename of the output .csv file.
        
        Returns:
        None
        """
        self.data.to_csv(output_filename, index=False)

# Example usage
#etl_pipeline = ETL_Pipeline()
#etl_pipeline.extract('input_data.csv')
#etl_pipeline.transform()
#etl_pipeline.load('output_data.csv')