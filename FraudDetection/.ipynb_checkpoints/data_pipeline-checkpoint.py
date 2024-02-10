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
        # Assuming the data has columns 'A', 'B', and 'C'
        # Cleaning and processing steps can be added here
        self.data['A'] = self.data['A'].fillna(0)
        self.data['B'] = self.data['B'].apply(lambda x: x.lower())
        self.data['C'] = self.data['C'].apply(lambda x: x.upper())

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