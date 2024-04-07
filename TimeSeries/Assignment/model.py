import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.impute import SimpleImputer
from data_pipeline import ETL_Pipeline
from metrics import Metrics
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose     
from pmdarima import auto_arima                              

class Fraud_Predictor_Deep_Model:
    """
    This class is used to create a deep learning model for predicting fraud in a given dataset.
    It uses a Long Short-Term Memory (LSTM) network to model the time series data and predict
    the probability of fraud in future time steps.
    """

    def __init__(self, data_file, n_input=365, n_features=1, model=Sequential(), n_splits=5):
        """
        Initializes the Fraud_Predictor_Deep_Model class with the given parameters.

        Args:
            data_file (str): The path to the input data file.
            n_input (int): The number of time steps to use as input to the LSTM network. Default is 365.
            n_features (int): The number of features in the input data. Default is 1.
            model (Sequential): The LSTM model to use for predicting fraud. Default is a new Sequential model.
            n_splits (int): The number of splits to use in cross-validation. Default is 5.
        """
        model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.n_features = n_features
        self.n_input = n_input
        self.count_model_daily = model
        self.fraud_model_daily = model
        self.n_splits = n_splits
        self.data_file = data_file
        self.data_daily = None
        self.data_daily_count = None
        self.data_daily_fraud = None
        self.scaler_daily_count = None
        self.scaler_daily_fraud = None
        self.generator_daily_count = None
        self.generator_daily_fraud = None
        self.predictions = None
        
    def initiate_etl(self):
        """
        Initialize and run the ETL pipeline, loading the transformed data into a DataFrame.
        """
        etl = ETL_Pipeline()
        etl.extract(self.data_file)
        etl.transform()
        etl.load()
        self.data_daily = pd.read_csv("daily.csv", index_col="trans_date")
        self.initiate_scaler()


    def train(self, test=False):
        """
        Train the model using the provided training data.
        :param test: Determines if the model will simply train the model for use or run cross validation on it
        """
        self.initiate_etl()
        self.initiate_generator()
        if test:
            self.test()
        else:
            self.count_model_daily.fit_generator(self.generator_daily_count, epochs=1)
            self.fraud_model_daily.fit_generator(self.generator_daily_fraud, epochs=1)
            self.create_predictions()

    def test(self, metric_func=Metrics("deep_report")):
        """
        Test the model using the provided testing data and calculate the metrics.
        """
        self.create_predictions()
        metric_func.run(self.predictions, self.data_daily[-self.n_input:])

    def create_predictions(self):
        """
        Uses the trained LSTM network to predict the number of transactions and the number of fraud cases for each day

        This function first initializes a list to store the predictions. It then gets the last n_input days of data from the input data and reshapes it into a batch of size (1, n_input, n_features) to use as the initial input to the LSTM network.
        """
        test_predictions = []
        date_range = pd.date_range(start='1/1/2022', end='12/31/2022') #year is arbitraury, just looking to get 365 days
        date_range = date_range.strftime('%m-%d')
        self.predictions = pd.DataFrame(index=date_range)
        first_eval_batch = self.data_daily_count[-self.n_input:]
        current_batch = first_eval_batch.reshape((1, self.n_input, self.n_features))
        
        for i in range(self.n_input):
            current_pred = self.count_model_daily.predict(current_batch)[0]
            test_predictions.append(current_pred) 
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        
        predictions = self.scaler_daily_count.inverse_transform(test_predictions)
        predictions = np.flip(predictions)
        self.predictions["transaction_count"] = predictions
        first_eval_batch = self.data_daily_fraud[-self.n_input:]
        current_batch = first_eval_batch.reshape((1, self.n_input, self.n_features))
        test_predictions = []
        for i in range(self.n_input):
            current_pred = self.fraud_model_daily.predict(current_batch)[0]
            test_predictions.append(current_pred) 
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        
        predictions = self.scaler_daily_fraud.inverse_transform(test_predictions)
        predictions = np.flip(predictions)
        self.predictions["fraud_count"] = predictions
        
    def initiate_scaler(self):
        """
        Sets up the scaler to be used in the model so that the data that will be predicted on for new data will be scaled to the same level that the model was scaled at to prevent any data skew
        """
        self.scaler_daily_count = MinMaxScaler()
        self.scaler_daily_fraud = MinMaxScaler()
        temp = self.data_daily.drop(columns=["non_fraud_count", "fraud_count"])
        self.scaler_daily_count.fit(temp)
        self.data_daily_count = self.scaler_daily_count.transform(temp)
        temp = self.data_daily.drop(columns=["non_fraud_count", "transaction_count"])
        self.scaler_daily_fraud.fit(temp)
        self.data_daily_fraud =  self.scaler_daily_fraud.transform(temp)
        
    def initiate_generator(self):
        """
        Sets up the generators for the deep model to use for training
        """
        n_input = 365
        self.generator_daily_count = TimeseriesGenerator(self.data_daily_count, self.data_daily_count, length=n_input, batch_size=1)
        self.generator_daily_fraud = TimeseriesGenerator(self.data_daily_fraud, self.data_daily_fraud, length=n_input, batch_size=1)
   
    def predict(self, value):
        """
        Returns the predicted number of transactions and the predicted number of fraud cases for the given date.

        This function takes a date as input and returns the predicted number of transactions and the predicted number of fraud cases for that date. The predictions are obtained from the predictions DataFrame, which was created by the create_predictions function.

        Args:
            value (str): The date for which to get the predictions. The date should be in the format 'MM-DD'.

        Returns:
            tuple: A tuple containing the predicted number of transactions and the predicted number of fraud cases for the given date.
        """
        pred_count = self.predictions.loc[value, 'transaction_count']
        pred_fraud = self.predictions.loc[value, 'fraud_count']
        return pred_count, pred_fraud



class Fraud_Predictor_Traditional_Model:
    """
    This class is used to create a traditional learning model for predicting fraud in a given dataset.
    It uses a SARIMAX to model the time series data and predict
    the probability of fraud in future time steps.
    """
    def __init__(self, data_file, n_input=365, n_features=1,n_splits=5):
        """
        This code defines a class `Fraud_Predictor_Traditional_Model` with an initializer that takes in four parameters:
        - `data_file`: The file containing the data for the model to train on.
        - `n_input`: The number of time steps to use as input for the model. Default is 365.
        - `n_features`: The number of features for each time step. Default is 1.
        - `n_splits`: The number of splits to use for cross-validation. Default is 5.
        """
        self.count_model_daily = None
        self.fraud_model_daily = None
        self.n_input = n_input
        self.n_splits = n_splits
        self.data_file = data_file
        self.data_daily = None
        self.data_daily_count = None
        self.data_daily_fraud = None
        self.predictions = None
        
    def initiate_etl(self):
        """
        Initialize and run the ETL pipeline, loading the transformed data into a DataFrame.
        """
        etl = ETL_Pipeline()
        etl.extract(self.data_file)
        etl.transform()
        etl.load()
        self.data_daily = pd.read_csv("daily.csv", index_col="trans_date")

    def train(self, test=False):
        """
        Train the model using the provided training data.
        :param cross_validate: Determines if the model will simply train the model for use or run cross validation on it
        """
        self.initiate_etl()
        if test:
            self.test()
        else:
            self.create_predictions()
            
    def test(self, metric_func=Metrics("trad_report")):
        """
        Test the model using the provided testing data and calculate the metrics.
        """
        self.create_predictions()
        metric_func.run(self.predictions, self.data_daily[-self.n_input:])

    def create_predictions(self):
        date_range, fraud_list, count_list = pd.date_range(start='1/1/2022', end='12/31/2022'), [], []
        date_range = date_range.strftime('%m-%d')
        self.predictions = pd.DataFrame(index=date_range)
        train = self.data_daily.iloc[:self.n_input]
        test = self.data_daily.iloc[-self.n_input:]
        self.count_model_daily = SARIMAX(train['transaction_count'],order=(0,1,0),seasonal_order=(1,0,[1,2],7),enforce_invertibility=False)
        self.fraud_model_daily = SARIMAX(train['fraud_count'],order=(0,0,1),seasonal_order=(1,0,1,7),enforce_invertibility=False)
        results_count = self.count_model_daily.fit()
        results_fraud = self.fraud_model_daily.fit()
        start=len(train)
        end=len(train)+len(test)-1
        predictions_count = results_count.predict(start=start, end=end, dynamic=False).rename('Predictions')
        
        for i in predictions_count:
            count_list.append(i)
        
        predictions_fraud = results_fraud.predict(start=start, end=end, dynamic=False).rename('Predictions')
        
        for i in predictions_fraud:
            fraud_list.append(i)
        
        self.predictions["transaction_count"] = count_list
        self.predictions["fraud_count"] = fraud_list
        
    def predict(self, value):
        """
        Returns the predicted number of transactions and the predicted number of fraud cases for the given date.

        This function takes a date as input and returns the predicted number of transactions and the predicted number of fraud cases for that date. The predictions are obtained from the predictions DataFrame, which was created by the create_predictions function.

        Args:
            value (str): The date for which to get the predictions. The date should be in the format 'MM-DD'.

        Returns:
            tuple: A tuple containing the predicted number of transactions and the predicted number of fraud cases for the given date.
        """
        pred_count = self.predictions.loc[value, 'transaction_count']
        pred_fraud = self.predictions.loc[value, 'fraud_count']
        return pred_count, pred_fraud
