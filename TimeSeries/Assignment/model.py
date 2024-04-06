import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.impute import SimpleImputer
from data_pipeline import ETL_Pipeline
from datasets import TimeSeriesFraudDataset
from metrics import Metrics
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima                              # for determining ARIMA orders

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

class Fraud_Predictor_Deep_Model:
    """
    Fraud Detector Model Class:
    This class is responsible for constructing the model, handling the necessary logic to take raw input data,
    and produce an output.
    """

    def __init__(self, data_file, n_input=365, n_features=1, model=Sequential(), n_splits=5):
        """
        Initialize the Fraud Detector Model.
        :param model: The model to be used for fraud detection. Default is RandomForestClassifier.
        :param n_splits: The number of folds to split the dataset into for cross-validation. Default is 5.
        """
        model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.count_model_daily = model
        self.fraud_model_daily = model
#        self.count_model_weekly = model
#        self.fraud_model_weekly = model
#        self.count_model_monthly = model
#        self.fraud_model_monthly = model
        self.n_splits = n_splits
        self.data_file = data_file
        self.data_daily = None
        self.data_daily_count = None
        self.data_daily_fraud = None
#        self.data_weekly = None
#        self.data_monthly = None
        self.scaler_daily_count = None
        self.scaler_daily_fraud = None
#        self.scaler_weekly = None
#        self.scaler_monthly = None
        self.generator_daily_count = None
        self.generator_daily_fraud = None
#        self.generator_weekly = None
#        self.generator_monthly = None
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
#        self.data_weekly = pd.read_csv("weekly.csv", index_col="trans_week")
#        self.data_monthly = pd.read_csv("monthly.csv", index_col="transaction_month")
        self.initiate_scaler()


    def train(self, test=False):
        """
        Train the model using the provided training data.
        :param cross_validate: Determines if the model will simply train the model for use or run cross validation on it
        """
        self.initiate_etl()
        self.initiate_generator()
#        split_set_weekly = self.data_fold(self.data_weekly)
#        split_set_monthly = self.data_fold(self.data_monthly)
        if test:
            self.test(self.data_daily_count)
            self.test(self.data_daily_fraud)
#            self.cross_validation(split_set_weekly)
#            self.cross_validation(split_set_monthly)
        else:
            self.count_model_daily.fit_generator(self.generator_daily_count, epochs=10)
            self.fraud_model_daily.fit_generator(self.generator_daily_fraud, epochs=10)

    def test(self, metric_func=Metrics()):
        """
        Test the model using the provided testing data and calculate the metrics.
        :param X: The feature set or independent variables.
        :param y: The target variable or dependent variable.
        """
        y_pred = self.pipeline.predict(X)
        metric_func.run(self.data_daily[-n_input:], self.predictions)

    def create_predictions(self, info):
        """
        Takes the data to be predicted on and produces a prediction
        :param info: data to be used to get a predicion
        """
        test_predictions = []
        date_range = pd.date_range(start='1/1/2022', end='12/31/2022')
        # Format the index as month-day
        date_range = date_range.strftime('%m-%d')
        # Create an empty DataFrame with the date range as the index
        self.predictions = pd.DataFrame(index=date_range)
        first_eval_batch = self.data_daily_count[-n_input:]
        current_batch = first_eval_batch.reshape((1, self.n_input, self.n_features))
        # go beyond len(test) to go into the unknown future (no way to measure success except wait!)
        for i in range(len(scaled_test)):

            # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
            current_pred = self.count_model_daily.predict(current_batch)[0]

            # store prediction
            test_predictions.append(current_pred) 

            # update batch to now include prediction and drop first value
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        predictions = self.scaler_daily_count.inverse_transform(test_predictions)
        predictions = np.flip(predictions)
        self.predictions["count"] = predictions
        first_eval_batch = self.data_daily_fraud[-n_input:]
        current_batch = first_eval_batch.reshape((1, self.n_input, self.n_features))
        for i in range(len(scaled_test)):

            # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
            current_pred = self.fraud_model_daily.predict(current_batch)[0]

            # store prediction
            test_predictions.append(current_pred) 

            # update batch to now include prediction and drop first value
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
#        self.scaler_weekly = MinMaxScaler()
#        self.scaler_monthly = MinMaxScaler()
        temp = self.data_daily.drop(columns=["non_fraud_count", "fraud_count"])
        self.scaler_daily_count.fit(temp)
        self.data_daily_count = self.scaler_daily_count.transform(temp)
        temp = self.data_daily.drop(columns=["non_fraud_count", "transaction_count"])
        self.scaler_daily_fraud.fit(temp)
        self.data_daily_fraud =  self.scaler_daily_fraud.transform(temp)
#        self.data_weekly = pd.DataFrame(self.scaler_monthly.fit_transform(self.data_monthly))
#        self.data_daily = pd.DataFrame(self.scaler.fit_transform(self.data_daily))
        
    def initiate_generator(self):
        """
        Sets up the generators for the deep model to use for training
        """
        n_input = 365
        self.generator_daily_count = TimeseriesGenerator(self.data_daily_count, self.data_daily_count, length=n_input, batch_size=1)
        self.generator_daily_fraud = TimeseriesGenerator(self.data_daily_fraud, self.data_daily_fraud, length=n_input, batch_size=1)
#        n_input = 26
#        self.generator_weekly = TimeseriesGenerator(self.data_weekly, self.data_weekly, length=n_input, batch_size=1)
#        n_input = 12
#        self.generator_monthly = TimeseriesGenerator(self.data_monthly, self.data_monthly, length=n_input, batch_size=1)


class Fraud_Predictor_Traditional_Model:
    """
    Fraud Detector Model Class:
    This class is responsible for constructing the model, handling the necessary logic to take raw input data,
    and produce an output.
    """

    def __init__(self, data_file, n_input=365, n_features=1,n_splits=5):
        """
        Initialize the Fraud Detector Model.
        :param model: The model to be used for fraud detection. Default is RandomForestClassifier.
        :param n_splits: The number of folds to split the dataset into for cross-validation. Default is 5.
        """
        self.count_model_daily = None
        self.fraud_model_daily = None
#        self.count_model_weekly = model
#        self.fraud_model_weekly = model
#        self.count_model_monthly = model
#        self.fraud_model_monthly = model
        self.n_splits = n_splits
        self.data_file = data_file
        self.data_daily = None
        self.data_daily_count = None
        self.data_daily_fraud = None
#        self.data_weekly = None
#        self.data_monthly = None
        self.scaler_daily_count = None
        self.scaler_daily_fraud = None
#        self.scaler_weekly = None
#        self.scaler_monthly = None
        self.generator_daily_count = None
        self.generator_daily_fraud = None
#        self.generator_weekly = None
#        self.generator_monthly = None
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
#        self.data_weekly = pd.read_csv("weekly.csv", index_col="trans_week")
#        self.data_monthly = pd.read_csv("monthly.csv", index_col="transaction_month")
        self.initiate_scaler()


    def train(self, test=False):
        """
        Train the model using the provided training data.
        :param cross_validate: Determines if the model will simply train the model for use or run cross validation on it
        """
        self.initiate_etl()
        self.initiate_generator()
#        split_set_weekly = self.data_fold(self.data_weekly)
#        split_set_monthly = self.data_fold(self.data_monthly)
        if test:
            self.test()
            self.test(self.data_daily_fraud)
#            self.cross_validation(split_set_weekly)
#            self.cross_validation(split_set_monthly)
        else:
            self.count_model_daily.fit_generator(self.generator_daily_count, epochs=10)
            self.fraud_model_daily.fit_generator(self.generator_daily_fraud, epochs=10)

    def test(self, X, y, fold, iteration, metric_func=Metrics()):
        """
        Test the model using the provided testing data and calculate the metrics.
        :param X: The feature set or independent variables.
        :param y: The target variable or dependent variable.
        """
        y_pred = self.pipeline.predict(X)
        metric_func.run(self.data_daily[-n_input:], self.predictions)

    def create_predictions(self, info):
        """
        Takes the data to be predicted on and produces a prediction
        :param info: data to be used to get a predicion
        """
        test_predictions = []
        date_range = pd.date_range(start='1/1/2022', end='12/31/2022')
        # Format the index as month-day
        date_range = date_range.strftime('%m-%d')
        # Create an empty DataFrame with the date range as the index
        self.predictions = pd.DataFrame(index=date_range)
        first_eval_batch = self.data_daily_count[-n_input:]
        current_batch = first_eval_batch.reshape((1, self.n_input, self.n_features))
        # go beyond len(test) to go into the unknown future (no way to measure success except wait!)
        for i in range(len(scaled_test)):

            # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
            current_pred = self.count_model_daily.predict(current_batch)[0]

            # store prediction
            test_predictions.append(current_pred) 

            # update batch to now include prediction and drop first value
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        predictions = self.scaler_daily_count.inverse_transform(test_predictions)
        predictions = np.flip(predictions)
        self.predictions["count"] = predictions
        first_eval_batch = self.data_daily_fraud[-n_input:]
        current_batch = first_eval_batch.reshape((1, self.n_input, self.n_features))
        for i in range(len(scaled_test)):

            # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
            current_pred = self.fraud_model_daily.predict(current_batch)[0]

            # store prediction
            test_predictions.append(current_pred) 

            # update batch to now include prediction and drop first value
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        predictions = self.scaler_daily_fraud.inverse_transform(test_predictions)
        predictions = np.flip(predictions)
        self.predictions["fraud_count"] = predictions
