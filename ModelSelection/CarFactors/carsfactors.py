# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class carsfactors:
    def __init__(self):
        self.modelLearn = False
        self.stats = 0

    def model_learn(self):
        # Importing the dataset into a pandas dataframe
        df = pd.read_csv("cars.csv")
        # note your selected features to address the concern.  Select "useful" columns.  You do not need many.
        useful_columns = ['transmission', 'color', 'odometer_value', 'year_produced', 'body_type', 'price_usd', 'duration_listed']
        df = df[useful_columns]
        # Remove Unwanted Columns
        df.dropna(inplace=True)
        # Seperate X and y (features and label)  The last feature "duration_listed" is the label (y)
        # Seperate X vs Y
        X = df.drop('duration_listed', axis=1)
        y = df['duration_listed']
        # Do the ordinal Encoder for car type to reflect that some cars are bigger than others.
        # This is the order 'universal','hatchback', 'cabriolet','coupe','sedan','liftback', 'suv', 'minivan', 'van','pickup', 'minibus','limousine'
        # make sure this is the entire set by using unique()
        # create a seperate dataframe for the ordinal number - so you must strip it out and save the column
        # make sure to save the OrdinalEncoder for future encoding due to inference
        car_type_order = ['universal', 'hatchback', 'cabriolet', 'coupe', 'sedan', 'liftback', 'suv', 'minivan', 'van', 'pickup', 'minibus', 'limousine']
        self.car_type_encoder = OrdinalEncoder(categories=[car_type_order], handle_unknown='use_encoded_value', unknown_value=-1)
        X['car_type'] = self.car_type_encoder.fit_transform(X[['body_type']])

        # Do onehotencoder the selected features - again you need to make a new dataframe with just the encoding of the transmission
        # save the OneHotEncoder to use for future encoding of transmission due to inference
        self.transmission_encoder = OneHotEncoder(sparse_output=False)
        self.X_transmission = pd.DataFrame(self.transmission_encoder.fit_transform(X[['transmission']]), columns=['transmission_' + col for col in self.transmission_encoder.categories_[0]])

        # Do onehotencoder for Color
        # Save the OneHotEncoder to use for future encoding of color for inference
        self.color_encoder = OneHotEncoder(sparse_output=False)
        X_color = pd.DataFrame(self.color_encoder.fit_transform(X[['color']]), columns=['color_' + col for col in self.color_encoder.categories_[0]])

        # combine all three together endocdings into 1 data frame (need 2 steps with "concatenate")
        # add the ordinal and transmission then add color
        preprocessor = ColumnTransformer([
            ('body_type', self.car_type_encoder, ['body_type']),
            ('transmission', self.transmission_encoder, ['transmission']),
            ('color', self.color_encoder, ['color'])
        ])
        X_encoded = preprocessor.fit_transform(X)
        # then dd to original data set
        columns_df = df[['odometer_value', 'year_produced', 'price_usd']]
        columns_array = columns_df.values
        X_encoded = np.concatenate((X_encoded, columns_array), axis=1)
        # delete the columns that are substituted by ordinal and onehot - delete the text columns for color, transmission, and car type

        # Splitting the dataset into the Training set and Test set

        # Feature Scaling - required due to different orders of magnitude across the features
        # make sure to save the scaler for future use in inference
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_encoded)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # Select useful model to deal with regression (it is not categorical for the number of days can vary quite a bit)
        # Train the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Model statistics
        self.stats = r2_score(y_test, self.model.predict(X_test))
        self.modelLearn = True

    # this demonstrates how you have to convert these values using the encoders and scalers above (if you choose these columns - you are free to choose any you like)
    def model_infer(self, transmission, color, odometer, year, bodytype, price):
        if (self.modelLearn == False):
            self.model_learn()

        # convert the body type into a numpy array that holds the correct encoding
        carTypeTest = self.car_type_encoder.transform([[bodytype]])

        # Convert the transmission into a numpy array with the correct encoding
        carHotTransmissionTest = self.transmission_encoder.transform([[transmission]])

        # Convert the color into a numpy array with the correct encoding
        carHotColorTest = self.color_encoder.transform([[color]])

        # Add the three above
        total = np.concatenate((carTypeTest, carHotTransmissionTest), 1)
        total = np.concatenate((total, carHotColorTest), 1)
        # Build a complete test array and then predict
        
        othercolumns = np.array([[odometer, year, price]])
        totaltotal = np.concatenate((total, othercolumns), 1)
        # Scale the features
        attempt = self.scaler.transform(totaltotal)

        # Determine prediction
        y_pred = self.model.predict(attempt)
        return str(y_pred[0])


    def model_stats(self):
        if (self.modelLearn == False):
            self.model_learn()
        return str(self.stats)