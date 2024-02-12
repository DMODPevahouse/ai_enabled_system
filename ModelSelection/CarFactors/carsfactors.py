# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class carsfactors:
    def __init__(self):
        self.modelLearn = False
        self.stats = 0

    def model_learn(self):
        # Importing the dataset into a pandas dataframe
        
        # note your selected features to address the concern.  Select "useful" columns.  You do not need many.

        #Remove Unwanted Columns
 
        # Seperate X and y (features and label)  The last feature "duration_listed" is the label (y)
        # Seperate X vs Y
 
        # Do the ordinal Encoder for car type to reflect that some cars are bigger than others.  
        # This is the order 'universal','hatchback', 'cabriolet','coupe','sedan','liftback', 'suv', 'minivan', 'van','pickup', 'minibus','limousine'
        # make sure this is the entire set by using unique()
        # create a seperate dataframe for the ordinal number - so you must strip it out and save the column
        # make sure to save the OrdinalEncoder for future encoding due to inference
    

        # Do onehotencoder the selected features - again you need to make a new dataframe with just the encoding of the transmission
        # save the OneHotEncoder to use for future encoding of transmission due to inference
 
        # Do onehotencoder for Color
        # Save the OneHotEncoder to use for future encoding of color for inference

        # combine all three together endocdings into 1 data frame (need 2 steps with "concatenate")
        # add the ordinal and transmission then add color

        # then dd to original data set
        
        #delete the columns that are substituted by ordinal and onehot - delete the text columns for color, transmission, and car type 

        # Splitting the dataset into the Training set and Test set 

        # Feature Scaling - required due to different orders of magnitude across the features
        # make sure to save the scaler for future use in inference

        # Select useful model to deal with regression (it is not categorical for the number of days can vary quite a bit)
        from sklearn.linear_model import 
        self.model = 
        self.model.fit(X_train, y_train)
        
        self.stats = self.model.score(X_train, y_train)
        self.modelLearn = True

    # this demonstrates how you have to convert these values using the encoders and scalers above (if you choose these columns - you are free to choose any you like)
    def model_infer(self,transmission, color, odometer, year, bodytype, price):
         if(self.modelLearn == False):
            self.model_learn()

        #convert the body type into a numpy array that holds the correct encoding
        carTypeTest = 
        carTypeTest = 
 
        #convert the transmission into a numpy array with the correct encoding
        carHotTransmissionTest = 
        carHotTransmissionTest = 
        
        #conver the color into a numpy array with the correct encoding
        carHotColorTest = 
        carHotColorTest = 

        #add the three above
        total = np.concatenate((carTypeTest,carHotTransmissionTest), 1)
        total = np.concatenate((total,carHotColorTest), 1)
        
        # build a complete test array and then predict
        othercolumns = np.array([[odometer ,year, price]])
        totaltotal = np.concatenate((total, othercolumns),1)

        #must scale
        attempt = 
        
        #determine prediction
        y_pred = 
        return str(y_pred)
        
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)
