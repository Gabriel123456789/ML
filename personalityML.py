## There is a process in the building of the model. First we have to import the sckit and other
## The flow of creating the model goes by: loading data, pre processing data, feature engineering, create bins/scales
## separate training and testing, scale the training valeus, tune the model, do a grid search for the best model
## and them evaluate the model with training data, accuracy and confusion matrix

import numpy as np
import pandas as pd

# Model selection works for creating training and testing data and tune the model
from sklearn.model_selection import train_test_split, GridSearchCV

# Scaler sets all numbers to a interval 0-1 and Encoder makes words binary
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Neighbors brings the model type to be used
from sklearn.neighbors import KNeighborsClassifier

# Metrics gives data of model accuracy
from sklearn.metrics import accuracy_score, confusion_matrix

# Impute is used to fill blank spaces using medians and knn algorithm
from sklearn.impute import SimpleImputer, KNNImputer

import matplotlib.pyplot as plt

## Load the data
data = pd.read_csv("personality_dataset.csv")
data.info()
print(data.isnull().sum())

###### Data Explained
#    - Time_spent_Alone: Hours spent alone daily (0–11). OK
#    - Stage_fear: Presence of stage fright (Yes/No). OK
#    - Social_event_attendance: Frequency of social events (0–10). OK
#    - Going_outside: Frequency of going outside (0–7). OK
#    - Drained_after_socializing: Feeling drained after socializing (Yes/No).
#    - Friends_circle_size: Number of close friends (0–15). OK
#    - Post_frequency: Social media post frequency (0–10). OK
#    - Personality: Target variable (Extrovert/Introvert).*

#How to relate them
# Presence of stage fright -> reduced social event attendance and reduced post frequency
# Reduced social attendance -> Increase hours spent alone, reduced number of friends, drained
# Time spent alone -> reduced going outside freq

def pre_processing(dataframe):
    


def fillig_blanks(dataframe):
    # Will firstly take yes/no columns into binary
    words_columns = ["Stage_fear", "Drained_after_socializing"]
    
    for column in words_columns:
        LE = LabelEncoder
        non_missing_rows = dataframe[column].notna()
        dataframe.loc[non_missing_rows,column] = LE.fit_transform(dataframe.loc[non_missing_rows, column])

    # Imputate data = Fill the blank spaces using simple imputer
    imputer_simple = SimpleImputer(strategy="most_frequent")
    dataframe[words_columns] = imputer_simple.fit_transform
