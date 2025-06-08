## There is a process in the building of the model. First we have to import the sckit and others
## The flow of creating the model goes by a pipeline: First we have to import our data and split training and test
# then we create bins only in the training set, then map words to binary(yes/no). With that, there is no data leakage
# Now we can scale our columns fill the blank spaces. Now the data is ready to be fed in the model

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
#    - Time_spent_Alone: Hours spent alone daily (0–11).
#    - Stage_fear: Presence of stage fright (Yes/No). 
#    - Social_event_attendance: Frequency of social events (0–10).
#    - Going_outside: Frequency of going outside (0–7). 
#    - Drained_after_socializing: Feeling drained after socializing (Yes/No).
#    - Friends_circle_size: Number of close friends (0–15). 
#    - Post_frequency: Social media post frequency (0–10). 
#    - Personality: Target variable (Extrovert/Introvert).*


# Separate the data
X = data.drop(columns="Personality")
Y = data["Personality"]

X_training,X_testing,Y_training,Y_testing = train_test_split(X,Y, test_size = 0.3, random_state=42)

# Create bins using only the training data
bins_events = pd.qcut(X_training["Social_event_attendance"],5,labels=False, retbins=True, duplicates='drop')
bins_friends = pd.qcut(X_training["Friends_circle_size"],5,labels=False, retbins=True, duplicates='drop')

# Apply the bins in both data
X_training["Social_event_attendance"] = pd.cut(X_training["Social_event_attendance"], bins=bins_events, labels=False)
X_testing["Social_event_attendance"] = pd.cut(X_testing["Friends_circle_size"], bins=bins_friends, labels=False)


#Make yes/no become binary
X_training["Drained_after_socializing"] = X_training["Drained_after_socializing"].map({"Yes":1,"No":0})
X_testing["Drained_after_socializing"] = X_testing["Drained_after_socializing"].map({"Yes":1,"No":0})
X_training["Stage_fear"] = X_training["Stage_fear"].map({"Yes":1,"No":0})
X_testing["Stage_fear"] = X_testing["Stage_fear"].map({"Yes":1,"No":0})


# Scale the data before filling the blanks
scaler = MinMaxScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)


# Now we will use the knn algorithim to fill the columns with yes/no words
