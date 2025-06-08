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
_, bins_events = pd.qcut(X_training["Social_event_attendance"],5,labels=False, retbins=True, duplicates='drop')
_, bins_friends = pd.qcut(X_training["Friends_circle_size"],3,labels=False, retbins=True, duplicates='drop')

# While testing there was a problem with bins values, so this was added
bins_events_unicos = np.unique(bins_events)
bins_friends_unicos = np.unique(bins_friends)

# Apply the bins in both data
X_training["Social_event_attendance"] = pd.cut(X_training["Social_event_attendance"], bins=bins_events_unicos, labels=False, include_lowest=True)
X_testing["Social_event_attendance"] = pd.cut(X_testing["Social_event_attendance"], bins=bins_events_unicos, labels=False, include_lowest=True)
X_training["Friends_circle_size"] = pd.cut(X_training["Friends_circle_size"], bins=bins_friends_unicos, labels=False,include_lowest=True)
X_testing["Friends_circle_size"] = pd.cut(X_testing["Friends_circle_size"], bins=bins_friends_unicos, labels=False,include_lowest=True)



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
# The knn imputer will analyze all the data from the near neighbors to fill the blanks
imputer = KNNImputer(n_neighbors=5)

X_training_clean = imputer.fit_transform(X_training)
X_testing_clean = imputer.transform(X_testing)

## The pipeline is FINISHED
# Now the data is clean and we can start tuning our model

def model_tuner(x_training, y_training):
    # Define the parameters to be used
    param_grid = {
        "n_neighbors": range(1,21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]    
    }
    model = KNeighborsClassifier()
    
    ## Use the grid search to find the best model
    grid = GridSearchCV(model,param_grid, cv = 5, n_jobs= -1)
    grid.fit(x_training,y_training)
    return grid.best_estimator_

best_model_grid = model_tuner(X_training_clean, Y_training)

# Create a accuracy tester - Evaluate if the model is good

def evaluate_model(model,x_testing,y_testing):
    prediction = model.predict(x_testing)
    accuracy = accuracy_score(y_testing, prediction)
    matrix_confusion = confusion_matrix(y_testing, prediction)
    return accuracy, matrix_confusion

model_accuracy,model_confusion_matrix = evaluate_model(best_model_grid, X_testing_clean, Y_testing)

# Output
print(f'Accuracy: {model_accuracy*100:.2f}%')
print(f'Confusion Matrix:')
print(model_confusion_matrix)