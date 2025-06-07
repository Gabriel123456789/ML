import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

##load data
data = pd.read_csv("tested.csv")
data.info()
print(data.isnull().sum()) #search for missing values

##We see ages missing, cabins(not important) and embarked

# Data cleaning
def preprocess_data(dataframe):
    #remove from the original dataframe not important things
    dataframe.drop(columns=["PassagensID", "Name", "Ticket", "Cabin"], inplace = True)
    dataframe["Embarked"].fillna("S", inplace = True) #inplace changes in the original data
    dataframe.drop(columns=["Embarked"], inplace = True)
    
    ##add the age filling function
    fill_ages(dataframe)
    
    #convert gender to numbers
    dataframe["Sex"] = dataframe["Sex"].map({"male":1,"female":0})
    
    #feature enginnering = create new columns of data
    dataframe["FamilySize"] = dataframe["SibSp"] + dataframe["Parch"] #we have data of siblings and parents on-board
    dataframe["IsAlone"] = np.where(dataframe["FamilySize"]==0,1,0) #1 for alone, 0 otherwise
    
    #create bins = range of things, ex: ranges of price they paid, age ranges, etc
    
    dataframe["Farebin"] = pd.qcut(dataframe["Fare"],4,labels=False) # qcut = ranges with same number of people
    dataframe["AgeBin"] = pd.cut(dataframe["Age"], bins = [0,12,20,40,60,np.inf],labels=False) #cut = defined ranges
    
    return dataframe

# Need to fill the ages that are missing

def fill_ages(dataframe):
    #will treat this with the mean of age in the class they were in
    age_fill_map = {} #create a map of class and age median
    for pclass in dataframe["Pclass"].unique():
        age_fill_map[pclass] = dataframe[dataframe["Pclass"] == pclass]["Age"].median()
    
    dataframe["Age"] = dataframe.apply(lambda row:age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1) #
    #above: the lambda receives each row(axis = 1) and attributes the value in the dictionary(median) if
    # the place it is is null, otherwise it keeps the row age value


# applying the preprocessing to the data
data  = preprocess_data(data)

# separate the data from the desired prediction data
X = data.drop(columns="Survived")
Y = data["Survived"]


# Separate data for training and testing
x_training, x_testing, y_training, y_testing = train_test_split(X,Y, test_size=0.25, random_state=42)

# Scale the data - Check if the data is ready for the model
scaler = MinMaxScaler()
x_training = scaler.fit_transform(x_training)
x_testing = scaler.transform(x_testing)


# Parameter Tuning - kNN - Defines the parameters used by our model
def tune_model(x_training, y_training):
    param_grid = {
        "n_neighbors": range(1,21), #the n number/parameter for knn changes
        "metrics": ["euclidean", "manhattan", "minkowski"], #different forms of calculating distance from neighbors
        "weights": ["uniform", "distance"]
    }
    
    #build the model itself
    model = KNeighborsClassifier()
    
    # Perform a grid search - uses the tune model to see wich parameters perfom better
    grid_search = GridSearchCV(model,param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_training,y_training) # Here we are training the grid search with our data
    
    return grid_search.best_estimator_

best_model = tune_model(x_training,y_training)


# Evaluate how well the model is predicting
def evalute_model(model,x_testing,y_testing):
    prediction = model.predict(x_testing) #Here the model can´t see the answers, only the x data
    accuracy = accuracy_score(y_testing, prediction) # compares the predicted with the answers
    matrix = confusion_matrix(y_testing,prediction) # a matrix that compares results x prediction
    
    return accuracy,matrix


# Give the evaluate function the best model and our testing datas
accuracy,conf_matrix = evalute_model(best_model,x_testing,y_testing)