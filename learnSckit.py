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

#data cleaning
def preprocess_data(dataframe):
    #remove from the original dataframe not important things
    dataframe.drop(columns=["PassagensID", "Name", "Ticket", "Cabin"], inplace = True)
    dataframe["Embarked"].fillna("S", inplace = True) #inplace changes in the original data
    dataframe.drop(columns=["Embarked"], inplace = True)
    

    #convert gender to numbers
    dataframe["Sex"] = dataframe["Sex"].map({"male":1,"female":0})
    
    #feature enginnering = create new columns of data
    dataframe["FamilySize"] = dataframe["SibSp"] + dataframe["Parch"] #we have data of siblings and parents on-board
    dataframe["IsAlone"] = np.where(dataframe["FamilySize"]==0,1,0) #1 for alone, 0 otherwise
    
    #create bins = range of things, ex: ranges of price they paid, age ranges, etc
    
    dataframe["Farebin"] = pd.qcut(dataframe["Fare"],4,labels=False) # qcut = ranges with same number of people
    dataframe["AgeBin"] = pd.cut(dataframe["Age"], bins = [0,12,20,40,60,np.inf],labels=False) #cut = defined ranges
    
    return dataframe

# need to fill the ages

def fill_ages(dataframe):
    #will treat this with the mean of age in the class they were in
    age_fill_map = {} #create a map of class and age median
    for pclass in dataframe["Pclass"].unique():
        age_fill_map[pclass] = dataframe[dataframe["Pclass"] == pclass]["Age"].median()
    
    dataframe["Age"] = dataframe.apply(lambda row:age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1) #
    #above: the lambda receives each row(axis = 1) and attributes the value in the dictionary(median) if
    # the place it is is null, otherwise it keeps the row age value