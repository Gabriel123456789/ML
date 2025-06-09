## This script will have the purpose of studying EDA - Exploratory data analysis and multiclassification models

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sbs

##Load our data
data = pd.read_csv("global_cancer_patients_2015_2024.csv")
#data.info()
#print(data.isnull().sum()) #Great, there is no missing data!!

## Data Columns
#Patient_ID               
#Age                      
#Gender - male/female/other                   
#Country_Region - words         
#Year                     
#Genetic_Risk             
#Air_Pollution            
#Alcohol_Use              
#Smoking                  
#Obesity_Level            
#Cancer_Type - words              
#Cancer_Stage - Stage x            
#Treatment_Cost_USD       
#Survival_Years           
#Target_Severity_Score    


##### First code part - EDA

# In this part of the code we use seaborn and matplotlib to make and show how different data relate to eachother
numerical_cols = [
    'Age', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level', 
    'Treatment_Cost_USD', 'Survival_Years', 'Target_Severity_Score'
]

words_cols = ["Gender", "Country_Region", "Cancer_Type", "Cancer_Stage"]

#Plot numerical heatmap
plt.figure()
sbs.heatmap(data[numerical_cols].corr(), annot=True)
plt.tight_layout()
plt.show()

## The heatmap shows a strong relation between severity and alcohol use, smoking and air poluttion.
# Otherwise, it showed a opposite relation between severity and treatment cost

# Analyse words and numerical data combined
plt.figure()
sbs.boxplot(data=data, x = "Cancer_Type", y = "Alcohol_Use")
plt.show()