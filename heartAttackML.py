## This script will have the purpose of studying EDA - Exploratory data analysis and multiclassification models

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
# Won´t need because there is no filling to do
#from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import seaborn as sbs

#Load data
data = pd.read_csv("heart.csv")
#data.info()
#print(data.isnull().sum()) ## Its complete yay!

## Data format
#Age: age of the patient [years]
#Sex: sex of the patient [M: Male, F: Female]
#ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
#RestingBP: resting blood pressure [mm Hg]
#Cholesterol: serum cholesterol [mm/dl]
#FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
#RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
#MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
#ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
#Oldpeak: oldpeak = ST [Numeric value measured in depression]
#ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
#HeartDisease: output class [1: heart disease, 0: Normal]

numerical_cols = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]

words_cols = [
    'Sex',
    'ChestPainType',
    'FastingBS',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]

# First part - EDA

## Will find forms of relationing categorical columns with heart disease 

# Shows higher frequency among men
def violin_sex_hd(dataframe):
    plt.figure()
    fig = sbs.violinplot(data=dataframe, x="HeartDisease", y = "Sex")
    plt.show()

# Shows higher risk with ASY pain and low risk with ATA pain 
def violin_chestpain_hd(dataframe):
    plt.figure()
    fig = sbs.violinplot(data=dataframe, x ="HeartDisease", y = "ChestPainType")
    plt.show()

# Doesn´t show interesting data
def scater_restbp_cholesterol(dataframe):
    plt.figure()
    scatterfig = sbs.scatterplot(data=dataframe, x = "RestingBP", y = "Cholesterol")
    plt.show()

# Shows relation between RestingBP/Age, MaxHR/Cholesterol, OldPeak/Age
def heat_numericals(dataframe,ncolumns):
    plt.figure()
    heatmap = sbs.heatmap(dataframe[ncolumns].corr(), annot=True)
    plt.tight_layout()
    plt.show()


## Second part - Pipeline and model training

# Start building the pipeline

# It is necessary to discretize some columns values, so we will build the preprocessor

# The columntransformer works with - instruction name - used tool - columns to apply
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(),numerical_cols),
        ( 'words', OneHotEncoder(handle_unknown='ignore'),words_cols)
    ]
)

    ## I wanted to go straight to model tuning with gridsearch but lets make a baseline model first
def base_model_training(data,preprocessor):
    X = data.drop(columns="HeartDisease")
    Y = data["HeartDisease"]
    X_training,X_testing,Y_training,Y_testing = train_test_split(X,Y, test_size=0.4, random_state=42)
    
    # Now the processor is ready we will apply it in our data and train it
    X_training_clean = preprocessor.fit_transform(X_training)
    X_testing_clean = preprocessor.transform(X_testing)

    # Here we create our base model, train and test it
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_training_clean,Y_training)
    predictions = baseline_model.predict(X_testing_clean)
    accuracy = accuracy_score(Y_testing,predictions)
    report = classification_report(Y_testing,predictions)
    print(f"Accuracy {accuracy*100:.2f}%")
    print(report)
