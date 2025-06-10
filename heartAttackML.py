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
def base_model_training_random_forest(data,preprocessor):
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
    conf_matrix = confusion_matrix(Y_testing, predictions)
    print(f"Accuracy {accuracy*100:.2f}%")
    print(report)
    print("Confusion Matrix:\n",conf_matrix)


###Baseline Results
# Accuracy 87.23%
#               precision    recall  f1-score   support

#            0       0.80      0.91      0.85       147
#            1       0.94      0.85      0.89       221

#     accuracy                           0.87       368
#     macro avg       0.87      0.88      0.87       368
#     weighted avg       0.88      0.87      0.87       368

def gridtuning_model_random_forest(data,preprocessor):
    X = data.drop(columns="HeartDisease")
    Y = data["HeartDisease"]
    X_training,X_testing,Y_training,Y_testing = train_test_split(X,Y, test_size=0.4, random_state=42)
    
    # Now the processor is ready we will apply it in our data, building the pipeline
    # The model will be defined as random forest
    model = RandomForestClassifier(random_state=42)
    
    # What happens is that we decide how the data in the pipeline will be processed and wich model is going
    # to predict our data. The gridsearch will be used to tune how the random forest model will work
    final_pipeline = Pipeline(steps=[('preprocessing', preprocessor), ('classifier', model)])
    
    # Define the parameters for the gridsearch
    param_grid = {
        # Indicates the number of trees
        'classifier__n_estimators' : [100,150],
        
        # Indicates the size of each tree
        'classifier__max_depth' :[10,30]
    }
    
    # Define how the tunning will work
    grid_model_tuning = GridSearchCV(final_pipeline,param_grid,cv=5,n_jobs=-1)
    
    # We give the model the raw data because the pipeline will clean it
    grid_model_tuning.fit(X_training,Y_training)
    
    # After the gridsearch test we will have the best parameters
    best_model = grid_model_tuning.best_estimator_
    
    # Now we can test our best model and check if its accurate
    prediction = best_model.predict(X_testing)
    accuracy = accuracy_score(Y_testing,prediction)
    conf_matrix = confusion_matrix(Y_testing,prediction)
    report = classification_report(Y_testing,prediction)
    print(f"Accuracy {accuracy*100:.2f}%")
    print(report)
    print("Confusion Matrix:\n",conf_matrix)

## Grid search tunning results with random forest and

# param_grid = {
#         # Indicates the number of trees
#         'classifier__n_estimators' : [100,150],
        
#         # Indicates the size of each tree
#         'classifier__max_depth' :[10,30]
#     }
# Accuracy 86.96%
#               precision    recall  f1-score   support

#            0       0.80      0.90      0.85       147
#            1       0.93      0.85      0.89       221

#     accuracy                           0.87       368
#     macro avg       0.86      0.87      0.87       368
#     weighted avg       0.88      0.87      0.87       368


## Start creating, testing and tuning different models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def various_models_baseline(data,preprocessor):
    ## Create a dictionary with our models names and modules
    models = {
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Let´s split our data
    X = data.drop(columns="HeartDisease")
    Y = data["HeartDisease"]
    X_training,X_testing,Y_training,Y_testing = train_test_split(X,Y, test_size=0.4, random_state=42)
    
    # Create a dictionary to save our baseline models results
    baseline_results = {}
    
    X_training_clean = preprocessor.fit_transform(X_training)
    X_testing_clean = preprocessor.transform(X_testing)
    
    for name, model in models.items():
        model.fit(X_training_clean,Y_training)
        predictions = model.predict(X_testing_clean)
        accuracy = accuracy_score(Y_testing,predictions)
        baseline_results[name] = accuracy
        print(f"Model {name}: Accuracy {accuracy*100:.2f}%")

## Accuracy results by models
# Model KNN: Accuracy 86.14%
# Model RandomForest: Accuracy 87.23%
# Model Logistic Regression: Accuracy 85.60%
# Model Gradient Boosting: Accuracy 86.41%
# Model Decision tree: Accuracy 73.91%


# With the results from each model baseline we can choose the best ones to tune and gridsearch
# Lets create the pipeline and gridsearch for randomforest (again, but with more tuning) and gradient boosting

# I am going to reuse the grid format of random forest we already built, but change the parameters

def gridtuning_random_forest_enhanced(data,preprocessor):
    X = data.drop(columns="HeartDisease")
    Y = data["HeartDisease"]
    X_training,X_testing,Y_training,Y_testing = train_test_split(X,Y, test_size=0.4, random_state=42)
    
    # Now the processor is ready we will apply it in our data, building the pipeline
    # The model will be defined as random forest
    model = RandomForestClassifier(random_state=42)
    
    # What happens is that we decide how the data in the pipeline will be processed and wich model is going
    # to predict our data. The gridsearch will be used to tune how the random forest model will work
    final_pipeline = Pipeline(steps=[('preprocessing', preprocessor), ('classifier', model)])
    
    # Define the parameters for the gridsearch
    param_grid = {
        # Indicates the number of trees
        'classifier__n_estimators' : [100,150,200,300,400,500],
        
        # Indicates the size of each tree
        'classifier__max_depth' :[10,30,None],
        
        'classifier__min_samples_leaf': [1,2,4,8]
    }
    
    # Define how the tunning will work
    grid_model_tuning = GridSearchCV(final_pipeline,param_grid,cv=5,n_jobs=-1)
    
    # We give the model the raw data because the pipeline will clean it
    grid_model_tuning.fit(X_training,Y_training)
    
    # After the gridsearch test we will have the best parameters
    best_model = grid_model_tuning.best_estimator_
    
    # Now we can test our best model and check if its accurate
    prediction = best_model.predict(X_testing)
    accuracy = accuracy_score(Y_testing,prediction)
    conf_matrix = confusion_matrix(Y_testing,prediction)
    report = classification_report(Y_testing,prediction)
    print(f"Accuracy {accuracy*100:.2f}%")
    print(report)
    print("Confusion Matrix:\n",conf_matrix)

## Results for the enhanced forest model
# Accuracy 88.04%
#               precision    recall  f1-score   support

#            0       0.82      0.90      0.86       147
#            1       0.93      0.87      0.90       221

#     accuracy                           0.88       368
#     macro avg       0.87      0.88      0.88       368
#     weighted avg       0.88      0.88      0.88       368

# [[132  15]
#  [ 29 192]]


## Let´s create a grid and pipeline for the gradient booster

def grid_gradient_booster_model(data, preprocessor):
    X = data.drop(columns="HeartDisease")
    Y = data["HeartDisease"]
    X_training,X_testing,Y_training,Y_testing = train_test_split(X,Y, test_size=0.4, random_state=42)
    
    model = GradientBoostingClassifier(random_state=42)
    final_pipeline = Pipeline(steps=[('preprocessing',preprocessor), ('classifier', model)])
    
    param_defs = {
        'classifier__n_estimators' : [100,150,200,300],
        'classifier__learning_rate' : [0.05,0.1],
        'classifier__max_depth': [3,5,7,9] ## tree depth
    }
    
    grid_tuning = GridSearchCV(final_pipeline,param_defs,cv=5,n_jobs=-1)
    
    grid_tuning.fit(X_training,Y_training)
    
    best_model = grid_tuning.best_estimator_
    prediction = best_model.predict(X_testing)
    accuracy = accuracy_score(Y_testing,prediction)
    conf_matrix = confusion_matrix(Y_testing,prediction)
    report = classification_report(Y_testing,prediction)
    print(f"Accuracy {accuracy*100:.2f}%")
    print(report)
    print("Confusion Matrix:\n",conf_matrix)

##Results of gradient booster after tunning
# Accuracy 86.41%
#               precision    recall  f1-score   support

#            0       0.79      0.90      0.84       147
#            1       0.93      0.84      0.88       221

#     accuracy                           0.86       368
#     macro avg       0.86      0.87      0.86       368
#     weighted avg       0.87      0.86      0.87       368

# [[133  14]
#  [ 36 185]]


## After analyzing it is clear that the tuned random forest was the best model yet, I am going to try some
# feature engineering so we can get better results

#heat_numericals(data,numerical_cols)

def feature_eng_MaxHR(dataframe,ncols):
    #how i tought of doing it
    #dataframe['MaxHR_Age_Formula'] = dataframe.apply(lambda row: 1 if (220-row['Age']>row['MaxHR']) else 0, axis=1)
    
    #correct way - the astype turns all the true values to 1 and falses to 0
    dataframe['MaxHR_Age_Formula'] = ((220-dataframe['Age'])>dataframe['MaxHR']).astype(int)
    ncols.append('MaxHR_Age_Formula')
    return dataframe,ncols

# Results with the maxHR feature - The new feature wasn´t good for the model predicition
# Accuracy 86.96%
#               precision    recall  f1-score   support

#            0       0.80      0.89      0.85       147
#            1       0.92      0.86      0.89       221

#     accuracy                           0.87       368
#     macro avg       0.86      0.87      0.87       368
#     weighted avg       0.87      0.87      0.87       368

# Confusion Matrix:
#  [[131  16]
#  [ 32 189]]

