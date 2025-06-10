## This script will have the purpose of studying EDA - Exploratory data analysis and multiclassification models

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
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

### OLD PREPROCESSOR - check line 422

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', MinMaxScaler(),numerical_cols),
#         ( 'words', OneHotEncoder(handle_unknown='ignore'),words_cols)
#     ]
# )


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
    print("Results for grid simpler random forest")
    results_show(best_model,X_testing,Y_testing)

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

def results_show(best_model,x_testing,y_testing):
    prediction = best_model.predict(x_testing)
    accuracy = accuracy_score(y_testing,prediction)
    conf_matrix = confusion_matrix(y_testing,prediction)
    report = classification_report(y_testing,prediction)
    print(f"Accuracy {accuracy*100:.2f}%")
    print(report)
    print("Confusion Matrix:\n",conf_matrix)
    return prediction


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
    print("Results for Grid Tuning enhanced random forest")
    prediction = results_show(best_model,X_testing,Y_testing)
    return prediction, X_testing, Y_testing

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
    print("Results for grid gradient booster")
    results_show(best_model,X_testing,Y_testing)

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

## I am going to analyse the confusion matrix and try to see patterns with these patientes

def false_neg_debug(dataframe,preprocessor, ncols):
    # Will make a change in the og function so we can have the model results here
    predictions, x_testing, y_testing = gridtuning_random_forest_enhanced(dataframe,preprocessor)
    
    # We have to create a new dataframe only with the false negatives patientes - sick people the model said were fine
    false_negative_mask = (predictions == 0) & (y_testing == 1)
    false_neg_dataframe = x_testing.loc[false_negative_mask]
    
    # We have to compare the false with the true so we can find divergencies
    true_positives_mask = (predictions == 1) & (y_testing == 1)
    true_positives_dataframe = x_testing.loc[true_positives_mask]
    
    # The .describe module gives statistics on the columns
    print("False negative statistics: ")
    print(false_neg_dataframe[ncols].describe())
    print("True positive statistics: ")
    print(true_positives_dataframe[ncols].describe())


## Results of the data comparison
# False negative statistics: 
#              Age   RestingBP  Cholesterol       MaxHR    Oldpeak
# count  29.000000   29.000000    29.000000   29.000000  29.000000
# mean   53.275862  129.275862   191.724138  148.758621   0.334483
# std     8.770472   21.011726   113.919425   25.299146   0.628647
# min    40.000000   95.000000     0.000000   97.000000  -0.700000
# 25%    47.000000  112.000000   172.000000  125.000000   0.000000
# 50%    54.000000  130.000000   236.000000  152.000000   0.000000
# 75%    58.000000  140.000000   265.000000  170.000000   0.600000
# max    73.000000  192.000000   319.000000  195.000000   2.500000
# True positive statistics: 
#               Age   RestingBP  Cholesterol       MaxHR     Oldpeak
# count  192.000000  192.000000   192.000000  192.000000  192.000000
# mean    55.807292  133.229167   172.364583  123.802083    1.303125
# std      8.359047   19.658061   130.411441   22.180185    1.089642
# min     31.000000   92.000000     0.000000   60.000000   -1.000000
# 25%     51.000000  120.000000     0.000000  109.750000    0.475000
# 50%     56.000000  131.000000   214.000000  122.500000    1.200000
# 75%     62.000000  145.000000   266.000000  140.000000    2.000000
# max     77.000000  200.000000   529.000000  182.000000    6.200000


## Analyzing this data we can see a problem that I did not see before; There are patients with cholesterol = 0
# This can disrupt our model training and now we have to see how to take care of it

# total_rows = len(data)
# cholesterol_zero_count = (data['Cholesterol'] == 0).sum()
# percentge = (cholesterol_zero_count/total_rows) * 100

## Result: The percentage of zero cholesterol patients is: 18.74%

# This shows that the data is enough that it´s dismissal will bring problems to the model training
# So now the data will have to be treated and the data will be imputate

data['Cholesterol'] = data['Cholesterol'].replace(0,np.nan)

## The preprocessor will have to be changed so it can impute ONLY at numerical columns

### NEM PREPROCESSOR - The difference is that now we will treat the cholesterol rows that were 0 into NaN
# and them imputate them, so:

numerical_treatment = Pipeline(steps=[('imputer',KNNImputer(n_neighbors=5)), ('scaler', MinMaxScaler())])
preprocessor = ColumnTransformer(transformers=[('num',numerical_treatment, numerical_cols),
        ('words', OneHotEncoder(handle_unknown='ignore'),words_cols)])

# Now I am going to test all our data again

## Results after imputation:
# Model KNN: Accuracy 85.05%
# Model RandomForest: Accuracy 85.60%
# Model Logistic Regression: Accuracy 84.78%
# Model Gradient Boosting: Accuracy 84.24%
# Model Decision tree: Accuracy 73.37%

# Accuracy 85.60%
#               precision    recall  f1-score   support

#            0       0.78      0.89      0.83       147
#            1       0.92      0.83      0.87       221

#     accuracy                           0.86       368
#     macro avg       0.85      0.86      0.85       368
#     weighted avg       0.86      0.86      0.86       368

# Confusion Matrix:
#  [[131  16]
#  [ 37 184]]
 
# Results for grid gradient booster
# Accuracy 84.51%
#               precision    recall  f1-score   support

#            0       0.76      0.90      0.82       147
#            1       0.92      0.81      0.86       221

#     accuracy                           0.85       368
#    macro avg       0.84      0.85      0.84       368
# weighted avg       0.86      0.85      0.85       368

# Confusion Matrix:
#  [[132  15]
#  [ 42 179]]
 
# Results for Grid Tuning enhanced random forest
# Accuracy 85.60%
#               precision    recall  f1-score   support

#            0       0.78      0.89      0.83       147
#            1       0.92      0.83      0.87       221

#     accuracy                           0.86       368
#     macro avg       0.85      0.86      0.85       368
#     weighted avg       0.86      0.86      0.86       368

# Confusion Matrix:
#  [[131  16]
#  [ 37 184]]
 
# Results for grid simpler random forest
# Accuracy 85.60%
#               precision    recall  f1-score   support

#            0       0.78      0.89      0.83       147
#            1       0.92      0.83      0.87       221

#     accuracy                           0.86       368
#     macro avg       0.85      0.86      0.85       368
#     weighted avg       0.86      0.86      0.86       368

# Confusion Matrix:
#  [[131  16]
#  [ 37 184]]