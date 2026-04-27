**Titanic Survival Prediction: Decision Tree Model**

This repository contains a modularized Python implementation of a Decision Tree Classifier to predict passenger survival on the Titanic using the classic Kaggle dataset.

**Project Overview**

The project follows a standard machine learning pipeline, organized into logical code blocks suitable for Jupyter Notebooks or standalone scripts. It handles data cleaning, feature engineering, model training, and generates a final submission file.

**Features Used**

The model utilizes the following passenger data points to make predictions:

Pclass: Ticket class (1st, 2nd, 3rd)

Sex: Gender (encoded as numerical)

Age: Passenger age (missing values imputed)

SibSp / Parch: Family relationships

Fare: Passenger fare

Embarked: Port of embarkation

FamilySize: Combined feature of SibSp + Parch

IsAlone: Binary indicator for passengers traveling without family

**Workflow**

Data Loading: Imports train.csv and test.csv.

Preprocessing: Handles missing values for Age, Embarked, Fare, and Cabin.

Feature Engineering: Converts categorical variables into numerical formats and creates new derived features.

Model Training: Implements a DecisionTreeClassifier with optimized hyperparameters (max depth, sample splits).

Evaluation: Validates performance using accuracy scores, classification reports, and confusion matrices.

Submission: Generates a submission.csv file ready for Kaggle.
