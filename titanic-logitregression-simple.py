# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# %%
traindf =pd.read_csv('/Users/saylijavadekar/Library/CloudStorage/Dropbox/Python/Kaggel/Titanic/train.csv')
traindf.head()
testdf =pd.read_csv('/Users/saylijavadekar/Library/CloudStorage/Dropbox/Python/Kaggel/Titanic/test.csv')
testdf.head()
testdf1 =pd.read_csv('/Users/saylijavadekar/Library/CloudStorage/Dropbox/Python/Kaggel/Titanic/test.csv')

# %%
# Import libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold, cross_validate
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# %%
# Data Preprocessing of traindf
traindf['Age'] = traindf['Age'].fillna(traindf['Age'].median())
traindf['Gender'] = traindf['Sex'].map({'male': 1, 'female': 0})
# Fill missing values with the most frequent value
most_frequent = traindf['Embarked'].mode()[0]
traindf['Embarked'].fillna(most_frequent, inplace=True)
traindf = pd.get_dummies(traindf, columns=['Embarked'], drop_first=False)  # One-hot encoding for Embarked
# other features to be added fare, sibsp, parch, pclass


# %%
traindf = pd.get_dummies(traindf, columns=['Pclass'], drop_first=False)
traindf['Pclass_L'] = traindf['Pclass_3'].apply(lambda x: 1 if x == 1 else 0)
traindf['Pclass_M'] = traindf['Pclass_2'].apply(lambda x: 1 if x == 1 else 0)

# %%
traindf['Embarked_q'] = traindf['Embarked_Q'].apply(lambda x: 1 if x == 1 else 0)
traindf['Embarked_s'] = traindf['Embarked_S'].apply(lambda x: 1 if x == 1 else 0)

# %%
# Trying two Model with and without regularisation

# Step 1 : The BASE MODEL.
# Base Model : Logistic Regression without Regularisation

X = traindf[['Age', 'Gender', 'Fare', 'SibSp', 'Parch', 'Embarked_q', 'Embarked_s', 'Pclass_L', 'Pclass_M']]
y = traindf['Survived']

# %%
# Ensure X contains only numeric data
X = X.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric columns to NaN
# Ensure y is binary and numeric
y = pd.to_numeric(y, errors='coerce')

print(X.dtypes)
print(y.dtypes)


# %%
X_with_const = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X_with_const)
result = model.fit()

# Display the summary table
print(result.summary())

# %%
scaler = StandardScaler()
columns_to_scale = ['Age', 'Fare']
# Apply the StandardScaler to these columns
traindf[columns_to_scale] = scaler.fit_transform(traindf[columns_to_scale])



# %%
# Now you can select the other columns along with the scaled ones
X = traindf[['Age', 'Gender', 'Fare', 'SibSp', 'Parch', 'Embarked_q', 'Embarked_s', 'Pclass_L', 'Pclass_M']]

# %%
X_with_const = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X_with_const)
result = model.fit()

# Display the summary table
print(result.summary())

# %%
# So the base model in itself looks fine. Now we will use this base model with crossvalidation to check the metrics of the model. 

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Initialize Logistic Regression classifier (no regularization)
model1 = LogisticRegression(solver='liblinear', C=1e10)  # C=1 means no regularization

# Initialize Logistic Regression classifier ( regularization)
model2 = LogisticRegression(solver='liblinear', penalty='l2')  # C=1 means no regularization



# %%
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results1 = cross_validate(model1, X, y, cv=kf, scoring=scoring, return_train_score=False)
cv_results2 = cross_validate(model2, X, y, cv=kf, scoring=scoring, return_train_score=False)


# %%
# Print metrics for each fold
for metric in scoring:
    print(f"\n{metric.capitalize()} for each fold of model1: {cv_results1[f'test_{metric}']}")
    print(f"Average {metric.capitalize()}: {cv_results1[f'test_{metric}'].mean():.4f} ± {cv_results1[f'test_{metric}'].std():.4f}")



# %%
# Print metrics for each fold
for metric in scoring:
    print(f"\n{metric.capitalize()} for each fold of model2: {cv_results2[f'test_{metric}']}")
    print(f"Average {metric.capitalize()}: {cv_results2[f'test_{metric}'].mean():.4f} ± {cv_results2[f'test_{metric}'].std():.4f}")    

# %%
# Data Preprocessing

testdf['Age'] = testdf['Age'].fillna(testdf['Age'].median())
testdf['Fare'] = testdf['Fare'].fillna(testdf['Fare'].median())
testdf['Gender'] = testdf['Sex'].map({'male': 1, 'female': 0})
# Fill missing values with the most frequent value
most_frequent = testdf['Embarked'].mode()[0]
testdf['Embarked'].fillna(most_frequent, inplace=True)
testdf = pd.get_dummies(testdf, columns=['Embarked'], drop_first=False)  # One-hot encoding for Embarked
testdf = pd.get_dummies(testdf, columns=['Pclass'], drop_first=False)
testdf['Pclass_L'] = testdf['Pclass_3'].apply(lambda x: 1 if x == 1 else 0)
testdf['Pclass_M'] = testdf['Pclass_2'].apply(lambda x: 1 if x == 1 else 0)
testdf['Embarked_q'] = testdf['Embarked_Q'].apply(lambda x: 1 if x == 1 else 0)
testdf['Embarked_s'] = testdf['Embarked_S'].apply(lambda x: 1 if x == 1 else 0)
scaler = StandardScaler()
columns_to_scale = ['Age', 'Fare']
# Apply the StandardScaler to these columns
testdf[columns_to_scale] = scaler.fit_transform(testdf[columns_to_scale])



# %%
print(X_train.isnull().sum())  
print(X_test.isnull().sum()) 

# %%
# Ensure X contains only numeric data
X_test = X_test.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric columns to NaN


print(X_test.dtypes)


# %%
X_test = testdf[['Age', 'Gender', 'Fare', 'SibSp', 'Parch', 'Embarked_q', 'Embarked_s', 'Pclass_L', 'Pclass_M']]


print(X_test.dtypes)

# %%
# Now that I have selected model 2, I will have to first train the traindf.

X_train = X = traindf[['Age', 'Gender', 'Fare', 'SibSp', 'Parch', 'Embarked_q', 'Embarked_s', 'Pclass_L', 'Pclass_M']]
y_train = traindf['Survived']



# %%
X_train.shape, y_train.shape, X_test.shape,

# %%
from sklearn.linear_model import LogisticRegression

# %%


model = LogisticRegression(solver='liblinear', penalty='l2')
model.fit(X_train, y_train)


# %%
y_pred = model.predict(X_test)

# %%
print("Train columns:", X_train.columns)
print("Test columns:", X_test.columns)

# %%
print("Train columns:", X_train.dtypes)
print("Test columns:", X_test.dtypes)

# %%
testdf1['Survived'] = y_pred

# %%
print(testdf1.head())

# %%
Submission = testdf1[['PassengerId', 'Survived']]
Submission.to_csv('/Users/saylijavadekar/Library/CloudStorage/Dropbox/Python/Kaggel/Titanic/submission.csv', index=False)


