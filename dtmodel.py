import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import WOEEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_style('whitegrid') 
sns.set_palette('pastel')  

import warnings
warnings.simplefilter("ignore")


train_df = pd.read_csv('fraudTrain.csv', index_col='Unnamed: 0')
test_df = pd.read_csv('fraudTest.csv', index_col='Unnamed: 0')


is_fraud = train_df["is_fraud"].value_counts()
print("Yes: ",is_fraud[1])
print("No: ",is_fraud[0])


# Pre-processing
unique_transaction_count = len(train_df['trans_num'].unique())
print("Total count of unique transaction numbers:", unique_transaction_count)

# Remove non-useful columns

columns_to_drop = ['first', 'unix_time', 'dob', 'cc_num', 'zip', 'city','street', 'state', 'trans_num', 'trans_date_trans_time']
train_df = train_df.drop(columns_to_drop, axis=1)
test_df = test_df.drop(columns_to_drop, axis=1)
train_df.head(2)


train_df['merchant'] = train_df['merchant'].apply(lambda x : x.replace('fraud_',''))

test_df['merchant'] = test_df['merchant'].apply(lambda x : x.replace('fraud_',''))

train_df.info()

test_df.info()

# Encoding
# Applying label encoding
train_df['gender'] = train_df['gender'].map({'F': 0, 'M': 1})

# Applying WOE encoding
for col in ['job','merchant', 'category', 'lat', 'last']:
    train_df[col] = WOEEncoder().fit_transform(train_df[col],train_df['is_fraud'])

# Applying label encoding
test_df['gender'] = test_df['gender'].map({'F': 0, 'M': 1})

# Applying WOE encoding
for col in ['job','merchant', 'category', 'lat', 'last']:
    test_df[col] = WOEEncoder().fit_transform(test_df[col],test_df['is_fraud'])


# Oversampling
X=train_df.drop("is_fraud",axis=1)
y=train_df['is_fraud']

# train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=65)

X_test=test_df.drop("is_fraud",axis=1)
y_test=test_df['is_fraud']

No_class = y_train[y_train==0]
yes_class = y_train[y_train==1]

print(len(No_class))
print(len(yes_class))


#!pip install -U imbalanced-learn

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=65)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(sum(y_train_smote== 0))
print(sum(y_train_smote== 1))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_smote)

X_val = scaler.transform(X_val)
X_test=scaler.transform(X_test)

len(X_train)

# Decision Tree
DT = DecisionTreeClassifier(max_depth=5, random_state=42)
DT.fit(X_train, y_train_smote)
predict_ID3 = DT.predict(X_test)
print(classification_report(y_test, predict_ID3))
ID3_accuracy = accuracy_score(predict_ID3,y_test)
print('ID3 model accuracy is: {:.2f}%'.format(ID3_accuracy*100))

import joblib

joblib.dump(DT, 'decision_tree_model.joblib')

