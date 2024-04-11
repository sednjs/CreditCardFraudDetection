import pandas as pd
from category_encoders import WOEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import joblib

import warnings
warnings.simplefilter("ignore")

# Load the data
train_df = pd.read_csv('fraudTrain.csv', index_col='Unnamed: 0')
test_df = pd.read_csv('fraudTest.csv', index_col='Unnamed: 0')

# Changing 'trans_date_trans_time' to datetime
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')

# creating 'hour' and 'month'
train_df['hour'] = train_df['trans_date_trans_time'].dt.hour
test_df['hour'] = test_df['trans_date_trans_time'].dt.hour

train_df['month'] = train_df['trans_date_trans_time'].dt.month
test_df['month'] = test_df['trans_date_trans_time'].dt.month

train_df.head()

# Pre-processing

# Remove non-useful columns for both test and train datasets
columns_to_drop = ['first', 'unix_time', 'dob', 'cc_num', 'zip', 'city','street', 'state', 'trans_num', 'trans_date_trans_time']
train_df = train_df.drop(columns_to_drop, axis=1)
test_df = test_df.drop(columns_to_drop, axis=1)
train_df.head(2)

# Remove unnecessary string in both test and train datasets
train_df['merchant'] = train_df['merchant'].apply(lambda x : x.replace('fraud_',''))
test_df['merchant'] = test_df['merchant'].apply(lambda x : x.replace('fraud_',''))

# Splitting training into train and validation
X=train_df.drop("is_fraud",axis=1)
y=train_df['is_fraud']

# train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

X_test=test_df.drop("is_fraud",axis=1)
y_test=test_df['is_fraud']

# Pipeline of WOEEncode, StandardScaler, and SMOTE (Oversampling)
X_train.isna().sum()

# Mapping gender to 1, 0
def map_gender(X):
    if X['gender'].dtype in ['int64', 'float64']:
        return X  
    else: 
        X['gender'] = X['gender'].map({'F': 0, 'M': 1})
        return X

# Preprocessing
preprocessor=ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(),['amt','lat','long','city_pop','merch_lat','merch_long','hour','month']), # for numerical var
        ('categorical', WOEEncoder(),['merchant', 'category', 'last','job']), # for categorical var
    ], 
    remainder='passthrough'
    
)

preprocessing_pipeline = make_pipeline(
    FunctionTransformer(map_gender),  
    preprocessor
)

X_train =preprocessing_pipeline.fit_transform(X_train,y_train)

smote=SMOTE(random_state=42)
X_train_s,y_train_s=smote.fit_resample(X_train,y_train)

# Preprocessing validation dataset with pipeline
X_val_p=preprocessing_pipeline.transform(X_val)
DT = DecisionTreeClassifier(max_depth=5, random_state=42)
DT.fit(X_train_s,y_train_s)
predict_DT = DT.predict(X_val_p)

print(classification_report(y_val, predict_DT))
DT_accuracy = accuracy_score(predict_DT,y_val)
print('Decision Tree accuracy is: {:.2f}%'.format(DT_accuracy*100))

# Save the model
joblib.dump((DT, preprocessing_pipeline), 'decision_tree_model.joblib')


