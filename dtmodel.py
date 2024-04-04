import pandas as pd
from category_encoders import WOEEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import joblib

# THESE DATASETS HAVE ALREADY DROP ALL OF THE FOLLOWING COLUMNS: ['first', 'unix_time', 'dob', 'cc_num', 'zip',
# 'city', 'street', 'state', 'trans_num', 'trans_date_trans_time']

# Load the data
train_df = pd.read_csv('fraudTrain.csv', index_col='Unnamed: 0')
test_df = pd.read_csv('fraudTest.csv', index_col='Unnamed: 0')

# Simple label encoding for 'gender'
train_df['gender'] = train_df['gender'].map({'F': 0, 'M': 1})
test_df['gender'] = test_df['gender'].map({'F': 0, 'M': 1})

# Initialize and fit WOEEncoder
categorical_columns = ['merchant', 'category', 'job', 'last']
woe_encoder = WOEEncoder(cols=categorical_columns)

# Apply WOEEncoder and replace the original dataframe
train_df_encoded = woe_encoder.fit_transform(train_df.drop(columns=['is_fraud']), train_df['is_fraud'])
test_df_encoded = woe_encoder.transform(test_df.drop(columns=['is_fraud']))

# Directly assign encoded DataFrames
train_df = train_df_encoded.join(train_df['is_fraud'])
test_df = test_df_encoded.join(test_df['is_fraud'])

# Handle class imbalance with SMOTE
X_train, X_val, y_train, y_val = train_test_split(train_df.drop(columns=['is_fraud']), train_df['is_fraud'], test_size=0.2, random_state=65)
smote = SMOTE(random_state=65)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df.drop(columns=['is_fraud']))

# Train a Decision Tree Classifier
DT = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
DT.fit(X_train_scaled, y_train_smote)

# Prediction and Evaluation
predictions = DT.predict(X_test_scaled)
print(classification_report(test_df['is_fraud'], predictions))
# Evaluation on the validation set
val_predictions = DT.predict(X_val_scaled)
print("Validation Set Evaluation")
print(classification_report(y_val, val_predictions))

# Cross validation
scores = cross_val_score(DT, X_train_scaled, y_train_smote, cv=5)
print("Cross-validation scores:", scores)

# Save the model, scaler, and encoder
joblib.dump(DT, 'decision_tree_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(woe_encoder, 'woe_encoder.joblib')

sample_predictions = DT.predict(X_val_scaled[:5])
print("Sample predictions:", sample_predictions)

print("Shape after encoding:", train_df_encoded.shape)
print("Shape after SMOTE:", X_train_smote.shape)
print("Shape after scaling:", X_train_scaled.shape)

