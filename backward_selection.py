'''
Backward selection Logistic Regession
'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.exceptions import DataConversionWarning
import pandas as pd
import warnings

df = pd.read_csv('cancer_patient.csv')

#Preparing dataset
X = df.drop(columns=[
    'Coughing of Blood', 'Level', 'Smoking',
    'Passive Smoker', 'Shortness of Breath', 'Wheezing', 'Cough_Bin'
], errors='ignore')

y = df['Cough_Bin']

# Train-test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection on training data
mylr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0)
selector = RFECV(estimator=mylr, step=1, cv=StratifiedKFold(5), scoring='f1')
selector.fit(X_train_scaled, y_train)

# Get selected features
selected_features = X.columns[selector.support_]
print("Best features selected (backward selection):")
print(selected_features)

# Reassign feature names
X_train_sel = pd.DataFrame(X_train_scaled[:, selector.support_], columns=selected_features)
X_test_sel = pd.DataFrame(X_test_scaled[:, selector.support_], columns=selected_features)

# Train model
mylr.fit(X_train_sel, y_train)

# Evaluate on test set
y_pred = mylr.predict(X_test_sel)
print("\nTest Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Use ModelSummary on the training set with named features
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DataConversionWarning)
    model_summary = ModelSummary(mylr, X_train_sel, y_train)
    model_summary.get_summary()
