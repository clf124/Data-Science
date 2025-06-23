'''
Final model metrics
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load the dataset
df = pd.read_csv("cancer_patient.csv")

# Define X and y
X = df.drop(columns=[
    'Coughing of Blood', 'Level', 'Smoking',
    'Passive Smoker', 'Shortness of Breath', 'Wheezing', 'Cough_Bin'
], errors='ignore')
y = df['Cough_Bin']

# Convert categorical variables to indicator variables
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Logistic Regression model
lr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0)

# Backward Selection
sfs = SFS(lr,
          k_features='best',
          forward=False,
          floating=False,
          scoring='accuracy',
          cv=5)

sfs.fit(X_train_scaled, y_train)

# Select the best features
selected_features = list(sfs.k_feature_idx_)
X_train_sel = X_train_scaled[:, selected_features]
X_val_sel = X_val_scaled[:, selected_features]

# Retrain model on selected features
lr.fit(X_train_sel, y_train)

# Evaluate
for label, X_data, y_data in [('Train', X_train_sel, y_train), ('Validation', X_val_sel, y_val)]:
    y_pred = lr.predict(X_data)
    print(f"{label} Accuracy: {accuracy_score(y_data, y_pred):.3f}")
    print(f"{label} Precision: {precision_score(y_data, y_pred):.6f}")
    print(f"{label} Recall: {recall_score(y_data, y_pred):.6f}")
    print()
