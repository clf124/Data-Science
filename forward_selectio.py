'''
Forward selection for Logistic Regression
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.exceptions import DataConversionWarning
import warnings

# Prepare dataset
X = df.drop(columns=[
    'Coughing of Blood', 'Level', 'Smoking',
    'Passive Smoker', 'Shortness of Breath', 'Wheezing', 'Cough_Bin'
], errors='ignore')
y = df['Cough_Bin']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Forward Feature Selection
remaining_features = list(X.columns)
selected_features = []
best_score = 0
threshold_improvement = 0.01  # Minimum improvement to keep a new feature

while remaining_features:
    scores_with_candidates = []

    for feature in remaining_features:
        candidate_features = selected_features + [feature]
        X_candidate = X_train[candidate_features]
        X_candidate_scaled = scaler.fit_transform(X_candidate)

        model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0)
        f1_scores = cross_val_score(model, X_candidate_scaled, y_train, cv=StratifiedKFold(5), scoring='f1')
        mean_score = np.mean(f1_scores)
        scores_with_candidates.append((mean_score, feature))

    scores_with_candidates.sort(reverse=True)
    best_candidate_score, best_candidate_feature = scores_with_candidates[0]

    if best_candidate_score > best_score + threshold_improvement:
        selected_features.append(best_candidate_feature)
        remaining_features.remove(best_candidate_feature)
        best_score = best_candidate_score
    else:
        break

print("Best features selected (forward selection):")
print(selected_features)

# Final model training and evaluation
X_train_sel = pd.DataFrame(scaler.fit_transform(X_train[selected_features]), columns=selected_features)
X_test_sel = pd.DataFrame(scaler.transform(X_test[selected_features]), columns=selected_features)

mylr = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0)
mylr.fit(X_train_sel, y_train)

y_pred = mylr.predict(X_test_sel)

print("\nTest Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Model summary
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DataConversionWarning)
    model_summary = ModelSummary(mylr, X_train_sel, y_train)
    model_summary.get_summary()
