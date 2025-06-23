'''
Generating four different plots to see individual feature effects on probability of Coughing Blood for the variables Genetic Risk, Chest Pain, Respiratory Distress and Level of Cancer.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('cancer_patient.csv')

# Features and target
features = ['Genetic Risk', 'Chest Pain', 'Respiratory Distress', 'Level_num']
target = 'Cough_Bin'

X = df[features]
y = df[target]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Get medians for baseline
medians = X_train.median().values.reshape(1, -1)
n_points = 100

# Plot each feature in its own subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    varied = np.tile(medians, (n_points, 1))
    feature_range = np.linspace(X_train[feature].min(), X_train[feature].max(), n_points)
    varied[:, i] = feature_range
    varied_scaled = scaler.transform(varied)
    probs = model.predict_proba(varied_scaled)[:, 1]

    ax = axes[i]
    ax.plot(feature_range, probs, color='steelblue')
    ax.set_title(f"Effect of '{feature}'")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Predicted Probability")
    ax.grid(True)

plt.suptitle("Individual Feature Effects on Probability of Coughing Blood", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
