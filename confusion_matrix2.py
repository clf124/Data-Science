'''
Plotting confusion matrix with balanced dataset
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv('cancer_patient.csv')

# Define features 
X = df.drop(columns=[
    'Coughing of Blood', 'Level', 'Smoking',
    'Passive Smoker', 'Shortness of Breath', 'Wheezing', 'Cough_Bin'
], errors='ignore')
y = df['Cough_Bin']


# Train model
model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Compute and plot confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "High"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Coughing of Blood Level Classification")
plt.show()
