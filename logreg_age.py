'''
Two Variables Losgitic Regression : Comparing age and coughing of blood 
'''
from sklearn.linear_model import LogisticRegression as logreg
import pandas as pd

df = pd.read_csv('cancer_patient.csv')
X=df['Age'].values.reshape(-1, 1)
y=df['Cough_Bin'].values.reshape(-1, 1)

mylr = logreg(class_weight='balanced')
mylr.fit(X, y.ravel())

model_summary = ModelSummary(mylr, X, y)
model_summary.get_summary()
