'''
Histogram Distribution of Coughing of Blood Variable
'''
import pandas as pd 
import matplotlib.pyplot as plt

df= pd.read_csv('cancer_patient.csv')  # read the csv file

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Coughing of Blood'], bins=8, edgecolor='black')

# Add labels and title
plt.title('Histogram distribution of Coughing of Blood ')
plt.xlabel('Coughing of Blood Level')
plt.ylabel('Frequency')

# Show the plot
plt.grid(True)
plt.show()
