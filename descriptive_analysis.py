import pandas as pd 
import matplotlib.pyplot as plt

 # read the csv file
df= pd.read_csv('cancer_patient.csv') 

# Calculate statistics
stats = df['Coughing of Blood'].describe()
stats['Median'] = df['Coughing of Blood'].median()
stats['Range'] = stats['max'] - stats['min']

# Format as table
stats_df = pd.DataFrame(stats)
stats_df.columns = ['Value']
stats_df.index = [
    'Count', 'Mean', 'Standard Deviation', 'Minimum', 
    '25th Percentile', 'Median (50th)', '75th Percentile', 
    'Maximum', 'Median', 'Range'
]

# Display table
print("\n Coughing of Blood - Statistical Summary:\n")
print(stats_df.round(2))
