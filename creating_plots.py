# Creating Plots
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_theme(style="whitegrid")

# LSTM Plots 
scenario_indx   = np.array(['Scenario1', 'Scenario5', 'Scenario9'])
accuracy        = np.array([63.61, 51.03, 36.98])

data = {
    'Scenarios': [1, 5, 9] * 4,
    'Accuracy': [
        41, 37, 35,  # Lookup Table
        49, 46, 37,  # K-Nearest Neighbors
        55, 44, 38,   # Neural Network
        63.61, 51.03, 36.98   # LSTM Network
    ],
    'Method': ['Lookup Table']*3 + ['K-Nearest Neighbors']*3 + ['Neural Network']*3 \
        + ['LSTM']*3
}
    
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 7))
sns.barplot(x='Accuracy', y='Scenarios', hue='Method', data=df, orient='h')

plt.xlabel('Top-1 Beam Prediction Accuracy [%]')
plt.ylabel('Scenarios')
plt.legend(title='Method', loc='lower right')
plt.grid(True)

plt.show()

# sns.set_color_codes("pastel")
# sns.barplot(x=accuracy[::-1], y=scenario_indx[::-1], color="b", label='LSTM')
# sns.barplot(x=accuracy, y=scenario_indx, color="g", label='LSTM')
# plt.show()