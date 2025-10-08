import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('market_data.csv')

# Example of a basic plot
df.plot(x='Date', y='Metric')
plt.show()

