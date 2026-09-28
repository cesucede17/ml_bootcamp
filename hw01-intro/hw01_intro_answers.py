import pandas as pd
import numpy as np

# Load dataset

df = pd.read_csv('../car_fuel_efficiency_2026.csv')

# 1) How many records are in the dataset?
records = len(df)
print('records:', records)

# 2) How many fuel types are in the dataset?
num_fuel_types = df['fuel_type'].nunique()
print('fuel_types:', num_fuel_types)

# 3) Missing values per column
after_missing = df.isnull().sum()
print('missing_values:')
print(after_missing)

# 4) Maximum fuel efficiency among cars from Asia
asia_max_fuel_efficiency = df.loc[df['origin'] == 'Asia', 'fuel_efficiency_mpg'].max()
print('asia_max_fuel_efficiency:', asia_max_fuel_efficiency)

# 5) Median value of horsepower
median_before = df['horsepower'].median()
print('median_before:', median_before)

# 6) Most frequent value of horsepower
most_frequent = df['horsepower'].mode()[0]
print('most_frequent:', most_frequent)

# 7) Fill missing horsepower values with the most frequent value, then recompute median
filled_df = df.copy()
filled_df['horsepower'] = filled_df['horsepower'].fillna(most_frequent)
median_after = filled_df['horsepower'].median()
print('median_after:', median_after)

# 8) Matrix multiplication exercise
asia = df[df['origin'] == 'Asia']
X_df = asia[['vehicle_weight', 'model_year']].head(7)
X = X_df.to_numpy()
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200], dtype=float)
w = XTX_inv.dot(X.T).dot(y)
print('w:', w)
print('sum_w:', w.sum())
