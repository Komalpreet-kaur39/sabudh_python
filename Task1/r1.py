# -------------------------------Pandas--------------------------------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Download the dataset (replace with your download method)
# Replace 'path/to/your/dataset.csv' with the actual path
url = "C:/Users/Maninder Singh/Downloads/archive/Sport car price.csv"
df = pd.read_csv(url)

# Display the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Clean the dataset
# Handle missing values (consider appropriate actions based on data analysis)
df.dropna(inplace=True)  # Drop rows with missing values (replace with imputation if needed)
# Handle duplicates
df.drop_duplicates(inplace=True)  # Remove duplicate rows

# Convert non-numeric data (consider appropriate conversions)
# Example: convert price to numeric (assuming price is a string)
df['Price (in USD)'] = df['Price (in USD)'].str.replace(',', '').astype(float)
# Convert 'Horsepower' to numeric, coercing non-numeric to NaN
df['Horsepower'] = pd.to_numeric(df['Horsepower'], errors='coerce')

# Handle missing values (replace with mean)
df['Horsepower'] = df['Horsepower'].fillna(df['Horsepower'].mean())

# Convert the '0-60 MPH Time (seconds)' column to numeric, coercing errors to NaN
df['0-60 MPH Time (seconds)'] = pd.to_numeric(df['0-60 MPH Time (seconds)'], errors='coerce')

# Drop rows where the '0-60 MPH Time (seconds)' is NaN
df.dropna(subset=['0-60 MPH Time (seconds)'], inplace=True)

# Explore the dataset
print("\nSummary statistics:")
print(df.describe())

# Group by car make and calculate average price
avg_price_by_make = df.groupby('Car Make')['Price (in USD)'].mean()
print("\nAverage price by car make:")
print(avg_price_by_make)

# Group by year and calculate average horsepower
avg_hp_by_year = df.groupby('Year')['Horsepower'].mean()
print("\nAverage horsepower by year:")
print(avg_hp_by_year)

# Fit linear regression model using DataFrame
model = LinearRegression()
model.fit(df[['Horsepower']], df['Price (in USD)'])

# Generate predictions using the same DataFrame
y = model.predict(df[['Horsepower']])

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(df['Horsepower'], df['Price (in USD)'], label='Data Points')
plt.plot(df['Horsepower'], y, color='red', label='Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('Price (in USD)')
plt.title('Price vs Horsepower (with linear regression)')
plt.legend()
plt.grid(True)
plt.show()

# Histogram of 0-60 MPH times with bins of 0.5 seconds
plt.figure(figsize=(8, 5))
plt.hist(df['0-60 MPH Time (seconds)'], bins=np.arange(0, df['0-60 MPH Time (seconds)'].max() + 0.5, 0.5))
plt.xlabel('0-60 mph (seconds)')
plt.ylabel('Number of cars')
plt.title('Histogram of 0-60 mph Times')
plt.grid(True)
plt.show()

# Filter for price > $500,000 and sort by horsepower (descending)
filtered_df = df[df['Price (in USD)'] > 500000].sort_values(by='Horsepower', ascending=False)
print("\nFiltered dataset (Price > $500,000, sorted by Horsepower):")
print(filtered_df)

# Export cleaned and transformed dataset to CSV
filtered_df.to_csv('cleaned_sports_car_prices.csv', index=False)

print("\nData cleaned, analyzed, and exported to 'cleaned_sports_car_prices.csv'")


