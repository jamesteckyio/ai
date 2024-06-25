import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

## columns name of the dataset
feature_names = [
    'gdp',
    'total_population',
    'inflation_rate',
    'working_age',
    'immigrant_population',
]
label_name = 'average_property_price_of_housing_estates'
column_names = feature_names + [label_name]

df = pd.read_csv('./dataset/property_prices.csv', 
                 skiprows=1,
                 usecols=[1,2,3,4,5,6],
                 names=column_names)
y = df['average_property_price_of_housing_estates']
df = df.drop(['average_property_price_of_housing_estates'], axis=1)
X = df[feature_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Use the model to make a prediction
# new_data = np.array([[28.7, 90, 1010.3, 10.1]])
new_data = pd.DataFrame([[3598.39, 7.41, 1.88, 5.16,2]], columns=feature_names)

rainfall_prediction = model.predict(new_data)
print("Predicted rainfall:", rainfall_prediction[0])