import numpy as np # Numerical computing library
import pandas as pd # Data manipulation library
import matplotlib.pyplot as plt # Data visualization library
from sklearn.model_selection import train_test_split # Function to split data into training and testing sets
from sklearn.linear_model import LinearRegression # Linear Regression model
from sklearn.metrics import mean_squared_error, r2_score # Evaluation metrics

# Load the dataset
data = pd.read_csv('salary_prediction_data.csv')

# Explore the dataset
print(data.head())

# Preprocess the data
data = data.dropna()  # Remove missing values

# Define features and target variable
X = data[['Experience']] # 'Experience' is our independent variable (input)
y = data['Salary'] # 'Salary' is our dependent variable (output)

# Split the data into training and testing sets
# We divide the data so the model learns from one part (train)
# and is tested on unseen data (test) to evaluate its performance
# test_size=0.2 means 20% of data will be used for testing
# random_state ensures reproducibility (same split every time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
# Mean Squared Error - measures how far predictions are from actual values
# Lower MSE means better accuracy
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
# R-squared - measures how well the model explains the variance in the data
# Values closer to 1.0 indicate a stronger correlation (better fit)
r2 = r2_score(y_test, y_pred)
print("R-squared: ", r2)

# Visualize the results
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Salary') # Actual data points
plt.scatter(X_test, y_pred, color='red', label='Predicted Salary') # Predicted data points
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salary')
plt.legend()
plt.show()

"""
Summary: This code implements a simple Linear Regression model to predict employee earnings
based on years of experience. While this simple model provides useful baseline insights, it does
not account for other factors such as education, job title or location, which could further 
improve prediction accuracy.

Steps:
1. Loaded and explored the dataset using Pandas.
2. Cleaned the data by removing missing values.
3. Split the dataset into training and testing sets.
4. Trained a Linear Regression model using the training data.
5. Predicted salaries using the testing data.
6. Evaluated the model using Mean Squared Error and R-squared.
7. Visualized the actual vs predicted salaries to observe model accuracy.
"""