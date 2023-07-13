import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv("D:\ML\p1\car data.csv")
# inspecting the first 5 rows of dataframe
print(car_dataset.head())
# checking the no. of rows and columns
print(car_dataset.shape)

#Checking for missing values
print("\nMissing Values:\n",car_dataset.isnull().sum())
# no missing values found in each column


# encoding fuel type column
car_dataset.replace({"Fuel_Type": {"Petrol": 0, "Diesel": 1, "CNG": 2}}, inplace=True)

# encoding seller type column
car_dataset.replace({"Seller_Type": {"Dealer": 0, "Individual": 1}}, inplace=True)

# encoding transmission column
car_dataset.replace({"Transmission": {"Manual": 0, "Automatic": 1}}, inplace=True)
# checking dataframe for encoded values
print("\nAfter Converting String to Numerical Data:\n")
print(car_dataset.head())

# convert Year to Age of each cars
Age = 2023 - car_dataset.Year

car_dataset.insert(0, "Age", Age)
car_dataset.drop('Year', axis=1, inplace=True)
print('\nUpdated Dataset With Age Column:\n')
print(car_dataset)
print('\nColumn Names:\n')
print(car_dataset.columns)

# Splitting the Data and Target

X = car_dataset.drop(["Car_Name", "Selling_Price"], axis=1)
Y = car_dataset["Selling_Price"]
print('\nData:\n')
print(X)
print('\nTarget:')
print(Y)

# Splitting Training and Test Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=2)
# loading the linear regression model
lin_reg_model = LinearRegression()

lin_reg_model.fit(X_train, Y_train)
# prediction on training data
training_data_prediction = lin_reg_model.predict(X_train)
# r squared error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error:", error_score)
# --this will tell us how close the values are
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices Vs Predicted Prices(Training Data)")
plt.show()
# the values that are predicted by our ML model is very close to the original sold price

# prediction on test data
test_data_prediction = lin_reg_model.predict(X_test)

# r squared error -- this time for test data
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error:", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices Vs Predicted Prices(Test Data)")
plt.show()
