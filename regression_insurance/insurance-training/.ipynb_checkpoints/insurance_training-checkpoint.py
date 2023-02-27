
# Import libraries
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Get the experiment run context
run = Run.get_context()

# load the dataset
print("Loading Data...")
df = pd.read_csv('insurance.csv')

# Separate features and labels
X = df[['age','bmi','children']].values
y = df['charges'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and Train the Model
model = LinearRegression().fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# calculate r2
r2 = model.score(X_test, y_test)
run.log('r2', round(r2,2))

# calculate rmse
rmse = mean_squared_error(y_test, y_pred)
run.log('RMSE', np.float(rmse))

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/insurance_model.pkl')

run.complete()
