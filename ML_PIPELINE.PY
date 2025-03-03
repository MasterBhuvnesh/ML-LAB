import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# 📌 Step 1: Create the dataset
data = pd.DataFrame({
    'Brand': ['Ford', 'BMW', 'Tesla', 'Ford', 'BMW'],
    'Fuel Type': ['Petrol', 'Diesel', 'Electric', 'Diesel', 'Petrol'],
    'Mileage': [18, 20, 25, 19, 17],
    'Price': [20000, 30000, 50000, 22000, 28000]  # Target Variable
})

# 📌 Step 2: Split dataset into Features (X) and Target (y)
X = data.drop(columns=['Price'])  # Features
y = data['Price']  # Target Variable

# 📌 Step 3: Define Preprocessing Steps
categorical_features = ['Brand', 'Fuel Type']
numerical_features = ['Mileage']

# OneHotEncoder for categorical data & StandardScaler for numerical data
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
    ('scaler', StandardScaler(), numerical_features)
])

# 📌 Step 4: Build the ML Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())  # Linear Regression Model
])

# 📌 Step 5: Train the Model
pipeline.fit(X, y)

# 📌 Step 6: Make a Prediction for a New Car
new_car = pd.DataFrame({'Brand': ['Tesla'], 'Fuel Type': ['Electric'], 'Mileage': [24]})
predicted_price = pipeline.predict(new_car)

print(f"Predicted Price: ${predicted_price[0]:,.2f}")
