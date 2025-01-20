import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load the cleaned and engineered dataset
file_path = "engineered_dataset.csv"
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}. Ensure the file exists.")
    exit()

# Step 2: Define feature columns and target variable
features = [
    'Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast',
    'Price', 'Discount', 'Category_Electronics', 'Category_Furniture',
    'Category_Groceries', 'Category_Toys', 'Revenue', 'Cumulative Sales'
]
target = 'Units Sold'

# Ensure no infinite or NaN values
if not np.isfinite(data[features]).all().all():
    print("Warning: Infinite or NaN values detected in feature columns.")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Step 3: Split the data into training and testing sets
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} rows")
print(f"Testing set size: {len(X_test)} rows")

# Step 4: Train a Random Forest Regressor model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Step 6: Save the model
model_file = "demand_forecasting_model.pkl"
joblib.dump(model, model_file)
print(f"Model saved as {model_file}")

# Step 7: Example prediction
new_data = X_test.iloc[:5]  
predictions = model.predict(new_data)

print("Predictions for new data:")
print(predictions)

#Save predictions for further analysis
predicted_df = pd.DataFrame({
    "Actual": y_test.iloc[:5].values,
    "Predicted": predictions
})
predicted_df.to_csv("predicted_vs_actual.csv", index=False)
print("Predictions saved to predicted_vs_actual.csv")
