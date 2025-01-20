import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl

# Step 1: Load and Clean Data
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)

    # Drop unnecessary columns
    data = data.drop(['Store ID', 'Region', 'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality'], axis=1)

    # Handle missing values
    data.fillna(method='bfill', inplace=True)

    # Normalize numerical columns
    for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    print("Data cleaned and ready for processing!")
    return data

# Step 2: Feature Engineering
def feature_engineering(data):
    # One-hot encoding for categories
    data = pd.get_dummies(data, columns=['Category'], prefix='Category', drop_first=True)

    # Add cumulative sales and revenue
    data['Cumulative Sales'] = data['Units Sold'].cumsum()
    data['Revenue'] = data['Units Sold'] * data['Price']

    # Save engineered data
    data.to_csv('engineered_dataset.csv', index=False)
    print("Feature engineering completed. Saved as 'engineered_dataset.csv'.")
    return data

# Step 3: Train Demand Forecasting Model
def train_demand_forecasting_model(data):
    # Define features (X) and target (y)
    X = data.drop(['Date', 'Product ID', 'Units Sold'], axis=1)
    y = data['Units Sold']

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:\nMean Squared Error (MSE): {mse:.4f}\nR^2 Score: {r2:.4f}")

    # Save model
    joblib.dump(model, 'demand_forecasting_model.pkl')
    print("Model saved as 'demand_forecasting_model.pkl'.")

    # Save predictions vs actual
    predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions.to_csv('predicted_vs_actual.csv', index=False)
    print("Predictions saved to 'predicted_vs_actual.csv'.")

    return model, X_test

# Step 4: Visualize Predictions
def visualize_predictions():
    data = pd.read_csv('predicted_vs_actual.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(data['Actual'], label='Actual', marker='o')
    plt.plot(data['Predicted'], label='Predicted', marker='x')
    plt.title('Predicted vs Actual Demand')
    plt.xlabel('Index')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid()
    plt.savefig('predicted_vs_actual_plot.png')
    plt.show()
    print("Visualization saved as 'predicted_vs_actual_plot.png'.")

# Step 5: Export Data for Tableau/Excel Integration
def export_data_to_excel():
    data = pd.read_csv('predicted_vs_actual.csv')
    excel_file = 'demand_forecasting_results.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name='Predicted vs Actual')
    print(f"Data saved to {excel_file} for Tableau/Excel integration.")

# Step 6: Stock-Level Optimization (Placeholder)
def stock_level_optimization():
    print("Placeholder for stock-level optimization (e.g., EOQ, ROP, safety stock).")
    # Add logic for EOQ, reorder points, etc., based on demand forecasting results.

# Step 7: Main Function
def main():
    # Load and clean data
    file_path = 'retail_store_inventory.csv'
    data = load_and_clean_data(file_path)

    # Perform feature engineering
    engineered_data = feature_engineering(data)

    # Train demand forecasting model
    model, X_test = train_demand_forecasting_model(engineered_data)

    # Visualize predictions
    visualize_predictions()

    # Export data to Excel
    export_data_to_excel()

    # Stock optimization (placeholder)
    stock_level_optimization()

# Run the main function
if __name__ == '__main__':
    main()

