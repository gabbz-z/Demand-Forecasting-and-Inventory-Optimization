import pandas as pd

# Load the dataset
file_path = "cleaned_dataset.csv"  
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Check the file path and try again.")
    exit()

# Ensure column names match
print("Columns in dataset:", df.columns)

# Feature Engineering Steps
try:
    # Feature 1: Revenue
    df['Revenue'] = df['Units Sold'] * df['Price']

    # Feature 2: Discounted Price
    df['Discounted Price'] = df['Price'] - df['Discount']

    # Feature 3: Sell-Through Rate
    df['Sell-Through Rate'] = df['Units Sold'] / df['Inventory Level']

    # Feature 4: Cumulative Metrics
    df['Cumulative Sales'] = df.groupby('Product ID')['Units Sold'].cumsum()

    # Feature 5: Rolling Average Sales
    df['Rolling Average Sales'] = df.groupby('Product ID')['Units Sold'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)

    # Feature 6: Lag Features
    df['Lag_1_Units_Sold'] = df.groupby('Product ID')['Units Sold'].shift(1)
    df['Lag_7_Units_Sold'] = df.groupby('Product ID')['Units Sold'].shift(7)

    # Feature 7: Time-Based Features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # Feature 8: One-Hot Encoding for 'Category'
    if 'Category' in df.columns:
        df = pd.get_dummies(df, columns=['Category'], prefix='Category', drop_first=True)
    else:
        print("Category column not found; skipping one-hot encoding.")

    # Handle missing values
    df.fillna(0, inplace=True)

    # Save the engineered dataset
    df.to_csv("engineered_dataset.csv", index=False)
    print("Feature engineering completed. Saved as 'engineered_dataset.csv'.")

except KeyError as e:
    print(f"Error during feature engineering: {e}")
    print("Check if all required columns are present in the dataset.")
