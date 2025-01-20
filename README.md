# Demand Forecasting and Inventory Optimization

## **Overview**
This project combines **demand forecasting** with an **inventory management process** to help businesses optimize stock levels, reduce inventory costs, and ensure data-driven decision-making. The system predicts demand for products using machine learning and automates key inventory management tasks, such as reorder point calculations and economic order quantity (EOQ).

## **Features**
1. **Demand Forecasting**:
   - Uses a machine learning model (Random Forest Regressor) to predict product demand.
   - Incorporates historical sales data, discounts, pricing, and other features.
   - Achieves high accuracy with low Mean Squared Error (MSE) and high R² score.

2. **Inventory Management**:
   - Automates inventory processes:
     - Calculates **Economic Order Quantity (EOQ)**.
     - Determines **Reorder Points** based on lead time and safety stock.
   - Provides recommendations for stock levels to avoid understocking or overstocking.

3. **Data Visualization**:
   - Generates charts comparing **predicted vs. actual demand** for validation.
   - Saves results in a Tableau/Excel-compatible format for reporting.

4. **Integration-Ready**:
   - Outputs predictions and inventory recommendations to `.csv` and `.xlsx` files.
   - Can be extended for direct Tableau integration using APIs.

## **System Flow**
1. **Data Preprocessing**:
   - Cleans raw data and applies feature engineering (e.g., category encoding, cumulative sales).
   - Handles missing or invalid values gracefully.

2. **Demand Forecasting**:
   - Splits data into training and testing sets.
   - Trains the Random Forest model to predict product demand.
   - Evaluates performance and saves predictions.

3. **Inventory Optimization**:
   - Implements placeholder functions for EOQ and reorder point logic.
   - Outputs recommendations for optimal stock management.

4. **Export and Visualization**:
   - Saves predictions and inventory metrics for further analysis.
   - Creates visualizations to evaluate model performance.


