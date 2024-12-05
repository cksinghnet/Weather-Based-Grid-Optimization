# Forecast Energy Demand

Forecasting Energy Demand Using Historical Weather and Grid Consumption Data

## Project Title
Forecasting Energy Demand Using Historical Weather and Grid Consumption Data

## Team Members
- Chandan Kumar Singh
- Monika Tyagi
- Mukesh Kumar Yadav
- Rishabh Mehrotra

## Problem Statement
Accurate forecasting of energy demand is essential for effective grid management, reducing energy wastage, and ensuring demand-supply balance. Energy consumption is heavily influenced by weather conditions such as temperature, humidity, and wind speed. This project aims to leverage historical weather data and grid consumption patterns to predict future energy demand, enabling improved resource allocation and sustainable energy management.

## Dataset Summary
The dataset used for this project includes:
- **Historical Weather Data**: Features such as temperature, humidity, wind speed, and cloudiness.
- **Grid Consumption Data**: Historical energy consumption records at regular intervals.

The dataset contains **[number of records]** records and **[number of features]** features. The data is sourced from **[source of the dataset]**.

## Approach and Methods
To forecast energy demand, we employed the following methodologies:

- **Data Preprocessing**: Aligned timestamps, handled missing values, and scaled features.
- **Feature Engineering**:
  - Temporal features: Extracted hour, day of the week, month, and season.
  - Lagged features: Incorporated past consumption data to capture temporal dependencies.
  - Weather features: Included temperature, humidity, wind speed, and other relevant parameters.
- **Machine Learning Models**:
  - Implemented regression models including Random Forest Regressor and Gradient Boosting Machines.
  - Hyperparameter tuning using grid search to optimize model performance.

## Key Features of the Project
- **Scalability**: Utilizes PySpark to process large datasets efficiently.
- **Integration of Weather Data**: Combines historical weather and grid consumption data for better prediction accuracy.
- **Evaluation Metrics**: Metrics like RMSE and MAPE are used to evaluate model performance.
- **Interpretability**: Identifies key drivers of energy demand through feature importance analysis.

## Tools and Technologies
- PySpark for distributed data processing
- Spark MLlib for machine learning
- Python for scripting and visualization
- Tools for data ingestion and preprocessing

## Results and Insights
- The best-performing model achieved an RMSE of **[value]** and a MAPE of **[value]%**.
- Feature importance analysis revealed that **[top key features]** were the most significant drivers of energy demand.
- Seasonal patterns and weather conditions showed strong correlations with energy consumption.

## Future Work
- Incorporate real-time weather data streams for live energy demand predictions.
- Explore deep learning models such as LSTMs and GRUs to capture complex temporal dependencies.
- Expand the dataset to include additional cities or regions for a broader application.
- Integrate renewable energy sources into demand predictions for sustainable energy management.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/[username]/forecast-energy-demand.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the PySpark scripts to preprocess data, train models, and evaluate results.
   ```bash
   spark-submit train_model.py
   ```

## Acknowledgements
We thank **[organization or individuals]** for providing the datasets and tools required for this project. Special thanks to the open-source community for the frameworks and libraries used.

---
