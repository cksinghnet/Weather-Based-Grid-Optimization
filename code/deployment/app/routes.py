from flask import Blueprint, render_template, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import Row

main = Blueprint('main', __name__)
CORS(main)

# Step 1: Start Spark Session (if not already running)
spark = SparkSession.builder.appName(
    "Grid Consumption Forecasting - Prediction").getOrCreate()

# Step 2: Load the saved model
loaded_model = RandomForestRegressionModel.load("../spark_rf_model")


def predict_consumption(city, date, features):
    # Add missing Lag_* features with default values if not present
    for lag in range(1, 25):
        if f"Lag_{lag}" not in features:
            # Default value (adjust based on use case)
            features[f"Lag_{lag}"] = 0

    # Create Row object for input data
    random_data = [
        Row(
            City=city,
            Date=date,
            **features
        )
    ]

    # Create DataFrame from input data
    future_data_df = spark.createDataFrame(random_data)

    # Assemble the features (using the same columns as in training)
    feature_columns = [
        "Temperature_C", "Feels_Like_C", "Humidity_percent", "Pressure_hPa",
        "Wind_Speed_ms", "Cloudiness_percent", "Rain_1h_mm", "Hour", "DayOfWeek", "Month"
    ] + [f"Lag_{lag_val}" for lag_val in range(1, 25)]

    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="features")
    future_data = assembler.transform(
        future_data_df).select("features", "Date", "City")

    # Predict future consumption
    predictions = loaded_model.transform(future_data)

    # Extract prediction
    prediction_result = predictions.select(
        "Date", "City", "prediction").collect()[0]["prediction"]
    return prediction_result


def read_weather_data_from_csv(file_path, target_date):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    target_date = pd.to_datetime(target_date)
    filtered_df = df[df['Date'] == target_date]

    weather_data = []
    for _, row in filtered_df.iterrows():
        weather_data.append({
            "City": row["City"],
            "Date": row["Date"].strftime("%Y-%m-%d %H:%M:%S"),
            "Temperature (C)": row["Temperature (C)"],
            "Feels Like (C)": row["Feels Like (C)"],
            "Humidity (%)": row["Humidity (%)"],
            "Pressure (hPa)": row["Pressure (hPa)"],
            "Wind Speed (m/s)": row["Wind Speed (m/s)"],
            "Cloudiness (%)": row["Cloudiness (%)"],
            "Rain (1h mm)": row["Rain (1h mm)"]
        })
    return weather_data


@main.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        city = request.form.get('city')
        date = request.form.get('date')

        # Read weather data from CSV based on the date passed
        weather_data = read_weather_data_from_csv(
            "../../dataset/future_predicted_weather_data.csv", date)

        if weather_data:
            weather_features = weather_data[0]
            # Rename keys to match feature requirements
            features = {
                'Temperature_C': weather_features["Temperature (C)"],
                'Feels_Like_C': weather_features["Feels Like (C)"],
                'Humidity_percent': weather_features["Humidity (%)"],
                'Pressure_hPa': weather_features["Pressure (hPa)"],
                'Wind_Speed_ms': weather_features["Wind Speed (m/s)"],
                'Cloudiness_percent': weather_features["Cloudiness (%)"],
                'Rain_1h_mm': weather_features["Rain (1h mm)"],
                'Hour': pd.to_datetime(weather_features["Date"]).hour,
                'DayOfWeek': pd.to_datetime(weather_features["Date"]).dayofweek,
                'Month': pd.to_datetime(weather_features["Date"]).month
            }

            # Predict consumption using the features from CSV
            prediction = predict_consumption(city, date, features)

    return render_template('index.html', prediction=prediction)
