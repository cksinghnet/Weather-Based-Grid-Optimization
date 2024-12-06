from flask import Blueprint, render_template, request
from flask_cors import CORS
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import Row

main = Blueprint('main', __name__)
CORS(main)

# Step 1: Start Spark Session (if not already running)
spark = SparkSession.builder.appName("Grid Consumption Forecasting - Prediction").getOrCreate()

# Step 2: Load the saved model
loaded_model = RandomForestRegressionModel.load("../spark_rf_model")

def generate_weather_features():
    features = {
        'Temperature_C': np.random.uniform(15, 30),
        'Feels_Like_C': np.random.uniform(15, 30),
        'Humidity_percent': np.random.uniform(40, 90),
        'Pressure_hPa': np.random.uniform(1000, 1025),
        'Wind_Speed_ms': np.random.uniform(0, 10),
        'Cloudiness_percent': np.random.uniform(0, 100),
        'Rain_1h_mm': np.random.uniform(0, 5),
        'Hour': 12,
        'DayOfWeek': 6,
        'Month': 12,
        **{f"Lag_{lag_val}": np.random.uniform(50, 150) for lag_val in range(1, 25)}
    }
    return features

def predict_consumption(city, date, features):
    # Step 3: Create a Row object for the input data
    random_data = [
        Row(
            City=city,
            Date=date,
            **features
        )
    ]

    # Step 4: Create DataFrame from input data
    future_data_df = spark.createDataFrame(random_data)

    # Step 5: Assemble the features (using the same columns as in training)
    feature_columns = [
        "Temperature_C", "Feels_Like_C", "Humidity_percent", "Pressure_hPa",
        "Wind_Speed_ms", "Cloudiness_percent", "Rain_1h_mm", "Hour", "DayOfWeek", "Month"
    ] + [f"Lag_{lag_val}" for lag_val in range(1, 25)]

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    future_data = assembler.transform(future_data_df).select("features", "Date", "City")

    # Step 6: Predict future consumption
    predictions = loaded_model.transform(future_data)

    # Step 7: Extract prediction
    prediction_result = predictions.select("Date", "City", "prediction").collect()[0]["prediction"]
    return prediction_result

@main.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    if request.method == 'POST':
        city = request.form.get('city')
        date = request.form.get('date')

        # Generate weather features using the function
        weather_features = generate_weather_features()

        # Predict consumption using the generated features
        prediction = predict_consumption(city, date, weather_features)

    return render_template('index.html', prediction=prediction)
