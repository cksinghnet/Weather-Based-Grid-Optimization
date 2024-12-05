from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, month, dayofweek, hour, unix_timestamp
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Grid Consumption Forecasting") \
    .getOrCreate()

# Load data
weather_file = "weather_data.csv"
consumption_file = "grid_consumption.csv"

weather_data = spark.read.csv(weather_file, header=True, inferSchema=True)
consumption_data = spark.read.csv(consumption_file, header=True, inferSchema=True)

# Convert timestamps to Spark format
weather_data = weather_data.withColumn("Date", col("Date").cast("timestamp"))
consumption_data = consumption_data.withColumn("Date", col("Date").cast("timestamp"))

# Round weather data to hourly timestamps
weather_data = weather_data.withColumn("Date", (unix_timestamp("Date") / 3600).cast("int") * 3600)

# Join the datasets on City and Date
merged_data = consumption_data.join(weather_data, on=["City", "Date"], how="inner")

# Feature Engineering
# Add temporal features
merged_data = merged_data \
    .withColumn("Hour", hour(col("Date"))) \
    .withColumn("DayOfWeek", dayofweek(col("Date"))) \
    .withColumn("Month", month(col("Date")))

# Add lagged consumption features
window_spec = Window.partitionBy("City").orderBy("Date")
for lag_val in range(1, 25):  # Lags for the past 24 hours
    merged_data = merged_data.withColumn(f"Lag_{lag_val}", lag("Consumption (MW)", lag_val).over(window_spec))

# Drop rows with missing lagged values
merged_data = merged_data.dropna()

# Prepare data for modeling
feature_columns = [
    "Temperature (C)", "Feels Like (C)", "Humidity (%)", "Pressure (hPa)",
    "Wind Speed (m/s)", "Cloudiness (%)", "Rain (1h mm)", "Hour", "DayOfWeek", "Month"
] + [f"Lag_{lag_val}" for lag_val in range(1, 25)]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(merged_data).select("features", col("Consumption (MW)").alias("label"))

# Split data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train a Random Forest Regressor
rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=100, maxDepth=10)
model = rf.fit(train_data)

# Evaluate the model
test_predictions = model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(test_predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the model for future use
model.save("grid_consumption_rf_model")
