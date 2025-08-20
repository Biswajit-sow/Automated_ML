import pandas as pd
from flaml.automl import AutoML
import numpy as np

# 1. Create a sample time series dataset
print("Preparing data...")
data = {
    'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100)),
    'value': pd.Series(range(100)) + np.random.rand(100) * 10
}
df = pd.DataFrame(data)

# Define the training and test split point
train_size = 90

# 2. Split the data correctly for FLAML's fit method
# X_train MUST be a 2D DataFrame (hence the double brackets [['date']])
X_train = df[['date']][:train_size]
# y_train is the 1D Series of values to be predicted
y_train = df['value'][:train_size]

# 3. Initialize the AutoML instance
automl = AutoML()

# 4. Define the settings for the forecast task
automl_settings = {
    "time_budget": 60,
    "metric": 'mape',
    "task": 'forecast',
    "log_file_name": "forecast.log",
}

# 5. Train the model with the correctly structured X_train and y_train
print("--- Starting AutoML Training for Time Series ---")
automl.fit(
    X_train=X_train,             # DataFrame with timestamps
    y_train=y_train,             # Series with target values
    period=10,                   # The number of future periods to forecast
    **automl_settings
)
print("--- AutoML Training Finished ---\n")

# 6. Display the results
print("--- Best Model Found ---")
print(f"Best learner: {automl.best_estimator}")
print(f"Best configuration: {automl.best_config}\n")

# 7. Make a prediction for the future
print("--- Forecasting Next 10 Periods ---")
# To predict, we must provide an X_test DataFrame with the future timestamps
future_dates = df[['date']][train_size:]
predictions = automl.predict(future_dates)

# Print the results in a nice format
forecast_df = pd.DataFrame({'date': future_dates['date'], 'predicted_value': predictions})
print(forecast_df)