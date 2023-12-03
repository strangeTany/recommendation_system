import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname,'../models')
test_data_path = os.path.join(dirname, 'data')

model = joblib.load(os.path.join(model_path, "best_xgb_model.pkl"))

test_x = pd.read_csv(os.path.join(test_data_path, 'test_x.csv'), sep='|')
test_y= pd.read_csv(os.path.join(test_data_path, 'test_y.csv'), sep='|')

predictions = model.predict(test_x)

# Calculate evaluation metrics
mse = mean_squared_error(test_y, predictions)
rmse = mean_squared_error(test_y, predictions, squared=False)
mae = mean_absolute_error(test_y, predictions)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)