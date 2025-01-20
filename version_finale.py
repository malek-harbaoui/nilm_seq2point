import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump


# Load and preprocess the dataset
df = pd.read_csv('data/all_buildings_data.csv')

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'building'])

# Extract features (aggregate power) and targets (appliance power consumption)
X = df['aggregate'].values.reshape(-1, 1)
y = df.drop(columns=['aggregate']).values  # All appliance columns as targets

# Standardize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Standardize targets
y = scaler.fit_transform(y)

# Create windows of data
def create_windows(data, targets, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

window_size = 60  # 60 points of time in our window

# Prepare the data
X, y = create_windows(X, y, window_size)
# Reshape X for Conv1D input
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the multi-output Seq2Point model
def multi_output_seq2point_model(input_shape, num_outputs):
    model = Sequential()
    model.add(Conv1D(16, 10, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv1D(32, 7, activation='relu', padding='same'))
    model.add(Conv1D(64, 5, activation='relu', padding='same'))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_outputs))  
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model



# Create the model
num_appliances = y_train.shape[1]
model = multi_output_seq2point_model((X_train.shape[1], 1), num_appliances)
# Model summary
print (model.summary())

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f'Model Mean Absolute Error: {mae}')

# Loss Over epochs plot 
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Save Loss plot
plt.savefig('loss_over_epochs.png')  
plt.clf()  

# MAE Over epochs Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Training Accuracy')
plt.plot(history.history['val_mae'], label='Validation Accuracy')
plt.title('mae Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('mae')
plt.legend()
plt.show()
# Save MAE over epochs plot
plt.savefig('mae_over_epochs.png')  
plt.clf()  

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Save model
"""
model.save('energy_disaggregation_model.h5')
model.save('energy_disaggregation_model.keras')
dump(model, 'energy_disaggregation_model.pk1')
"""

# List of appliance names
appliances = [
    'laptop computer', 'television', 'light', 'HTPC', 'food processor', 'toasted sandwich maker',
    'toaster', 'microwave', 'computer monitor', 'audio system', 'audio amplifier', 'broadband router',
    'ethernet switch', 'USB hub', 'tablet computer charger', 'radio', 'wireless phone charger', 'mobile phone charger',
    'coffee maker', 'computer', 'external hard disk', 'desktop computer', 'printer', 'immersion heater',
    'security alarm', 'projector', 'server computer', 'running machine', 'network attached storage', 'fridge',
    'air conditioner'
]


def predict_appliance_consumption(aggregate_window, new_scaler, window_size):
    # Convert to array
    aggregate_window = np.array(aggregate_window).reshape(-1, 1)
    # Normalise aggregate_window
    aggregate_window = new_scaler.fit_transform(aggregate_window)
    # Reshape dimensions to fit the model input (1, window_size, 1)
    aggregate_window = aggregate_window.reshape((1, window_size, 1))
    # Predictions using the model
    predictions = model.predict(aggregate_window)
    # Clip to turn negative values to 0
    predictions = np.clip(predictions, 0, None)
    # reverse Minmax_scaler
    real_predictions = new_scaler.inverse_transform(predictions)
    return predictions, real_predictions
    
# Test model on new data
# Exemple of aggregate_window size 60
aggregate_window = [
    100, 52.31, 1.0, 135.28, 2.0, 1.0, 1.0, 0.0, 1.0, 4.89, 51.10, 0.0, 6.0,
    51.44, 51.44, 50.52, 135.28, 1.0, 0.0, 0.0, 13.88, 13.88, 57.26, 50.93,
    102.20, 102.20, 135.28, 135.28, 135.28, 135.28, 0.0, 135.28, 99.8, 98.7,
    102.3, 101.2, 100.5, 102.0, 105.0, 99.9, 98.4, 102.1, 101.3, 100.4, 100.7,
    102.8, 101.5, 100.3, 101.7, 100.9, 102.2, 100.6, 99.5, 101.4, 100.2, 100.1,
    99.6, 98.5, 100.0, 99.7
]


# predictions based on aggregate_window
predictions , real_predictions= predict_appliance_consumption(aggregate_window, MinMaxScaler(),60)

# Map appliance names to their predicted consumption values
predicted_consumption_scaled = dict(zip(appliances, predictions[0]))
predicted_consumption_real = dict(zip(appliances, real_predictions[0]))


# print real and scaled prediction results
print("Predicted appliance consumption (real scale):")
for appliance, consumption in predicted_consumption_real.items():
    print(f"{appliance}: {consumption:.2f}")

print("\nPredicted appliance consumption (normalized scale):")
for appliance, consumption in predicted_consumption_scaled.items():
    print(f"{appliance}: {consumption:.4f}")











