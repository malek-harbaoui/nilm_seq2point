# NILM Energy Disaggregation using Seq2Point Model
This project focuses on **Non-Intrusive Load Monitoring (NILM)** using the Seq2Point deep learning model. The Seq2Point architecture predicts the power consumption of individual appliances from aggregated energy consumption data.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Outputs](#outputs)
- [Explanation](#explanation)
- [Results](#results)

## Overview
The **Seq2Point model** uses convolutional layers to extract temporal patterns from aggregated energy data and predict the power consumption of individual appliances. It is designed for multi-output tasks, with one output per appliance.

## Dataset
The dataset file `all_buildings_data.csv` contains:
- `aggregate`: The aggregated energy consumption (input to the model).
- Remaining columns: Individual appliance consumption values (targets).

## Requirements
Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_name>
```
2. Ensure the dataset file `all_buildings_data.csv` is in the project root directory.
3. Run the script to train the model:
```bash
python <script_name>.py
```
4. Predictions can be made using the `predict_appliance_consumption` function, passing an aggregate energy window of size 60.
## Outputs
Upon successful execution, the following files and images will be generated:

- **Model Files**:
  - `energy_disaggregation_model.keras`: Saved model in Keras format.
  - `energy_disaggregation_model.h5`: Saved model in HDF5 format.
  - `energy_disaggregation_model.pk1`: Pickled model file.

- **Loss and MAE Plots**:
  - `loss_over_epochs.png`: Training and validation loss over epochs.
  - `mae_over_epochs.png`: Training and validation MAE over epochs.

## Explanation
### Model Architecture
The Seq2Point model uses:
- **Conv1D layers**: Extract temporal features from the input data.
- **Flatten layer**: Flatten the extracted features.
- **Dense layers**: Map features to appliance consumption values.

### Training Process
1. **Data Preprocessing**:
   - Normalize inputs and targets using `MinMaxScaler`.
   - Create windows of data for temporal modeling (window size = 60).

2. **Model Training**:
   - Train with a batch size of 64 for 10 epochs, using 80% of the training data.
   - Evaluate the model using MAE, MSE, and R-squared metrics.

3. **Prediction**:
   - Predict appliance consumption values for a given aggregate window.

## Results
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R²)

- **Predicted Appliance Consumption**:
  The model predicts normalized and real-scale consumption values for each appliance, mapped using their names.

## Gitignore
The repository includes a `.gitignore` file to exclude unnecessary files and folders, such as virtual environments and temporary files.
---

Feel free to explore the results and adapt the model to your NILM tasks. 
