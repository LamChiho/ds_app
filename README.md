# ds_app
## Introduction
This is a simple Streamlit application designed to assist with basic data preprocessing and machine learning modeling tasks. It allows users to upload a CSV file, perform data cleaning operations such as handling missing values, select features and target columns, choose a machine learning model, and evaluate the model's performance.

## Installation
To run the application locally, follow these steps:

Clone or download this repository to your local machine.
Install the required dependencies by running:
### pip install -r requirements.txt

## Usage
Run the application by executing the following command in your terminal:
### streamlit run service1.py
Once the application is running, you will be prompted to upload a CSV file containing your dataset.
After uploading the file, you can view the first few rows of the data and its descriptive statistics.
Choose whether to fill missing values and select the target column and any columns to exclude from the analysis.
Select the mission type (regression or classification) and the model type (XGBoost, LightGBM, or CatBoost).
Adjust the test data ratio using the slider.
Once all options are set, click "Ready?" to train and evaluate the selected model.
The application will display the evaluation metrics (MSE for regression, F1 score and accuracy for classification) for the trained model.

## Dependencies
streamlit
pandas
numpy
scipy
altair
catboost
xgboost
lightgbm
scikit-learn
