# Bitcoin Price Prediction
All data can be found in the Data folder. The final dataset (after merging, preprocessing, feature selection) is `final_data.csv`. All models train on data from this file.

# Code Architecture

* Data_Preprocessing

Generates the final dataset by merging the other CSV files found in the Data folder. Performs basic preprocessing, imputation to remove NAs, and feature selection before exporting the data in a general format for all models.

Note that `merged.csv` represents the combined datasets of Market data, Futures data, Onchain data, and Google Trends data. Those datasets were simply combined (inner join) with no imputation, so we see some null values from there. Our team decided against uploading the original files due to the large number of CSV files pre-merge.

* Model_ARIMA

Application of ARIMA model on final dataset.

* Model_Prophet_Rolling

Application of Prophet model on final dataset. Rolling and non-rolling versions were both tested, but only rolling model is included here.

* Model_LSTM

Application of LSTM model on final dataset.

* Model_GradientBoost

Application of Gradient Boosted Regression model on final dataset. Our chosen model.

* SwingTradingSimulator

Short Python script that simulates trades based on a simple swing trading strategy based on the output of a prediction model, and plots the results for viewing.
