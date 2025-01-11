# Ethereum Price Prediction Using Sentiment Analysis and Machine Learning

This repository contains two Jupyter Notebooks that collectively demonstrate a unique approach to Ethereum price prediction. By combining sentiment analysis of cryptocurrency news with historical price data from Yahoo Finance, the project builds and refines machine learning models to predict Ethereum price trends.

## Features

1. **Sentiment Analysis**:
   - Utilizes the `TextBlob` library for sentiment analysis instead of a pre-trained Hugging Face model.
   - Analyzes cryptocurrency-related text data for sentiment polarity, resulting in a custom dataset for further analysis.
   - Provides more interpretable and effective sentiment scores compared to traditional models.

2. **Machine Learning for Price Prediction**:
   - Integrates sentiment data with historical Ethereum price data from Yahoo Finance.
   - Uses a **Random Forest Classifier** as the baseline model.
   - Applies advanced techniques such as backtesting and **XGBoost** to improve prediction accuracy.

3. **Backtesting**:
   - Evaluates the performance of the prediction model using past data to ensure robustness.

## Notebooks Overview

### 1. `eth_sentiment.ipynb`
   - Performs sentiment analysis on cryptocurrency-related data using `TextBlob`.
   - Generates a CSV file containing the sentiment scores.
   - This step is crucial for creating the sentiment analysis dataset used in the next notebook.

### 2. `Eth_data.ipynb`
   - Merges the sentiment analysis data with historical price data from Yahoo Finance.
   - Trains a machine learning model (Random Forest Classifier) using the combined dataset.
   - Implements backtesting to validate the model's performance.
   - Enhances the model using XGBoost for improved accuracy.

## Steps for Execution

1. **Run Sentiment Analysis Notebook**:
   - Open and execute `eth_sentiment.ipynb`.
   - Perform sentiment analysis on the cryptocurrency-related data using `TextBlob`.
   - Generate a CSV file (`sentiment_data.csv`) that contains sentiment scores for each data entry.

2. **Train and Evaluate Prediction Model**:
   - Open and execute `Eth_data.ipynb`.
   - Load the sentiment analysis data and merge it with historical Ethereum price data from Yahoo Finance.
   - Train a **Random Forest Classifier** using the combined dataset.
   - Use backtesting to validate model performance and fine-tune parameters.
   - Implement **XGBoost** to further enhance the model's predictive accuracy.

3. **Model Validation**:
   - Compare predicted outcomes with actual historical prices.
   - Evaluate performance metrics such as accuracy, precision, recall, and F1-score.

## Key Learnings

- **Integration of Sentiment Analysis**:
  - Sentiment analysis data significantly influences Ethereum price prediction.
  - The choice of sentiment analysis tool (e.g., `TextBlob` vs. pre-trained models) impacts model performance.

- **Iterative Model Improvement**:
  - A baseline Random Forest model is used initially.
  - Techniques like backtesting and advanced algorithms (XGBoost) are employed to iteratively enhance performance.

- **Feature Engineering**:
  - Combining historical price data with sentiment analysis creates a robust feature set for training the model.

## Common Errors and Resolutions

1. **Missing Data**:
   - Ensure the historical price data and sentiment analysis CSV are correctly generated and accessible.
   - Handle missing values in the dataset before training the model.

2. **Library Installation Issues**:
   - Install all dependencies using the provided `requirements.txt` file to avoid compatibility problems.

3. **Model Overfitting**:
   - Use backtesting and cross-validation to identify and mitigate overfitting.

## Suggestions for Improvement

- Explore additional sentiment analysis tools or datasets to enrich the feature set.
- Experiment with other machine learning models like LSTM or Prophet for time series prediction.
- Develop a dashboard for real-time sentiment analysis and price prediction.

## Future Work

1. **Real-Time Integration**:
   - Automate the pipeline to fetch live sentiment data and historical prices.
   - Update the model dynamically with new data.

2. **Additional Features**:
   - Incorporate trading volume, market news, or social media sentiment as additional features.
   - Use external factors (e.g., Bitcoin price trends) for multi-variable predictions.

3. **Scalability**:
   - Extend the model to predict prices of other cryptocurrencies or financial assets.
   - Deploy the model as a web application or API for wider accessibility.

## Acknowledgements

This project is inspired by a tutorial provided by the [DataQuest YouTube channel](https://www.youtube.com/user/DataquestIO). The base code has been adapted with a slightly different approach, replacing the Hugging Face pre-trained sentiment analysis model with `TextBlob`, resulting in improved interpretability and better results. 

Special thanks to DataQuest for providing the foundational concepts and guidance that shaped this work.



  
