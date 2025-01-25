# PortfolioWise: Data-Driven Investment Strategies

**PortfolioWise** is a machine learning-based project designed to develop and implement a portfolio selection strategy. Using historical stock price data, the project aims to optimize asset allocation within a portfolio, balancing risk and return through predictive models and financial metrics like the Sharpe Ratio.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)
- [Files in the Repository](#files-in-the-repository)
- [Authors](#authors)

## Introduction
This project focuses on creating a machine learning-based portfolio selection strategy using historical stock price data from the S&P 500 index. By employing predictive models, including decision tree regression and linear regression, the project allocates portfolio weights to optimize risk-adjusted returns, measured through the Sharpe Ratio.

## Features
- Historical stock data preprocessing.
- Predictive modeling using:
  - Decision tree regression.
  - Linear regression with normalization and standardization.
- Portfolio allocation based on expected returns and risk.
- Sharpe Ratio and portfolio variance calculation for performance evaluation.

## Data
The dataset comprises historical stock price data of S&P 500 companies over five years, with the following attributes:
- Daily stock prices.
- Trading volumes.
- Open, close, high, and low prices.
- Additional features like daily price changes and percentage changes.

The test dataset includes stock price data for 22 days following the training period to evaluate the strategy in unseen conditions.

## Methodology
1. **Data Preprocessing**:
   - Downloading and cleaning stock price data from Yahoo Finance.
   - Generating features like daily price changes and percentage changes.

2. **Model Training**:
   - Building individual models (decision tree regression and linear regression) for each stock.
   - Evaluating models using RMSE and MAE metrics.

3. **Portfolio Allocation**:
   - Assigning weights to stocks based on predicted returns and historical risk.
   - Normalizing weights to ensure the portfolio sum equals 1.

4. **Performance Metrics**:
   - Sharpe Ratio: Evaluating risk-adjusted returns.
   - Portfolio Variance: Measuring volatility.

## Results
Key performance metrics:
- **Sharpe Ratio**: 0.3773, indicating positive risk-adjusted returns.
- **Portfolio Variance**: 0.0002136, reflecting low portfolio volatility.

These results demonstrate the strategy's ability to balance returns and risk effectively.

## Setup and Usage
### Requirements
Install the necessary libraries using:
```bash
pip install numpy pandas scikit-learn joblib yfinance
Running the Code
Train and save models for portfolio selection:
bash
Copier
Modifier
python portfolio.py
Evaluate portfolio performance with the provided test dataset.

## Files in the Repository
Portfolio Selection Project Report.pdf: Comprehensive report summarizing methods and results.
portfolio.py: Python script for training models and allocating portfolio weights.
all_models.joblib: Serialized models for portfolio predictions.
## Authors
Daniel Dahan (ID: 345123624)
Simon Bellilty (ID: 345233563)
Acknowledgments
Yahoo Finance(https://finance.yahoo.com/) for stock data.
Scikit-learn Documentation(https://scikit-learn.org/) for regression models.
