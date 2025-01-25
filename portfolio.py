
import pandas as pd

import joblib


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, Normalizer


class Portfolio:
    def __init__(self, weights=None):
        self.weights = weights

    def train(self, train_data: pd.DataFrame):
        data = train_data
        dict_models = {}
        column_names_1 = data['Adj Close']
        # print(column_names_1.columns)
        best_result_per_ticker = pd.DataFrame(columns=['Ticker', 'Model', 'RMSE'])
        best_result_per_ticker = list()
        i = 1
        for ticker_name in column_names_1.columns:
            # print(i)
            i += 1
            result = pd.DataFrame(columns=['Ticker', 'Model', 'RMSE'])
            # print(ticker_name)
            stock_a = pd.DataFrame()
            stock_a['Adj Close'] = data['Adj Close'][ticker_name]
            stock_a['Volume'] = data['Volume'][ticker_name]
            stock_a['Open'] = data['Open'][ticker_name]
            stock_a['Low'] = data['Low'][ticker_name]
            stock_a['High'] = data['High'][ticker_name]
            # stock_a['Close'] = data['Close'][ticker_name]
            stock_a['changeduringday'] = ((data['High'][ticker_name] - data['Low'][ticker_name]) / data['Low'][
                ticker_name]) * 100
            stock_a['changefrompreviousday'] = (abs(stock_a['Adj Close'].shift() - stock_a['Adj Close']) /
                                                data['Adj Close'][ticker_name]) * 100
            # print(stock_a)

            X_stock_a = stock_a.drop(['Adj Close'], axis=1)
            y_stock_a = stock_a['Adj Close']

            X_stock_train = X_stock_a[0:-22]
            X_stock_test = X_stock_a[-22:]
            y_stock_train = y_stock_a[0:-22]
            y_stock_test = y_stock_a[-22:]


            Lr_pipeline_nor = Pipeline([
                ('imputer', SimpleImputer(missing_values=np.nan, strategy="mean")),
                # Use the "median" to impute missing vlaues
                ('normalizer', Normalizer()),
                ('lr', LinearRegression())

            ])

            clean_indices = y_stock_train.dropna().index
            X_stock_train = X_stock_train.loc[clean_indices]
            y_stock_train = y_stock_train.loc[clean_indices]

            clean_indices = y_stock_test.dropna().index
            X_stock_test = X_stock_test.loc[clean_indices]
            y_stock_test = y_stock_test.loc[clean_indices]

            Lr_pipeline_nor.fit(X_stock_train, y_stock_train)
            # Data prep pipeline

            Lr_pipeline_std = Pipeline([
                ('imputer', SimpleImputer(missing_values=np.nan, strategy="mean")),
                # Use the "median" to impute missing vlaues
                ('scaler', StandardScaler()),
                ('lr', LinearRegression())

            ])

            Lr_pipeline_std.fit(X_stock_train, y_stock_train)

            # y_pred = Lr_pipeline_nor.predict(X_stock_test)
            # petit test perso
            """mse = mean_squared_error(y_stock_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_stock_test, y_pred)
            r2 = r2_score(y_stock_test, y_pred)

            print(f"Mean Squared Error: {mse}")
            print(f"Root Mean Squared Error: {rmse}")
            print(f"Mean Absolute Error: {mae}")
            print(f"R-squared: {r2}")"""
            # We tried Mean Absolute Error to understand how the metrics look but we use
            # RMSE for the actual accuracy measurement of our mode
            # Linear Regression with normalisation and standardisation
            lr_stock_predictions_nor = Lr_pipeline_nor.predict(X_stock_test)
            lr_mae_nor = mean_absolute_error(y_stock_test, lr_stock_predictions_nor)
            # print('Lr MAE with Normalization', lr_mae_nor)
            lr_mse_nor = mean_squared_error(y_stock_test, lr_stock_predictions_nor)
            lr_rmse_nor = np.sqrt(lr_mse_nor)
            rmse_row = [ticker_name, Lr_pipeline_nor, lr_rmse_nor]

            result.loc[-1] = rmse_row  # adding a row
            result.index = result.index + 1  # shifting index

            lr_stock_predictions_std = Lr_pipeline_std.predict(X_stock_test)
            lr_mse_std = mean_squared_error(y_stock_test, lr_stock_predictions_std)
            lr_rmse_std = np.sqrt(lr_mse_std)
            rmse_row = [ticker_name, Lr_pipeline_std, lr_rmse_std]

            result.loc[-1] = rmse_row  # adding a row
            result.index = result.index + 1  # shifting index

            best_result_per_ticker.append(np.array(result.iloc[0, :]))
            dict_models.update({ticker_name: np.array(result.iloc[0, :])[1]})

        best_result_per_ticker_df = pd.DataFrame(data=best_result_per_ticker, columns=['Ticker', 'Model', 'RMSE'])
        print(dict_models)
        joblib.dump(dict_models, 'all_models.joblib')

    def get_portfolio(self, train_data: pd.DataFrame):
        data = train_data
        column_names_1 = data['Adj Close']
        # Load the dictionary of models
        loaded_dict_models = joblib.load('all_models.joblib')
        # preds=[]
        asset_risks = {}
        predicted_values = {}
        for ticker_name in column_names_1.columns:
            ticker_to_predict = ticker_name

            if ticker_name in loaded_dict_models.keys():
                loaded_model = loaded_dict_models[ticker_to_predict]

                # print(ticker_name)
                stock_a = pd.DataFrame()
                stock_a['Adj Close'] = data['Adj Close'][ticker_name]
                stock_a['Volume'] = data['Volume'][ticker_name]
                stock_a['Open'] = data['Open'][ticker_name]
                stock_a['Low'] = data['Low'][ticker_name]
                stock_a['High'] = data['High'][ticker_name]
                # stock_a['Close'] = data['Close'][ticker_name]
                stock_a['changeduringday'] = ((data['High'][ticker_name] - data['Low'][ticker_name]) / data['Low'][
                    ticker_name]) * 100
                stock_a['changefrompreviousday'] = (abs(stock_a['Adj Close'].shift() - stock_a['Adj Close']) /
                                                    data['Adj Close'][ticker_name]) * 100
                # print(stock_a)

                X_stock_a = stock_a.drop(['Adj Close'], axis=1)

                y_pred = loaded_model.predict(X_stock_a)
                y_pred = y_pred[-1]
                y_pred_minus_1 = stock_a['Adj Close'][-1]
                # print("y_pred",y_pred)

                # print("y_pred_minus_1",y_pred_minus_1)
                value = ((y_pred - y_pred_minus_1) / y_pred_minus_1) * 100
                # preds.append(value)
                asset_risks.update({ticker_name: np.std(stock_a['Adj Close'])})

                predicted_values.update({ticker_name: value})

                # print(value)
                # break
            else:
                asset_risks.update({ticker_name: -1})
                predicted_values.update({ticker_name: 0})
        # print("asset_risks",asset_risks)
        # print("predicted_values",predicted_values)
        # Calculate a weighted combination of predicted values and risk for each asset
        allocation_weights = {}
        for asset in predicted_values:
            if predicted_values[asset] >= 0:
                risk_weight = 1 / asset_risks[asset]  # Inverse of risk as weight
                weight_sum = risk_weight + predicted_values[asset]
                allocation_weights[asset] = float(risk_weight) / float(weight_sum)
            else:
                allocation_weights[asset] = 0.0

        # Normalize allocation weights to sum up to 1 (percentage-based allocation)
        total_weight = sum(allocation_weights.values())
        # print("total_weight",total_weight)
        allocation_percentages = {asset: weight / total_weight for asset, weight in allocation_weights.items()}
        print("allocation_percentages", allocation_percentages)
        print(sum(allocation_percentages.values()))
        # Adjust allocation percentages to ensure the sum is closer to 1

        adjustment = 1 - sum(allocation_percentages.values())
        allocation_percentages['AAPL'] += adjustment
        print(allocation_percentages)
        print(sum(allocation_percentages.values()))
        return np.array(list(allocation_percentages.values()), dtype=float)

    def save_weights(self, filename):
        joblib.dump(self.weights, filename)

    def load_weights(self, filename):
        self.weights = joblib.load(filename)

    def sharpe_ratio(self, weights, expected_returns, cov_matrix):
        # Calculate the Sharpe ratio of the portfolio
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = -portfolio_return / portfolio_risk
        return sharpe_ratio