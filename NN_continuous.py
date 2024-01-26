import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import time
import os

class NN_linear:
    '''
TODO: TODO:
short the X variables (maybe I need another loop for different X)? plus check names on the table


    '''

    def __init__(self, *, file_paths, table_path, y_variables, selected_feature_columns, table_heads, first_heading, second_heading, levels = 3, neurons = 500, splits=10, early_stopping=False):
        self.file_paths = file_paths
        self.table_path = table_path
        self.y_variables = y_variables
        self.selected_feature_columns = selected_feature_columns
        self.first_heading = first_heading
        self.second_heading = second_heading
        self.table_heads = [first_heading, second_heading, "Clinical variable", "DMO", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R-squared (R^2)"]
        self.levels = levels
        self.neurons = neurons
        self.splits = splits
        self.early_stopping = early_stopping
    def train(self):

        table = pd.DataFrame(columns= self.table_heads)

        for file in self.file_paths:
            df = pd.read_csv(file)
            X = df[self.selected_feature_columns]

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            for y_variable in self.y_variables:
                y = df[y_variable]

                # Define the Multi-layer Perceptron (MLP) Regressor model
                model = MLPRegressor(
                    hidden_layer_sizes=tuple([self.neurons]*self.levels),
                    max_iter=1000,
                    random_state=0,
                    early_stopping=self.early_stopping,
                    validation_fraction=0.1,
                    n_iter_no_change=1000
                )
                start_time = time.time()

                # Create a KFold cross-validation object with the desired number of folds
                kf = KFold(n_splits=self.splits, random_state=0, shuffle=True)

                # Use cross_val_predict to get predictions for each fold
                y_pred_list = cross_val_predict(model, X, y, cv=kf)

                # Calculate and print regression metrics
                mse = mean_squared_error(y, y_pred_list)
                mse = round(mse, 3)
                r2 = r2_score(y, y_pred_list)
                r2 = round(r2, 3)
                rmse = np.sqrt(mse)
                rmse = round(rmse, 3)

                print("Evaluation results:")
                end_time = time.time()  # Record the end time
                training_time = end_time - start_time
                print("Training Time:", training_time)
                print("Mean Squared Error (MSE):", mse)
                print("Root Mean Squared Error (RMSE):", rmse)
                print("R-squared (R2):", r2)

                filename = os.path.splitext(os.path.basename(file))[0]

                results_df = pd.DataFrame({
                    self.first_heading: [y.name],
                    self.second_heading: [X.name],
                    "Mean Squared Error (MSE)": [mse],
                    "Root Mean Squared Error (RMSE)": [rmse],
                    "R-squared (R^2)": [r2],
                })

                table = pd.concat([table, results_df], ignore_index=True).reset_index(drop=True)

                # Save the results to a CSV file
                table_path = self.table_path
                table.to_csv(table_path, mode='w')`