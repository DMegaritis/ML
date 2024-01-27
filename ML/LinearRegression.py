import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import time
import os


class Linear_Regression:
    def __init__(self, *, file_paths, table_path, target_variables, input_feature_columns,
                 first_heading, second_heading, splits: int = 10):
        self.file_paths = file_paths
        self.table_path = table_path
        self.target_variables = target_variables
        self.input_feature_columns = input_feature_columns
        self.first_heading = first_heading
        self.second_heading = second_heading
        self.table_heads = [first_heading, second_heading, "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)",
                            "R-squared (R^2)"]
        self.splits = splits

    def train(self):
        table = pd.DataFrame(columns=self.table_heads)

        for file in self.file_paths:
            df = pd.read_csv(file)

            # Extract features and labels from your DataFrame
            X = df[self.input_feature_columns]

            # Feature scaling on the input data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            for target_variable in self.target_variables:
                target = df[target_variable]

                model = LinearRegression()

                start_time = time.time()

                # Perform k-fold cross-validation
                y_pred_list = cross_val_predict(model, X, target, cv=self.splits)

                # Calculate and print regression-specific evaluation metrics
                mse = mean_squared_error(target, y_pred_list)
                mse = round(mse, 3)
                rmse = np.sqrt(mse)
                rmse = round(rmse, 3)
                r2 = r2_score(target, y_pred_list)
                r2 = round(r2, 3)

                # Printing the results
                end_time = time.time()  # Record the end time
                training_time = end_time - start_time
                print("Training Time for", file, ":", training_time)
                print("Evaluation results for", file, ":")
                print("Mean Squared Error (MSE):", mse)
                print("R-squared (R^2):", r2)
                print("Root Mean Squared Error (RMSE):", rmse)

                # Create a DataFrame for the results
                results_df = pd.DataFrame({
                    self.first_heading: [target_variable],
                    self.second_heading: [self.second_heading],
                    "Mean Squared Error (MSE)": [mse],
                    "Root Mean Squared Error (RMSE)": [rmse],
                    "R-squared (R^2)": [r2],
                })

                table = pd.concat([table, results_df], ignore_index=True).reset_index(drop=True)

            # Save the results to a CSV file
            table_path = self.table_path
            table.to_csv(table_path, mode='w')

        return self
