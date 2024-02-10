import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import time


class NN_Regression:
    '''
    This class train a Multi-layer Perceptron (MLP) Regressor model for regression tasks using k-fold cross-validation
     and saves the results to a table.

    Attributes
    ----------
    file_paths : list of str
        List of file paths for training data.
    table_path : str
        Path to save the results table.
    target_variables : list of str
        List of target variable names. Target variables are continuous.
    input_feature_columns : list of str
        List of selected input feature column names.
    first_heading : str
        Heading for the first column in the results table.
    second_heading : str
        Heading for the second column in the results table.
    table_heads : list of str
        List of column headings for the results table.
    levels : int
        Number of hidden layers in the MLP model.
    neurons : int
        Number of neurons in each hidden layer.
    splits : int
        Number of folds for cross-validation.
    early_stopping : bool
        Whether to use early stopping during training.

    Methods
    -------
    train()
        Train the MLP model using cross-validation and save results to a CSV file.
    '''

    def __init__(self, *, file_paths: List[str], table_path: str, target_variables: [List[str]] = None,
                 input_feature_columns: List[str], first_heading: str, second_heading: str,
                 levels: int = 3, neurons: int = 500, splits: int = 10, early_stopping: bool = False):

        if file_paths is None:
            raise ValueError("Please input file_paths")
        if table_path is None:
            raise ValueError("Please input table_path")
        if target_variables is None:
            raise ValueError("Please input target_variables")
        if input_feature_columns is None:
            raise ValueError("Please input input_feature_columns")
        if first_heading is None:
            raise ValueError("Please input first_heading")
        if second_heading is None:
            raise ValueError("Please input second_heading")
        if levels is None:
            raise ValueError("Please input levels")
        if neurons is None:
            raise ValueError("Please input neurons")
        if splits is None:
            raise ValueError("Please input splits")
        if early_stopping is None:
            raise ValueError("Please input early_stopping")

        self.file_paths = file_paths
        self.table_path = table_path
        self.target_variables = target_variables
        self.input_feature_columns = input_feature_columns
        self.first_heading = first_heading
        self.second_heading = second_heading
        self.table_heads = [first_heading, second_heading, "Mean Squared Error (MSE)",
                            "Root Mean Squared Error (RMSE)", "R-squared (R^2)"]
        self.levels = levels
        self.neurons = neurons
        self.splits = splits
        self.early_stopping = early_stopping

    def train(self):
        '''
        Train the Multi-layer Perceptron (MLP) Regressor model using k-fold cross-validation.

        This method reads data from the specified file paths stored locally, performs feature scaling,
        and trains the MLP model using cross-validation. Results are saved to a CSV file. Input data must be in CSV

        Returns
        -------
        self : NN_Regression
            Returns an instance of the NN_Regression class for use in a pipeline.
        '''

        table = pd.DataFrame(columns= self.table_heads)

        for file in self.file_paths:
            df = pd.read_csv(file)
            df = df[self.input_feature_columns + self.target_variables]
            df = df.dropna()
            X = df[self.input_feature_columns]

            # Feature scaling on the input data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Define the Multi-layer Perceptron (MLP) Regressor model
            model = MLPRegressor(
                hidden_layer_sizes=tuple([self.neurons] * self.levels),
                max_iter=1000,
                random_state=0,
                early_stopping=self.early_stopping,
                validation_fraction=0.1,
                n_iter_no_change=1000
            )

            for target_variable in self.target_variables:
                target = df[target_variable]

                start_time = time.time()

                # Create a KFold cross-validation object with the desired number of folds
                kf = KFold(n_splits=self.splits, random_state=0, shuffle=True)

                # Use cross_val_predict to get predictions for each fold
                y_pred_list = cross_val_predict(model, X, target, cv=kf)

                # Calculate and print regression metrics
                mse = mean_squared_error(target, y_pred_list)
                mse = round(mse, 3)
                r2 = r2_score(target, y_pred_list)
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

                results_df = pd.DataFrame({
                    self.first_heading: [target.name],
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
