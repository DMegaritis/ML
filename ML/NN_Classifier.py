import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import clone
import time
from matplotlib import pyplot as plt
import os


class NN_Classifier:
    """
    This class trains a Multi-layer Perceptron (MLP) Classifier model for classification tasks using k-fold
     cross-validation and saves the results to a table.

    Attributes
    ----------
    file_paths : list of str
        List of file paths for training data.
    table_path : str
        Path to save the results table.
    target_variables : list of str
        List of target variable names. Target variables are categorical.
    input_feature_columns_continuous : list of str
        List of selected continuous input feature column names.
    input_feature_columns_categorical : list of str
        List of selected continuous input feature column names.
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
    scaler : str
        Whether to scale the continuous input features.
    early_stopping : bool
        Whether to use early stopping to prevent overfitting.

    Methods
    -------
    train()
        Train the MLP model using cross-validation and save results to a CSV file.
    """

    def __init__(self, *, file_paths: List[str], table_path: str, target_variables: List[str] = None,
                 input_feature_columns_continuous: List[str], input_feature_columns_categorical: List[str] = None,
                 first_heading: str, second_heading: str, levels: int = 3, neurons: int = 100, splits: int = 10,
                 scaler="yes", early_stopping: bool = False):

        if file_paths is None:
            raise ValueError("Please input file_paths")
        if table_path is None:
            raise ValueError("Please input table_path")
        if target_variables is None:
            raise ValueError("Please input target_variables")
        if input_feature_columns_continuous is None:
            raise ValueError("Please input input_feature_columns_continuous")
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
        if scaler is None:
            raise ValueError("Please input scaler")
        if early_stopping is None:
            raise ValueError("Please input early_stopping")

        self.file_paths = file_paths
        self.table_path = table_path
        self.target_variables = target_variables
        self.input_feature_columns_continuous = input_feature_columns_continuous
        self.input_feature_columns_categorical = input_feature_columns_categorical
        self.first_heading = first_heading
        self.second_heading = second_heading
        self.table_heads = [first_heading, second_heading, "Kappa", "F1", "Sensitivity", "Precision", "Accuracy", "AUC"]
        self.levels = levels
        self.neurons = neurons
        self.splits = splits
        self.scaler = scaler
        self.early_stopping = early_stopping

    def train(self):
        '''
        Train the Multi-layer Perceptron (MLP) Classifier model using k-fold cross-validation.

        This method reads data from the specified file paths stored locally, performs feature scaling,
        and trains the MLP model using cross-validation. Results are saved to a CSV file. Input data must be in CSV.

        Returns
        -------
        self : NN_Classifier
            Returns an instance of the NN_Classifier class for use in a pipeline.
        '''

        table = pd.DataFrame(columns=self.table_heads)

        for file in self.file_paths:
            df = pd.read_csv(file)

            if self.input_feature_columns_categorical is not None:
                df = df[self.input_feature_columns_continuous + self.input_feature_columns_categorical + self.target_variables]
                df = df.dropna()
                # Categorical
                X_categorical = df[self.input_feature_columns_categorical]
                # One-hot encoding on the categorical input data
                encoder = OneHotEncoder()
                X_categorical = encoder.fit_transform(X_categorical).toarray()
                # Continuous
                X_continuous = df[self.input_feature_columns_continuous]
                # Feature scaling on the continuous input data
                if self.scaler == "yes":
                    scaler = StandardScaler()
                    X_continuous = scaler.fit_transform(X_continuous)
                elif self.scaler == "no":
                    X_continuous = X_continuous.values

            else:
                df = df[self.input_feature_columns_continuous + self.target_variables]
                df = df.dropna()
                X_continuous = df[self.input_feature_columns_continuous]
                # Feature scaling on the continuous input data (optional)
                if self.scaler == "yes":
                    scaler = StandardScaler()
                    X_continuous = scaler.fit_transform(X_continuous)
                elif self.scaler == "no":
                    X_continuous = X_continuous.values

            # Combine the continuous and categorical features
            if self.input_feature_columns_categorical is not None:
                X = pd.concat([pd.DataFrame(X_continuous), pd.DataFrame(X_categorical)], axis=1)
            else:
                X = pd.DataFrame(X_continuous)

            # Define the Multi-layer Perceptron (MLP) Classifier model
            model = MLPClassifier(
                hidden_layer_sizes=tuple([self.neurons] * self.levels),
                max_iter=1000,
                random_state=0,
                early_stopping=self.early_stopping,
                validation_fraction=0.1,
                n_iter_no_change=1000
            )

            for target_variable in self.target_variables:
                target = df[target_variable]
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(target)
                range = max(y_encoded) - min(y_encoded)

                start_time = time.time()

                model = clone(model)
                kf = KFold(n_splits=self.splits, random_state=0, shuffle=True)

                # Use cross_val_predict to get predictions for each fold
                y_pred_list = cross_val_predict(model, X, y_encoded, cv=kf)
                y_prob_list = cross_val_predict(model, X, y_encoded, cv=self.splits, method='predict_proba')[:, 1]

                # Calculate and print various evaluation metrics
                kappa = cohen_kappa_score(y_encoded, y_pred_list)
                mean_f1 = f1_score(y_encoded, y_pred_list, average='weighted')
                mean_sensitivity = recall_score(y_encoded, y_pred_list, average='macro')
                mean_precision = precision_score(y_encoded, y_pred_list, average='macro')
                mean_accuracy = accuracy_score(y_encoded, y_pred_list)
                if range == 1:
                    auc = roc_auc_score(y_encoded, y_prob_list)
                elif range > 1:
                    auc = "non-binary target"

                # rounding
                kappa = round(kappa, 3)
                mean_f1 = round(mean_f1, 3)
                mean_sensitivity = round(mean_sensitivity, 3)
                mean_precision = round(mean_precision, 3)
                mean_accuracy = round(mean_accuracy, 3)
                if range == 1:
                    auc = round(auc, 3)

                print("Evaluation results:")
                end_time = time.time()  # Record the end time
                training_time = end_time - start_time
                print("Training Time:", training_time)

                print("Kappa:", kappa)
                print("F1:", mean_f1)
                print("Sensitivity:", mean_sensitivity)
                print("Precision:", mean_precision)
                print("Accuracy:", mean_accuracy)
                print("AUC:", auc)

                # Create a DataFrame for the results
                results_df = pd.DataFrame({
                    self.first_heading: [target_variable],
                    self.second_heading: [self.second_heading],
                    "Kappa": [kappa],
                    "F1": [mean_f1],
                    "Sensitivity": [mean_sensitivity],
                    "Precision": [mean_precision],
                    "Accuracy": [mean_accuracy],
                    "AUC": [auc]
                })

                # Plot ROC curve
                if range == 1:
                    fpr, tpr, _ = roc_curve(y_encoded, y_prob_list)
                    plt.figure()
                    plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
                    plt.xlabel('Specificity')
                    plt.ylabel('Sensitivity')
                    plt.title(f'ROC Curve for {target_variable}')
                    plt.legend(loc='lower right')
                    plt.show()

                table = pd.concat([table, results_df], ignore_index=True).reset_index(drop=True)

            # Save the results to a CSV file
            table_path = self.table_path
            table.to_csv(table_path, mode='w')

        return self
