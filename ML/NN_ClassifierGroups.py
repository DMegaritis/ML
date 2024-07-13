import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import clone
import time
from matplotlib import pyplot as plt
import os


class NN_ClassifierGroups:
    """
    This class trains a Multi-layer Perceptron (MLP) Classifier model for classification tasks using k-fold
     cross-validation and saves the results to a table. This class is an extension of the NN_Classifier class
     working with grouped data (i.e., high frequency data).

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
        Whether to use early stopping.
    group : str
        Group column name for GroupKFold.

    Methods
    -------
    train()
        Train the MLP model using cross-validation and save results to a CSV file.
    """

    def __init__(self, *, file_paths: List[str], table_path: str, target_variables: List[str] = None,
                 input_feature_columns_continuous: List[str], input_feature_columns_categorical: List[str] = None,
                 first_heading: str, second_heading: str, levels: int = 3, neurons: int = 100, splits: int = 10,
                 scaler="yes", early_stopping: bool = False, group):

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
        if group is None:
            raise ValueError("Please input group")

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
        self.group = group

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
                df = df[self.input_feature_columns_continuous + self.input_feature_columns_categorical + self.group + self.target_variables]
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
                df = df[self.input_feature_columns_continuous + self.group +self.target_variables]
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

                model = clone(model)

                start_time = time.time()

                results_list = []
                group_kfold = GroupKFold(n_splits=self.splits)
                for train_index, test_index in group_kfold.split(X, y_encoded, groups=df[self.group]):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    results_list.append({
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "y_prob": y_prob
                    })

                # Aggregate evaluation metrics across all folds
                aggregated_y_test = np.concatenate([result["y_test"] for result in results_list])
                aggregated_y_pred = np.concatenate([result["y_pred"] for result in results_list])
                aggregated_y_prob = np.concatenate([result["y_prob"] for result in results_list])

                kappa = cohen_kappa_score(aggregated_y_test, aggregated_y_pred)
                mean_f1 = f1_score(aggregated_y_test, aggregated_y_pred, average='weighted')
                mean_sensitivity = recall_score(aggregated_y_test, aggregated_y_pred, average='macro')
                mean_precision = precision_score(aggregated_y_test, aggregated_y_pred, average='macro')
                mean_accuracy = accuracy_score(aggregated_y_test, aggregated_y_pred)
                auc = roc_auc_score(aggregated_y_test, aggregated_y_prob)

                # Rounding
                kappa = round(kappa, 3)
                mean_f1 = round(mean_f1, 3)
                mean_sensitivity = round(mean_sensitivity, 3)
                mean_precision = round(mean_precision, 3)
                mean_accuracy = round(mean_accuracy, 3)
                if range == 1:
                    auc = round(auc, 3)

                print("Evaluation results:")
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
                    fpr, tpr, _ = roc_curve(aggregated_y_test, aggregated_y_prob)
                    plt.figure(figsize=(10, 15))
                    plt.plot(fpr, tpr, color='orange', label=f'(Kappa = {kappa:.2f}, F1 = {mean_f1:.2f}, Sensitivity = {mean_sensitivity:.2f}, Precision = {mean_precision:.2f}, Accuracy = {mean_accuracy:.2f}, AUC = {auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
                    plt.xlabel('Specificity', fontsize=8)
                    plt.ylabel('Sensitivity', fontsize=8)
                    plt.title(f'ROC Curve for {target_variable}')
                    plt.legend(loc='lower left', fontsize=10, bbox_to_anchor=(0, -0.2))
                    plt.tight_layout(pad=3)
                    plt.show()

                table = pd.concat([table, results_df], ignore_index=True).reset_index(drop=True)

                end_time = time.time()  # Record the end time
                training_time = end_time - start_time
                print("Training Time:", training_time)

        # Save the results to a CSV file
        table_path = self.table_path
        table.to_csv(table_path, mode='w')

        return self
