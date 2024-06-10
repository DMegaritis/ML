import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, accuracy_score
import time

'''
    This class trains a Logistic Regression model for classification tasks using k-fold
    cross-validation and saves the results to a table.

    Attributes
    ----------
    file_paths : list of str
        List of file paths for training data.
    table_path : str
        Path to save the results table.
    target_variables : list of str
        List of target variable names. 
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
    splits : int
        Number of folds for cross-validation.

    Methods
    -------
    train()
        Train the Logistic Regression model using k-fold cross-validation and save results to a CSV file.

    Returns
    -------
    self : Logistic_Regression
        Returns an instance of the Logistic_Regression class for use in a pipeline.
    '''


class Logistic_Regression:
    def __init__(self, *, file_paths, table_path, target_variables, input_feature_columns_continuous,
                 input_feature_columns_categorical=None, first_heading, second_heading, splits: int = 10):

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
        if splits is None:
            raise ValueError("Please input splits")

        self.file_paths = file_paths
        self.table_path = table_path
        self.target_variables = target_variables
        self.input_feature_columns_continuous = input_feature_columns_continuous
        self.input_feature_columns_categorical = input_feature_columns_categorical
        self.first_heading = first_heading
        self.second_heading = second_heading
        self.table_heads = [first_heading, second_heading,"Kappa", "F1", "Sensitivity", "Precision", "Accuracy"]
        self.splits = splits

    def train(self):
        table = pd.DataFrame(columns= self.table_heads)

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
                scaler = StandardScaler()
                X_continuous = scaler.fit_transform(X_continuous)

            else:
                df = df[self.input_feature_columns_continuous + self.target_variables]
                df = df.dropna()
                X_continuous = df[self.input_feature_columns_continuous]
                # Feature scaling on the continuous input data
                scaler = StandardScaler()
                X_continuous = scaler.fit_transform(X_continuous)

            # Combine the continuous and categorical features
            if self.input_feature_columns_categorical is not None:
                X = pd.concat([pd.DataFrame(X_continuous), pd.DataFrame(X_categorical)], axis=1)
            else:
                X = pd.DataFrame(X_continuous)

            for target_variable in self.target_variables:
                target = df[target_variable]

                # Label encoding on the target variable
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(target)

                model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)

                start_time = time.time()
                # Perform k-fold cross-validation
                y_pred_list = cross_val_predict(model, X, y_encoded, cv=self.splits)

                # Calculate and print various evaluation metrics
                kappa = cohen_kappa_score(y_encoded, y_pred_list)
                mean_f1 = f1_score(y_encoded, y_pred_list, average='weighted')
                mean_sensitivity = recall_score(y_encoded, y_pred_list, average='macro')
                mean_precision = precision_score(y_encoded, y_pred_list, average='macro')
                mean_accuracy = accuracy_score(y_encoded, y_pred_list)

                # Rounding
                kappa = round(kappa, 3)
                mean_f1 = round(mean_f1, 3)
                mean_sensitivity = round(mean_sensitivity, 3)
                mean_precision = round(mean_precision, 3)
                mean_accuracy = round(mean_accuracy, 3)

                print("Evaluation results:")
                end_time = time.time()  # Record the end time
                training_time = end_time - start_time
                print("Training Time:", training_time)

                print("Kappa:", kappa)
                print("F1:", mean_f1)
                print("Sensitivity:", mean_sensitivity)
                print("Precision:", mean_precision)
                print("Accuracy:", mean_accuracy)

                # Create a DataFrame for the results
                results_df = pd.DataFrame({
                    self.first_heading: [target_variable],
                    self.second_heading: [self.second_heading],
                    "Kappa": [kappa],
                    "F1": [mean_f1],
                    "Sensitivity": [mean_sensitivity],
                    "Precision": [mean_precision],
                    "Accuracy": [mean_accuracy]
                })

                table = pd.concat([table, results_df], ignore_index=True).reset_index(drop=True)

            # Save the results to a CSV file
            table_path = self.table_path
            table.to_csv(table_path, mode='w')

        return self
