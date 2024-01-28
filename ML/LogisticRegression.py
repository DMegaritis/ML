import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, accuracy_score
import time


class Logistic_Regression:
    def __init__(self, *, file_paths, table_path, target_variables, input_feature_columns,
                 first_heading, second_heading, splits: int = 10):

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
        if splits is None:
            raise ValueError("Please input splits")

        self.file_paths = file_paths
        self.table_path = table_path
        self.target_variables = target_variables
        self.input_feature_columns = input_feature_columns
        self.first_heading = first_heading
        self.second_heading = second_heading
        self.table_heads = [first_heading, second_heading,"Kappa", "F1", "Sensitivity", "Precision", "Accuracy"]
        self.splits = splits

    def train(self):
        table = pd.DataFrame(columns= self.table_heads)

        for file in self.file_paths:
            df = pd.read_csv(file)

            # Extract features and labels from your DataFrame
            X = df[self.input_feature_columns]

            # Feature scaling on the input data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            for target_variable in self.target_variables:
                target = df[target_variable]
                # no need for label encoding if classes are numeric
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
