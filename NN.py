import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score, balanced_accuracy_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
import time
import os

file_paths = [
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/Cadence.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/Duration.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/StrideDuration.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/StrideLength.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/WalkingSpeed.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/SingleSupportDuration.csv'
]

y_variables = ['GOLD', 'CAT_high']

table = pd.DataFrame(columns=["Clinical variable", "DMO", "Kappa",
                              "F1", "Sensitivity", "Precision", "Accuracy"])

for filecsv_file_path in file_paths:

    df = pd.read_csv(filecsv_file_path)
    selected_feature_columns = ['max', 'min', 'mean', 'median', 'X25p', 'X75p', 'X90p', 'X95p', 'cv', 'Height', 'Age', 'nb']
    X = df[selected_feature_columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for y_variable in y_variables:
        y = df[y_variable]

        # Label encoding if your classes are not numeric
        #label_encoder = LabelEncoder()
        #y_encoded = label_encoder.fit_transform(y)

        # Define the Multi-layer Perceptron (MLP) model
        model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=0)
        model = clone(model)
        kf = KFold(n_splits=10, random_state=0, shuffle=True)

        start_time = time.time()

        # Use cross_val_predict to get predictions for each fold
        y_pred_list = cross_val_predict(model, X, y, cv=kf)

        # Calculate and print various evaluation metrics
        kappa = cohen_kappa_score(y, y_pred_list)
        mean_f1 = f1_score(y, y_pred_list, average='weighted')
        mean_sensitivity = recall_score(y, y_pred_list, average='macro')
        mean_precision = precision_score(y, y_pred_list, average='macro')
        mean_accuracy = accuracy_score(y, y_pred_list)

        # rounding
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


        filename = os.path.splitext(os.path.basename(filecsv_file_path))[0]

        # Create a DataFrame for the results
        results_df = pd.DataFrame({
            "Clinical variable": [y.name],
            "DMO": [filename],
            "Kappa": [kappa],
            "F1": [mean_f1],
            "Sensitivity": [mean_sensitivity],
            "Precision": [mean_precision],
            "Accuracy": [mean_accuracy]
        })

        table = pd.concat([table, results_df], ignore_index=True).reset_index(drop=True)

        # Save the results to a CSV file
        table_path = '/Users/dimitrismegaritis/Documents/TVS/machine learning/result_tables/NN.csv'
        table.to_csv(table_path, mode='w')



