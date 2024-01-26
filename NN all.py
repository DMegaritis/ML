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
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/masterfileall_wide.csv',
]

y_variables = ['GOLD', 'CAT_high']

table = pd.DataFrame(columns=["Clinical variable", "DMO", "Kappa",
                              "F1", "Sensitivity", "Precision", "Accuracy"])

for filecsv_file_path in file_paths:

    df = pd.read_csv(filecsv_file_path)
    selected_feature_columns = ['max_AverageStrideDuration', 'max_AverageStrideLength', 'max_Cadence', 'max_Duration', 'max_SingleSupportDuration', 'max_WalkingSpeed', 'max_AverageStrideDuration_10', 'max_AverageStrideLength_10', 'max_Cadence_10', 'max_Duration_10', 'max_SingleSupportDuration_10', 'max_WalkingSpeed_10', 'max_AverageStrideDuration_30', 'max_AverageStrideLength_30', 'max_Cadence_30', 'max_Duration_30', 'max_SingleSupportDuration_30', 'max_WalkingSpeed_30', 'min_AverageStrideDuration', 'min_AverageStrideLength', 'min_Cadence', 'min_Duration', 'min_SingleSupportDuration', 'min_WalkingSpeed', 'min_AverageStrideDuration_10', 'min_AverageStrideLength_10', 'min_Cadence_10', 'min_Duration_10', 'min_SingleSupportDuration_10', 'min_WalkingSpeed_10', 'min_AverageStrideDuration_30', 'min_AverageStrideLength_30', 'min_Cadence_30', 'min_Duration_30', 'min_SingleSupportDuration_30', 'min_WalkingSpeed_30', 'mean_AverageStrideDuration', 'mean_AverageStrideLength', 'mean_Cadence', 'mean_Duration', 'mean_SingleSupportDuration', 'mean_WalkingSpeed', 'mean_AverageStrideDuration_10', 'mean_AverageStrideLength_10', 'mean_Cadence_10', 'mean_Duration_10', 'mean_SingleSupportDuration_10', 'mean_WalkingSpeed_10', 'mean_AverageStrideDuration_30', 'mean_AverageStrideLength_30', 'mean_Cadence_30', 'mean_Duration_30', 'mean_SingleSupportDuration_30', 'mean_WalkingSpeed_30', 'median_AverageStrideDuration', 'median_AverageStrideLength', 'median_Cadence', 'median_Duration', 'median_SingleSupportDuration', 'median_WalkingSpeed', 'median_AverageStrideDuration_10', 'median_AverageStrideLength_10', 'median_Cadence_10', 'median_Duration_10', 'median_SingleSupportDuration_10', 'median_WalkingSpeed_10', 'median_AverageStrideDuration_30', 'median_AverageStrideLength_30', 'median_Cadence_30', 'median_Duration_30', 'median_SingleSupportDuration_30', 'median_WalkingSpeed_30', 'X25p_AverageStrideDuration', 'X25p_AverageStrideLength', 'X25p_Cadence', 'X25p_Duration', 'X25p_SingleSupportDuration', 'X25p_WalkingSpeed', 'X25p_AverageStrideDuration_10', 'X25p_AverageStrideLength_10', 'X25p_Cadence_10', 'X25p_Duration_10', 'X25p_SingleSupportDuration_10', 'X25p_WalkingSpeed_10', 'X25p_AverageStrideDuration_30', 'X25p_AverageStrideLength_30', 'X25p_Cadence_30', 'X25p_Duration_30', 'X25p_SingleSupportDuration_30', 'X25p_WalkingSpeed_30', 'X50p_AverageStrideDuration', 'X50p_AverageStrideLength', 'X50p_Cadence', 'X50p_Duration', 'X50p_SingleSupportDuration', 'X50p_WalkingSpeed', 'X50p_AverageStrideDuration_10', 'X50p_AverageStrideLength_10', 'X50p_Cadence_10', 'X50p_Duration_10', 'X50p_SingleSupportDuration_10', 'X50p_WalkingSpeed_10', 'X50p_AverageStrideDuration_30', 'X50p_AverageStrideLength_30', 'X50p_Cadence_30', 'X50p_Duration_30', 'X50p_SingleSupportDuration_30', 'X50p_WalkingSpeed_30', 'X75p_AverageStrideDuration', 'X75p_AverageStrideLength', 'X75p_Cadence', 'X75p_Duration', 'X75p_SingleSupportDuration', 'X75p_WalkingSpeed', 'X75p_AverageStrideDuration_10', 'X75p_AverageStrideLength_10', 'X75p_Cadence_10', 'X75p_Duration_10', 'X75p_SingleSupportDuration_10', 'X75p_WalkingSpeed_10', 'X75p_AverageStrideDuration_30', 'X75p_AverageStrideLength_30', 'X75p_Cadence_30', 'X75p_Duration_30', 'X75p_SingleSupportDuration_30', 'X75p_WalkingSpeed_30', 'X90p_AverageStrideDuration', 'X90p_AverageStrideLength', 'X90p_Cadence', 'X90p_Duration', 'X90p_SingleSupportDuration', 'X90p_WalkingSpeed', 'X90p_AverageStrideDuration_10', 'X90p_AverageStrideLength_10', 'X90p_Cadence_10', 'X90p_Duration_10', 'X90p_SingleSupportDuration_10', 'X90p_WalkingSpeed_10', 'X90p_AverageStrideDuration_30', 'X90p_AverageStrideLength_30', 'X90p_Cadence_30', 'X90p_Duration_30', 'X90p_SingleSupportDuration_30', 'X90p_WalkingSpeed_30', 'X95p_AverageStrideDuration', 'X95p_AverageStrideLength', 'X95p_Cadence', 'X95p_Duration', 'X95p_SingleSupportDuration', 'X95p_WalkingSpeed', 'X95p_AverageStrideDuration_10', 'X95p_AverageStrideLength_10', 'X95p_Cadence_10', 'X95p_Duration_10', 'X95p_SingleSupportDuration_10', 'X95p_WalkingSpeed_10', 'X95p_AverageStrideDuration_30', 'X95p_AverageStrideLength_30', 'X95p_Cadence_30', 'X95p_Duration_30', 'X95p_SingleSupportDuration_30', 'X95p_WalkingSpeed_30', 'StdDev_AverageStrideDuration', 'StdDev_AverageStrideLength', 'StdDev_Cadence', 'StdDev_Duration', 'StdDev_SingleSupportDuration', 'StdDev_WalkingSpeed', 'StdDev_AverageStrideDuration_10', 'StdDev_AverageStrideLength_10', 'StdDev_Cadence_10', 'StdDev_Duration_10', 'StdDev_SingleSupportDuration_10', 'StdDev_WalkingSpeed_10']

    X = df[selected_feature_columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for y_variable in y_variables:
        y = df[y_variable]

        # Label encoding if your classes are not numeric
        #label_encoder = LabelEncoder()
        #y_encoded = label_encoder.fit_transform(y)

        # Define the Multi-layer Perceptron (MLP) model
        model = MLPClassifier(hidden_layer_sizes=(500, 500, 500), max_iter=10000, random_state=0)
        model = clone(model)
        kf = KFold(n_splits=5, random_state=0, shuffle=True)

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
        table_path = '/Users/dimitrismegaritis/Documents/TVS/machine learning/result_tables/NNall.csv'
        table.to_csv(table_path, mode='w')





