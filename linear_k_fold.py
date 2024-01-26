import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

# Define a list of file paths
file_paths = [
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/Cadence.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/Duration.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/StrideDuration.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/StrideLength.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/WalkingSpeed.csv',
    '/Users/dimitrismegaritis/Documents/TVS/machine learning/masterfiles/SingleSupportDuration.csv'
]

y_variables = ['FEV1.L', 'FVC.L', 'FEV1.FVC', 'FEV1.pred', 'FVC.pred', 'CAT']

table = pd.DataFrame(columns=["Clinical variable", "DMO", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R-squared (R^2)"])

for filecsv_file_path in file_paths:
    df = pd.read_csv(filecsv_file_path)

    # Extract features and labels from your DataFrame
    selected_feature_columns = ['max', 'min', 'mean', 'median', 'X25p', 'X75p', 'X90p', 'X95p', 'cv', 'Height', 'Age']
    X = df[selected_feature_columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for y_variable in y_variables:
        y = df[y_variable]

        model = LinearRegression()

        start_time = time.time()

        num_folds = 10
        y_pred_list = cross_val_predict(model, X, y, cv=num_folds)

        # Calculate and print regression-specific evaluation metrics
        mse = mean_squared_error(y, y_pred_list)
        mse = round(mse, 3)
        rmse = np.sqrt(mse)
        rmse = round(rmse, 3)
        r2 = r2_score(y, y_pred_list)
        r2 = round(r2, 3)

        # Printing the results
        end_time = time.time()  # Record the end time
        training_time = end_time - start_time
        print("Training Time for", filecsv_file_path, ":", training_time)
        print("Evaluation results for", filecsv_file_path, ":")
        print("Mean Squared Error (MSE):", mse)
        print("R-squared (R^2):", r2)
        print("Root Mean Squared Error (RMSE):", rmse)

        filename = os.path.splitext(os.path.basename(filecsv_file_path))[0]

        # Create a DataFrame for the results
        results_df = pd.DataFrame({
            "Clinical variable": [y.name],
            "DMO": [filename],
            "Mean Squared Error (MSE)": [mse],
            "Root Mean Squared Error (RMSE)": [rmse],
            "R-squared (R^2)": [r2],
        })

        table = pd.concat([table, results_df], ignore_index=True).reset_index(drop=True)


        # Save the results to a CSV file
        table_path = '/Users/dimitrismegaritis/Documents/TVS/machine learning/result_tables/linear_reg.csv'
        table.to_csv(table_path, mode='w')
