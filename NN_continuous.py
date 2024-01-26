import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import time
import os

# Load your data as you did before
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
    selected_feature_columns = ['max', 'min', 'mean', 'median', 'X25p', 'X75p', 'X90p', 'X95p', 'cv', 'Height', 'Age',
                                'nb']
    X = df[selected_feature_columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for y_variable in y_variables:
        y = df[y_variable]

        # Define the Multi-layer Perceptron (MLP) Regressor model
        model = MLPRegressor(
            hidden_layer_sizes=(100, 100),
            max_iter=1000,
            random_state=0,
            #early_stopping=True,  # Enable early stopping
            #validation_fraction=0.1,  # Fraction of training data used for validation
            #n_iter_no_change=2000  # Number of iterations with no improvement to wait for
        )
        start_time = time.time()

        # Create a KFold cross-validation object with the desired number of folds
        kf = KFold(n_splits=10, random_state=0, shuffle=True)

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

        filename = os.path.splitext(os.path.basename(filecsv_file_path))[0]

        results_df = pd.DataFrame({
            "Clinical variable": [y.name],
            "DMO": [filename],
            "Mean Squared Error (MSE)": [mse],
            "Root Mean Squared Error (RMSE)": [rmse],
            "R-squared (R^2)": [r2],
        })

        table = pd.concat([table, results_df], ignore_index=True).reset_index(drop=True)

        # Save the results to a CSV file
        table_path = '/Users/dimitrismegaritis/Documents/TVS/machine learning/result_tables/NN_continuous.csv'
        table.to_csv(table_path, mode='w')




