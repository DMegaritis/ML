from ML.NN_Regression import NN_Regression
from ML.NN_Classifier import NN_Classifier
from ML.LogisticRegression import Logistic_Regression
from ML.LinearRegression import Linear_Regression

'''
Analysis for the ERS24 abstract.
'''

# Specifying parameters
file_paths = [
              '/Users/dimitrismegaritis/Documents/ERS24/files/Cadence.csv',
              '/Users/dimitrismegaritis/Documents/ERS24/files/StrideDuration.csv',
              '/Users/dimitrismegaritis/Documents/ERS24/files/StrideLength.csv',
              '/Users/dimitrismegaritis/Documents/ERS24/files/WalkingSpeed.csv',
              '/Users/dimitrismegaritis/Documents/ERS24/files/SingleSupportDuration.csv'
              ]

# NN regression
table_path = '/Users/dimitrismegaritis/Documents/ERS24/results/results_NN_Regression.csv'
target_variables = ['FEV1.pred']
input_feature_columns = ["max", "min", "mean", "median", "X25p", "X50p", "X75p", "X90p", "X95p", "StdDev", "cv"]
first_heading = 'clinical_variable'
second_heading = 'disease_status'
levels = 3
neurons = 100
splits = 5
early_stopping = False


# Creating an instance of NN_Linear
nn_linear_instance = NN_Regression(
    file_paths=file_paths,
    table_path=table_path,
    target_variables=target_variables,
    input_feature_columns=input_feature_columns,
    first_heading=first_heading,
    second_heading=second_heading,
    levels=levels,
    neurons=neurons,
    splits=splits,
    early_stopping=early_stopping
)

# Calling the train method to train the model and save results to a CSV file
nn_linear_instance.train()

# NN classigication
table_path = '/Users/dimitrismegaritis/Documents/ERS24/results/results_NN_Classifier.csv'
target_variables = ['FEV1.pred']
input_feature_columns_continuous = ["max", "min", "mean", "median", "X25p", "X50p", "X75p", "X90p", "X95p", "StdDev", "cv"]
first_heading = 'clinical_variable'
second_heading = 'disease_status'
levels = 3
neurons = 100
splits = 5
early_stopping = False

# Creating an instance of NN_Classifier
nn_classifier = NN_Classifier(
    file_paths=file_paths,
    table_path=table_path,
    target_variables=target_variables,
    input_feature_columns_continuous=input_feature_columns_continuous,
    first_heading=first_heading,
    second_heading=second_heading,
    levels=levels,
    neurons=neurons,
    splits=splits,
    early_stopping=early_stopping
)

nn_classifier.train()



# Linear regression
table_path = '/Users/dimitrismegaritis/Documents/ERS24/results/Linear_Regression.csv'
target_variables = ['FEV1.pred']
input_feature_columns = ["max", "min", "mean", "median", "X25p", "X50p", "X75p", "X90p", "X95p", "StdDev", "cv"]
first_heading = 'clinical_variable'
second_heading = 'disease_status'
splits = 5

Linear_Reg = Linear_Regression(file_paths=file_paths,
                               table_path=table_path,
                               target_variables=target_variables,
                               input_feature_columns=input_feature_columns,
                               first_heading=first_heading,
                               second_heading=second_heading,
                               splits=splits)

Linear_Reg.train()




# Logistic regression
table_path = '/Users/dimitrismegaritis/Documents/ERS24/results/results_Logistic.csv'
target_variables = ['FEV1.pred']
input_feature_columns_continuous = ["max", "min", "mean", "median", "X25p", "X50p", "X75p", "X90p", "X95p", "StdDev", "cv"]
first_heading = 'clinical_variable'
second_heading = 'disease_status'
splits = 5


Log_Reg = Logistic_Regression(file_paths=file_paths,
    table_path=table_path,
    target_variables=target_variables,
    input_feature_columns_continuous=input_feature_columns_continuous,
    first_heading=first_heading,
    second_heading=second_heading,
    splits=splits)

Log_Reg.train()
