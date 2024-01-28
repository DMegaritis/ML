from ML.NN_Regression import NN_Regression

'''
This is an example on how to use the NN_Regressions class to train a Multi-layer Perceptron (MLP) Regressor model using
 example data. This version predicts continuous variables.
'''

# Specifying parameters
file_paths = ['/Users/dimitrismegaritis/PycharmProjects/ML/examples/example_data_regression.csv']
table_path = '/Users/dimitrismegaritis/PycharmProjects/ML/examples/results_NN_Regression.csv'
target_variables = ['g', 'h']
input_feature_columns = ['a', 'b', 'c']
first_heading = 'clinical_variable'
second_heading = 'diseasse_status'
levels = 3
neurons = 500
splits = 10
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
