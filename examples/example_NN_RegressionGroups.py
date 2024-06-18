from ML.NN_RegressionGroups import NN_RegressionGroups

'''
This is an example on how to use the NN_ClassifierGroups class to train a Multi-layer Perceptron (MLP) Regressor model using
example data. This version predicts class values (categorical variables).
'''

# Specifying parameters
file_paths = ['/Users/dimitrismegaritis/PycharmProjects/MLTable/examples/example_data_groups.csv']
table_path = '/Users/dimitrismegaritis/PycharmProjects/MLTable/examples/results_NN_RegressionGroups.csv'
target_variables = ['a']
input_feature_columns_continuous = ['b', 'c']
first_heading = 'clinical_variable'
second_heading = 'disease_status'
levels = 2
neurons = 50
splits = 5
scaler = 'yes'
early_stopping = False
group = ['ID']

# Creating an instance of NN_Classifier
nn_classifier = NN_RegressionGroups(
    file_paths=file_paths,
    table_path=table_path,
    target_variables=target_variables,
    input_feature_columns_continuous=input_feature_columns_continuous,
    first_heading=first_heading,
    second_heading=second_heading,
    levels=levels,
    neurons=neurons,
    splits=splits,
    scaler=scaler,
    early_stopping=early_stopping,
    group=group
)

nn_classifier.train()
