from ML.LogisticRegression import Logistic_Regression

'''
This is an example on how to use the Logist_Regression class to train a logistic regression machine learning model 
using example data. This version predicts class values (categorical variables).
'''

# Specifying parameters
file_paths = ['/Users/dimitrismegaritis/PycharmProjects/MLTable/examples/example_data_classification.csv']
table_path = '/Users/dimitrismegaritis/PycharmProjects/MLTable/examples/results_LogisticRegression.csv'
target_variables = ['g', 'h']
input_feature_columns_continuous = ['a', 'b', 'c']
input_feature_columns_categorical = ['i']
first_heading = 'clinical_variable'
second_heading = 'disease_status'
splits = 10

Log_Reg = Logistic_Regression(file_paths=file_paths,
    table_path=table_path,
    target_variables=target_variables,
    input_feature_columns_continuous=input_feature_columns_continuous,
    input_feature_columns_categorical=input_feature_columns_categorical,
    first_heading=first_heading,
    second_heading=second_heading,
    splits=splits)

Log_Reg.train()
