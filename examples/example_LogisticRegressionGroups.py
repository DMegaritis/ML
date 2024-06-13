from ML.LogisticRegressionGroups import Logistic_Regression_Groups

'''
This is an example on how to use the Logist_Regression_Groups class to train a logistic regression machine learning model 
using example data. This version predicts class values (categorical variables) while grouping the data in prespecified groups.
'''

# Specifying parameters
file_paths = ['/Users/dimitrismegaritis/PycharmProjects/MLTable/examples/example_data_classification_groups.csv']
table_path = '/Users/dimitrismegaritis/PycharmProjects/MLTable/examples/results_LogisticRegressionGroups.csv'
target_variables = ['target']
input_feature_columns_continuous = ['a', 'b', 'c']
first_heading = 'clinical_variable'
second_heading = 'disease_status'
splits = 5
group = ['ID']

Log_Reg = Logistic_Regression_Groups(file_paths=file_paths,
    table_path=table_path,
    target_variables=target_variables,
    input_feature_columns_continuous=input_feature_columns_continuous,
    first_heading=first_heading,
    second_heading=second_heading,
    splits=splits,
    group=group)

Log_Reg.train()
