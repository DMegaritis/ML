from ML.LinearRegression import Linear_Regression

'''
This is an example on how to use the Linear_Regression class to train a linear regression machine learning model 
using example data. This version predicts continuous variables.
'''

# Specifying parameters
file_paths = ['/Users/dimitrismegaritis/PycharmProjects/ML/examples/example_data_regression.csv']
table_path = '/Users/dimitrismegaritis/PycharmProjects/ML/examples/results_LinearRegression.csv'
target_variables = ['g', 'h']
input_feature_columns = ['a', 'b', 'c']
first_heading = 'clinical_variable'
second_heading = 'performance'
splits = 10

Linear_Reg = Linear_Regression(file_paths=file_paths,
                               table_path=table_path,
                               target_variables=target_variables,
                               input_feature_columns=input_feature_columns,
                               first_heading=first_heading,
                               second_heading=second_heading,
                               splits=splits)

Linear_Reg.train()
