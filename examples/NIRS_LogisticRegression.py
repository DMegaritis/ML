from ML.LogisticRegressionTimeSeries import Logistic_Regression_TimeSeries

'''
Analysis for the BTS24 abstract.
'''

# Specifying parameters
file_paths = [r'C:\Users\klch3\PycharmProjects\ML\aggregated_NIRS.csv']
table_path = r'C:\Users\klch3\PycharmProjects\ML\Result_TOI.csv'

target_variables = ['population']
input_feature_columns_continuous = ['TOI_1', 'TOI_2', 'TOI_3', 'TOI_4']
input_feature_columns_categorical = ['Comment']
first_heading = 'TOI'
second_heading = 'disease_group'
splits = 5
group = ['ID']
scaler = "yes"

Log_Reg = Logistic_Regression_TimeSeries(file_paths=file_paths,
    table_path=table_path,
    target_variables=target_variables,
    input_feature_columns_continuous=input_feature_columns_continuous,
    input_feature_columns_categorical=input_feature_columns_categorical,
    first_heading=first_heading,
    second_heading=second_heading,
    splits=splits,
    scaler=scaler,
    group=group)

Log_Reg.train()
