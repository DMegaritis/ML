from ML.LogisticRegressionTimeSeries import Logistic_Regression_TimeSeries
from ML.NN_ClassifierTimeSeries import NN_ClassifierTimeSeries

'''
Analysis for the BTS24 abstract.
'''


#%%
# Logistic Regression Specifying parameters
file_paths = [r'C:\Users\klch3\PycharmProjects\MLTable\data/aggregated_NIRS.csv']
table_path = r'C:\Users\klch3\PycharmProjects\MLTable\Result_Reg_TOI.csv'

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


#%%
# Neural Network Time Series Specifing parameters
file_paths = [r'C:\Users\klch3\PycharmProjects\MLTable\data/aggregated_NIRS.csv']
table_path = r'C:\Users\klch3\PycharmProjects\MLTable\Result_NN_TOI.csv'
target_variables = ['population']
input_feature_columns_continuous = ['TOI_1', 'TOI_2', 'TOI_3', 'TOI_4']
input_feature_columns_categorical = ['Comment']
first_heading = 'TOI'
second_heading = 'disease_group'
levels = 3
neurons = 500
splits = 5
scaler = 'yes'
early_stopping = False
group = ['ID']

NN_TimeSeries = NN_ClassifierTimeSeries(file_paths=file_paths,
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
    group=group)

NN_TimeSeries.train()
