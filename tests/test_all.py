import pytest
from ML.LinearRegression import Linear_Regression
from ML.LogisticRegression import Logistic_Regression
from ML.NN_Classifier import NN_Classifier
from ML.NN_Regression import NN_Regression


class TestLinearReg:

    def test_no_parameter_LinearReg(self):
        with pytest.raises(ValueError):
            Linear_Regression(file_paths=None,
                               table_path=None,
                               target_variables=None,
                               input_feature_columns=None,
                               first_heading=None,
                               second_heading=None,
                               splits=None)


class TestLogisticReg:

    def test_no_parameter_LogReg(self):
        with pytest.raises(ValueError):
            Logistic_Regression(file_paths=None,
                               table_path=None,
                               target_variables=None,
                               input_feature_columns=None,
                               first_heading=None,
                               second_heading=None,
                               splits=None)


class TestNN_Regression:

    def test_no_parameter_NNReg(self):
        with pytest.raises(ValueError):
            NN_Regression(file_paths=None,
                                table_path=None,
                                target_variables=None,
                                input_feature_columns=None,
                                first_heading=None,
                                second_heading=None,
                                levels=None,
                                neurons=None,
                                splits=None,
                                early_stopping=None)


class TestNN_Regression:

    def test_no_parameter_NNLinear(self):
        with pytest.raises(ValueError):
            NN_Classifier(file_paths=None,
                          table_path=None,
                          target_variables=None,
                          input_feature_columns=None,
                          first_heading=None,
                          second_heading=None,
                          levels=None,
                          neurons=None,
                          splits=None,
                          early_stopping=None)
