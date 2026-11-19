from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#TODO: must handle what to do in the case that both the charge and minutes are missing]

class VoicemailMessagesImputer(BaseEstimator, TransformerMixin):
    '''
    Imputes missing values in the "Voicemail_Messages" column based on the "Voicemail_Plan" column.

    If "Voicemail_Plan" is "no", missing values in "Voicemail_Messages" are imputed with 0.

    If "Voicemail_Plan" is "yes", missing values in "Voicemail_Messages" can be imputed with 0, 
    the mean, or the median of the non-missing values in "Voicemail_Messages" based on the strategy
    '''

    ##TODO: refine this method to use SimpleImputer
    
    def __init__(self, strategy='zero'):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        mask_neg = (
            (X["Voicemail_Plan"] == "no") &
            (X["Voicemail_Messages"].isna())
            )
         
        mask_pos = (
            (X["Voicemail_Plan"] == "yes") &
            (X["Voicemail_Messages"].isna())
            )
        
        X.loc[mask_neg, "Voicemail_Messages"] = 0

        if self.strategy == 'mean':
            mean_val = X.loc[X["Voicemail_Plan"] == "yes", "Voicemail_Messages"].mean()
            X.loc[mask_pos, "Voicemail_Messages"] = mean_val
        elif self.strategy == 'median':
            median_val = X.loc[X["Voicemail_Plan"] == "yes", "Voicemail_Messages"].median()
            X.loc[mask_pos, "Voicemail_Messages"] = median_val
        elif self.strategy == 'zero':
            X.loc[mask_pos, "Voicemail_Messages"] = 0
        else:
            raise ValueError("Invalid strategy. Choose 'zero', 'mean', or 'median'.")        
        return X
    

    pass
    # '''
    # Imputes the charges for the international charges using linear regression based on the data
    # '''
    # def fit(self,X,y = None):
    #     # Select rows where Intl_Charge is not missing
    #     train_data = X.dropna(subset=['Intl_Charge', 'Intl_Minutes'])

    #     X_train = train_data[['Intl_Minutes']]
    #     y_train = train_data['Intl_Charge']

    #     # Train the regression model
    #     self.lin_model = LinearRegression()
    #     self.lin_model.fit(X_train, y_train)

    #     return self

    # def transform(self,X):
    #     int_X = X.copy()

    #     # Identify rows with missing international charges
    #     missing = int_X['Intl_Charge'].isna()

    #     # Predict charges for rows with missing values
    #     X_missing = int_X.loc[missing, ['Intl_Minutes']]

    #     int_X.loc[missing, 'Intl_Charge'] = self.lin_model.predict(X_missing)

    #     return int_X

class InternationalPlanImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in the "Intl_Plan" column based on the "Intl_Minutes" and "Intl_Calls" columns.
    """
    
    def fit(self,X,y = None):
        return self
    
    def transform(self,X):
        X = X.copy()
        
        mask_pos= (
            (X["Intl_Plan"].isna()) &
            (X["Intl_Minutes"] > 0) |
            (X["Intl_Calls"] > 0)
            )
        
        X.loc[mask_pos, "Intl_Plan"] = "yes"
        X.loc[~mask_pos, "Intl_Plan"] = "no"
        return X
    
class MinuteImputer(BaseEstimator, TransformerMixin):
    '''
    Imputes the minutes based on the linear relationship between minutes and charges.
    
    '''
    
    def fit(self, X, y=None):

        self.time_spans = ["Day", "Eve", "Night", "Intl"]
        self.lin_models = {}

        for time in self.time_spans:
            charge_field = f"{time}_Charge"
            minutes_field = f"{time}_Minutes"

            # Select rows where both variables are available
            train_data = X.dropna(subset=[charge_field, minutes_field])

            X_train = train_data[[charge_field]]
            y_train = train_data[minutes_field]

            # Train a separate regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            self.lin_models[time] = model

        return self

    def transform(self, X):

        X = X.copy()

        for time in self.time_spans:
            charge_field = f"{time}_Charge"
            minutes_field = f"{time}_Minutes"

            # Identify missing minutes
            missing = X[minutes_field].isna()

            # Predict missing minutes using available charges
            X_missing = X.loc[missing, [charge_field]]

            if not X_missing.empty:
                X.loc[missing, minutes_field] = (
                    self.lin_models[time].predict(X_missing)
                )

        return X
    
class ChargeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.time_spans = ["Day", "Eve", "Night", "Intl"]
        self.lin_models = {}

        for time in self.time_spans:
            charge_field = f"{time}_Charge"
            minutes_field = f"{time}_Minutes"

            # Select rows where both variables are available
            train_data = X.dropna(subset=[charge_field, minutes_field])

            X_train = train_data[[minutes_field]]
            y_train = train_data[charge_field]

            # Train a separate regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            self.lin_models[time] = model
        return self

    def transform(self, X):
        X = X.copy()

        for time in self.time_spans:
            charge_field = f"{time}_Charge"
            minutes_field = f"{time}_Minutes"

            # Identify missing charges
            missing = X[charge_field].isna()

            # Predict missing charges
            X_missing = X.loc[missing, [minutes_field]]

            if not X_missing.empty:
                X.loc[missing, charge_field] = self.lin_models[time].predict(X_missing)
        return X
    
class CallImputer(BaseEstimator, TransformerMixin):
    '''
    As seen in analysis.ipynb we cannot use a more sophisticated imputation method such as 
    knn or linear regression. Since the qq-plot shows an approximate normal distribution,
    we can use the mean as an imputation method
    '''
    
    def fit(self, X, y=None):
        self.time_spans = ["Day", "Eve", "Night", "Intl"]
        self.imputers = {}

        for time in self.time_spans:
            call_field = [f"{time}_Calls"]

            imputer = SimpleImputer(strategy='mean')
            imputer.fit(X[call_field])

            self.imputers[time] = imputer

        return self

    def transform(self, X):   
        X = X.copy()

        for time in self.time_spans:
            call_field = [f"{time}_Calls"]

            X.loc[:, call_field] = self.imputers[time].transform(X[call_field])

        return X
    

class MasterImputer(BaseEstimator, TransformerMixin):
    def __init__(self, voicemail_strategy='zero'):
        self.voicemail_strategy = voicemail_strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = VoicemailMessagesImputer(strategy=self.voicemail_strategy).fit_transform(X)
        X = InternationalPlanImputer().fit_transform(X)
        X = ChargeImputer().fit_transform(X)
        X = MinuteImputer().fit_transform(X)
        X = CallImputer().fit_transform(X)
        return X