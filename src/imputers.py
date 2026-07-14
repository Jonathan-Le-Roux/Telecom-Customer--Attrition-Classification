from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import KNNImputer

class VoicemailImputer(BaseEstimator, TransformerMixin):
    '''
    Imputes missing values in the "Voicemail_Messages" column based on the "Voicemail_Plan" column.

    If "Voicemail_Plan" is "no", missing values in "Voicemail_Messages" are imputed with 0.

    If "Voicemail_Plan" is "yes", missing values in "Voicemail_Messages" can be imputed with 0, 
    the mean, or the median of the non-missing values in "Voicemail_Messages" based on the'''
    
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
    Imputes the missing values of each _Minutes column based on the corresponding 
    _Charge column and the average charge per minute for that time span.
    If both _Minutes and _Charge are missing, the imputer calculates the mean or median minutes per call
    '''


    ### TODO: can be calculated using DUration.

    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):   
        time_spans = ["Day", "Eve", "Night",'Intl']
        X = X.copy()

        for time in time_spans:
            time_minutes = f"{time}_Minutes"
            time_charge = f"{time}_Charge"
            time_calls = f"{time}_Calls"

            mask_not_na = X[time_minutes].notna()
            minute_rate = (X.loc[mask_not_na, time_charge] / X.loc[mask_not_na, time_minutes]).mean()

            #edge case: both night_charge and night_minutes are nan. 
            # In this case we calculate mean and median minutes per call and 
            # give the user the option to choose which one to use.

            edge_mask = (
                (X[time_minutes].isna()) &
                (X[time_charge].isna()) &
                (X[time_calls] > 0)
            )

            if self.strategy == 'mean':
                mean_minutes_per_call = (X.loc[mask_not_na, time_minutes] / X.loc[mask_not_na, time_calls]).mean()
                X.loc[edge_mask, time_minutes] = mean_minutes_per_call 
            elif self.strategy == 'median':
                median_minutes_per_call = (X.loc[mask_not_na, time_minutes] / X.loc[mask_not_na, time_calls]).median()
                X.loc[edge_mask, time_minutes] = median_minutes_per_call
            else:
                raise ValueError("Invalid strategy. Choose 'mean' or 'median'.")

            #base case
            mask_na = (X[time_minutes].isna())
            X.loc[mask_na, time_minutes] = X.loc[mask_na, time_charge ] / minute_rate
        
        return X
    
class ChargeImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):   
        time_spans = ["Day", "Eve", "Night",'Intl']
        X = X.copy()

        for time in time_spans:
            time_minutes = f"{time}_Minutes"
            time_charge = f"{time}_Charge"
            time_calls = f"{time}_Calls"

            mask_not_na = (X[time_charge].notna() & X[time_minutes]>0)
            
            charge_rate = (X.loc[mask_not_na, time_charge] / X.loc[mask_not_na, time_minutes]).mean()
        
            #edge case: both night_charge and night_minutes are nan.
            # In this case we calculate mean and median charge per call and
            # give the user the option to choose which one to use.      

            # may want to consider dropping these rows at a late stage
            edge_mask = (
                (X[time_minutes].isna()) &
                (X[time_charge].isna()) &
                (X[time_calls] > 0)
            )

            if self.strategy == 'mean':
                mean_minutes_per_call = (X.loc[mask_not_na, time_minutes] / X.loc[mask_not_na, time_calls]).mean()
                X.loc[edge_mask, time_charge] = mean_minutes_per_call * charge_rate
            elif self.strategy == 'median':
                median_minutes_per_call = (X.loc[mask_not_na, time_minutes] / X.loc[mask_not_na, time_calls]).median()
                X.loc[edge_mask, time_charge] = median_minutes_per_call * charge_rate
            else:
                raise ValueError("Invalid strategy. Choose 'mean' or 'median'.")
            
            #base case
            mask_na = (X[time_charge].isna())
            X.loc[mask_na, time_charge] = X.loc[mask_na, time_minutes] * charge_rate

        return X
    
class CallImputer(BaseEstimator, TransformerMixin):

    # number of calls may need to be computed using a different strategy than minutes and charge.
    # Should maybe use a kNN imputer or something similar to predict the number of calls based
    

    
    def fit(self, X, y=None):
        return self

    def transform(self, X):   
        time_spans = ["Day", "Eve", "Night",'Intl']
        X = X.copy()
        for time in time_spans:
            #this is going to be called last in the pipeline, so we can assume that minutes and charge have already been imputed.

            time_fields = [f"{time}_Minutes", f"{time}_Calls", f"{time}_Charge"]
            
            X_knn = X[time_fields].copy()
            knn_imputer = KNNImputer(n_neighbors=3)
            X_knn_imputed = knn_imputer.fit_transform(X_knn)
            X.loc[:, time_fields] = X_knn_imputed
        return X
    

class MasterImputer(BaseEstimator, TransformerMixin):
    def __init__(self, voicemail_strategy='zero', minute_strategy='mean', charge_strategy='mean'):
        self.voicemail_strategy = voicemail_strategy
        self.minute_strategy = minute_strategy
        self.charge_strategy = charge_strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = VoicemailImputer(strategy=self.voicemail_strategy).fit_transform(X)
        X = InternationalPlanImputer().fit_transform(X)
        X = MinuteImputer(strategy=self.minute_strategy).fit_transform(X)
        X = ChargeImputer(strategy=self.charge_strategy).fit_transform(X)
        X = CallImputer().fit_transform(X)
        return X