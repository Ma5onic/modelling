""""
Contains functionality specific for the RV project
"""

import numpy as np

def mape(y_true, y_pred):
    """"
    Calculates the mean absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def duplicate_testset(testset, mileage_range, duration_range, splitting_variable, splitting_values):

    import pandas as pd
    from itertools import product
    
    testset['make_model']= testset['make'] + ' ' + testset['model_cons']
    make_model_size = pd.DataFrame({'size' : testset.groupby(by = 'make_model').size()})
    testset=pd.merge(testset,make_model_size,left_on='make_model',right_on=make_model_size.index)
    testset_nomileage_noduration = testset.drop(['mileage','duration','make_model',splitting_variable],axis=1)
    
    duplicated_columns = pd.DataFrame(list(product(testset['sf_objectid'],mileage_range, duration_range,splitting_values)), columns=['sf_objectid','mileage', 'duration',splitting_variable])
    duplicated_testset=pd.merge(duplicated_columns,testset_nomileage_noduration,on='sf_objectid',how='left')
    
    return duplicated_testset
